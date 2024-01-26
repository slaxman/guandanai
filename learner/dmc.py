import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn
from .file_writer import FileWriter
from models.model import Model
from .process import get_batch, create_buffers, create_optimizer, act
from envutil.utils import log
from torch import autocast

mean_episode_return_buf = deque(maxlen=100)

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss

def learn(actor_models,
          model,
          batch,
          optimizer,
          training_device,
          max_grad_norm,
          lock):
    """Performs a learning (optimization) step."""
    if training_device != "cpu":
        device = torch.device('cuda:' + str(training_device))
    else:
        device = torch.device('cpu')
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['target'][batch['done']]
    mean_episode_return_buf.append(torch.mean(episode_returns).to(device))
    with lock:
        with autocast(device_type='cuda', dtype=torch.float16):
            learner_outputs = model(obs_z, obs_x)
            loss = compute_loss(learner_outputs, target)
        stats = {
            'mean_episode_return': torch.mean(torch.stack([_r for _r in mean_episode_return_buf])).item(),
            'loss': loss.item(),
        }


        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model().load_state_dict(model.state_dict())
        return stats

free_queue = {}
full_queue = {}
models = {}
buffers = {}

def train(flags):
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")

    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'
    if flags.extra_cpu_actor:
        assert not flags.actor_device_cpu
        device_iterator = list(device_iterator)
        device_iterator.append('cpu')
    #if flags.train_actor:
    #    device_iterator = list(device_iterator)
    #    device_iterator.append(int(flags.training_device))

    num_actors = range(flags.num_actors)
    print('num actors: ',num_actors)
    print('device it:',device_iterator)

    for device in device_iterator:
        # model = Model(device=device)
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    # Initialize buffers
    buffers = create_buffers(T, flags.num_buffers, device_iterator, all_cpu=False)

    # Initialize queues
    ctx = mp.get_context('spawn')
    flags.supervise = False
    #pool = {}

    for device in device_iterator:
        free_queue[device] = ctx.SimpleQueue()
        full_queue[device] = ctx.SimpleQueue()

    # Learner model for training
    learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizer(flags.learning_rate, flags.momentum, flags.epsilon, flags.alpha, learner_model)

    First = False

    if First:
        # Stat Keys
        stat_keys = [
            'mean_episode_return_first',
            'loss_first',
            'mean_episode_return_second',
            'loss_second',
            'mean_episode_return_third',
            'loss_third',
            'mean_episode_return_forth',
            'loss_forth',
        ]
    else:
        stat_keys = [
            'mean_episode_return',
            'loss'
        ]

    frames, stats = 0, {k: 0 for k in stat_keys}
    saves = 0
    position_frames = 0

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):

        checkpoint_states = torch.load(
            checkpointpath,
            map_location=("cuda:" + str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        # for k in ['first', 'second', 'third', 'forth']:
        # for the first time to load old model data, only

        if First:
            k = 'first'
            learner_model.get_model().load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers.load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                if not flags.rule_generator:
                    models[device].get_model().load_state_dict(learner_model.get_model().state_dict())
                else:
                    models[device] = None
            stats = checkpoint_states["stats"]
            # print('stats=',stats)
            oldstats = {}
            oldstats['mean_episode_return'] = stats['mean_episode_return_first']
            oldstats['loss'] = stats['loss_first']
            stats = oldstats
            frames = checkpoint_states["frames"]
            saves = checkpoint_states["saves"]
            position_frames = checkpoint_states["position_frames"]
            position_frames = position_frames['first']
            log.info(f"Resuming preempted job, current stats:\n{stats}")
        else:
            learner_model.get_model().load_state_dict(checkpoint_states["model_state_dict"])
            optimizers.load_state_dict(checkpoint_states["optimizer_state_dict"])
            for device in device_iterator:
                if not flags.rule_generator:
                    models[device].get_model().load_state_dict(learner_model.get_model().state_dict())
                else:
                    models[device] = None
            stats = checkpoint_states["stats"]
            frames = checkpoint_states["frames"]
            saves = checkpoint_states["saves"]
            position_frames = checkpoint_states["position_frames"]
            log.info(f"Resuming preempted job, current stats:\n{stats}")


        # Starting actor processes

    threads_actor = []

    seq = 0
    print('device iterateor:', device_iterator)
    for device in device_iterator:
        if device == 'cpu' and flags.extra_cpu_actor:
            num_actors = range(0,flags.extra_cpu_actor_num)
        if device == int(flags.training_device) and flags.train_actor:
            num_actors = range(0,flags.train_actor_num)
        for i in num_actors:
            print('.. start actor server ..', device,num_actors)
            thread = ctx.Process(
                target=act,
                args=(
                seq, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            thread.start()
            threads_actor.append(thread)
            seq += 1


    def batch_and_learn(i, device, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device], full_queue[device], buffers[device],
                                  flags.batch_size, local_lock)
            _stats = learn(models, learner_model.get_model(), batch,
                           optimizers, flags.training_device, flags.max_grad_norm, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                #to_log = frames
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = threading.Lock()
    position_locks =threading.Lock()

    for device in device_iterator:
        for i in range(flags.num_threads):
            # for position in ['first', 'second', 'third', 'forth']:
            thread = threading.Thread(
                target=batch_and_learn, name='batch-and-learn-%d' % i,
                args=(i, device, locks[device], position_locks))
            thread.start()
            threads.append(thread)

    def checkpoint(frames, saves):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _model = learner_model.get_model()

        torch.save({
            'model_state_dict': _model.state_dict(),
            'optimizer_state_dict': optimizers.state_dict(),
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames,
            'saves': saves
        }, checkpointpath)

        # Save the weights for evaluation purpose
        model_weights_dir = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid,  '_weights_' + str(saves) + '.tar')))
        torch.save({
            'model_state_dict': _model.state_dict(),
            'frames': frames,
            'saves': saves
        }, model_weights_dir)
        '''for position in ['first', 'second', 'third', 'forth']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position + '_weights_' + str(saves) + '.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)'''

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 30
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(60)

            if timer() - last_checkpoint_time > flags.save_interval * 30:
                checkpoint(frames, saves)
                saves += 1
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            log.info(
                'After %i frames: @ %.1f fps (avg@ %.1f fps)  Stats:\n%s',
                frames,
                fps,
                fps_avg,
                pprint.pformat(stats))

    except KeyboardInterrupt:
        #print('begin to save...')
        #with open(pool_path, 'wb') as f:
        #    pickle.dump(pool, f)
        #print('save success!')
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames, saves)
    plogger.close()
