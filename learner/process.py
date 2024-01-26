import typing
import torch
import zmq
import pickle
import traceback
from .card2array import action_one_hot_code
from envutil.utils import log
from .env_utils import WrapEnv

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_buffers(unroll_length, num_buffers, device_iterator, all_cpu=False):
    T = unroll_length
    buffers = {}
    # positions = ['first', 'second', 'third', 'forth']
    # for actor in [num_actors]:
    for device in device_iterator:
        buffers[device] = {}
        x_dim = 1172
        specs = dict(
            done=dict(size=(T,), dtype=torch.bool),
            target=dict(size=(T,), dtype=torch.float32),
            obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
            obs_action=dict(size=(T, 133), dtype=torch.int8),
            obs_z=dict(size=(T, 5, 532), dtype=torch.int8),
        )
        _buffers: Buffers = {key: [] for key in specs}
        for _ in range(num_buffers):
            for key in _buffers:
                if all_cpu:
                    _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                elif not device == "cpu":
                    _buffer = torch.empty(**specs[key]).to(torch.device('cuda:' + str(device))).share_memory_()
                else:
                    _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                _buffers[key].append(_buffer)
        buffers[device] = _buffers
    return buffers

def get_batch(free_queue,
              full_queue,
              buffers,
              batch_size,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(batch_size)]
        #print('get_batch indices:',indices)
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        #print('m=',m)
        free_queue.put(m)
    return batch


# def get_batch_from_pool(free_queue,
#                         full_queue,
#                         pool,
#                         buffers,
#                         batch_size,
#                         lock):
#     """
#     This function will sample a batch from the buffers based
#     on the indices received from the full queue. It will also
#     free the indices by sending it to full_queue.
#     """
#     with lock:
#         if np.random.rand() < 0.80 and len(pool) > 50:
#             indices = None
#         else:
#             indices = [full_queue.get() for _ in range(batch_size)]
#             #print('get_batch_pool indices:',indices)
#     if indices is not None:
#         batch = {
#             key: torch.stack([buffers[key][m] for m in indices], dim=1)
#             for key in buffers
#         }
#         for m in indices:
#             free_queue.put(m)
#         pool.append(copy.deepcopy(batch))
#     else:
#         batch = random.choice(pool)
#     return batch


def create_optimizer(learning_rate, momentum, epsilon, alpha, learner_model):
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        eps=epsilon,
        alpha=alpha)
    return optimizer

def act(actor_id, device, free_queue, full_queue, model, buffers, flags):
    positions = ['first', 'second', 'third', 'forth']
    try:
        T = flags.unroll_length
        exp_epsilon = flags.exp_epsilon

        log.info('Device Actor Backend Thread %i started.', actor_id)
        done_buf = {p: [] for p in positions}
        # episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        context = zmq.Context()
        context.linger = 0  # For removing linger behavior
        socket = context.socket(zmq.REP)
        # socket.setsockopt(zmq.RCVTIMEO, 10000)
        print('listen on :', f'tcp://{flags.wip}:{5000+actor_id}')
        socket.bind(f'tcp://{flags.wip}:{5000+actor_id}')
        env = None
        count = 1

        while True:
            message = socket.recv()
            # 处理任务
            state = pickle.loads(message)
            # actor_id = state.actor_id

            if env is None:
                env = WrapEnv(device)
                position, obs, env_output = env.reset(state)
            else:
                position, obs, env_output = env.step(state)
            # here to process episode
            if env_output['done']:
                socket.send(pickle.dumps(b'ok'))
                for p in positions:
                    diff = size[p] - len(target_buf[p])
                    if diff > 0:
                        done_buf[p].extend([False for _ in range(diff - 1)])
                        done_buf[p].append(True)
                        pos_id = env.get_position_player_id(p)
                        episode_return = env_output['episode_return'][pos_id]
                        target_buf[p].extend([episode_return for _ in range(diff)])

                for p in positions:
                    while size[p] > T:
                        index = free_queue.get()
                        if index is None:
                            break
                        for t in range(T):
                            buffers['done'][index][t, ...] = done_buf[p][t]
                            # buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                            buffers['target'][index][t, ...] = target_buf[p][t]
                            buffers['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                            buffers['obs_action'][index][t, ...] = obs_action_buf[p][t]
                            buffers['obs_z'][index][t, ...] = obs_z_buf[p][t]
                        full_queue.put(index)
                        done_buf[p] = done_buf[p][T:]
                        # episode_return_buf[p] = episode_return_buf[p][T:]
                        target_buf[p] = target_buf[p][T:]
                        obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                        obs_action_buf[p] = obs_action_buf[p][T:]
                        obs_z_buf[p] = obs_z_buf[p][T:]
                        size[p] -= T
                env.first_player = None
                count += 1

            elif not env_output['done']:
                stage = obs['stage']
                assert stage == 'play'
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                with torch.inference_mode():
                    action_index = model.step(position, obs['z_batch'], obs['x_batch'], exp_epsilon=exp_epsilon)
                action_index = int(action_index.cpu().detach().numpy())
                # send to game
                socket.send(pickle.dumps(action_index, pickle.HIGHEST_PROTOCOL))
                action = obs['legal_actions'][action_index]
                action_tensor = torch.from_numpy(action_one_hot_code(action))
                obs_action_buf[position].append(action_tensor)
                size[position] += 1

    except Exception as e:
        log.error('Exception in worker process %i', )
        traceback.print_exc()
        raise e
