import pickle
import time
from argparse import ArgumentParser
import multiprocess as mp
import zmq
from game.rule import ruleAgent
from envutil.utils import log
from game.wrapenv import GuanDanEnv
import redis

parser = ArgumentParser()
parser.add_argument('--ip', type=str, default='192.168.1.100',
                    help='IP address of learner server')
parser.add_argument('--redisip', type=str, default='192.168.1.100',
                    help='IP address of redis server')

parser.add_argument('--bip', type=str, default='192.168.1.100',
                    help='IP address of broker server')
parser.add_argument('--broker_port', type=int, default=12000,
                    help='broker address')


def clean_port(args):
    r = redis.Redis(host=f'{args.redisip}', port=6379)
    rediskey = 'allports'
    r.delete(rediskey)

class Actor():
    def __init__(self, actor_id, args) -> None:
        self.args = args
        self.count = 0
        self.actor_id = actor_id
        self.env = GuanDanEnv(self.actor_id)

        # 初始化zmq
        self.context = zmq.Context()
        self.context.linger = 0
        # to learner
        #self.socket = self.context.socket(zmq.DEALER)
        #self.socket.setsockopt_string(zmq.IDENTITY, str(actor_id))
        #self.socket.connect(f'tcp://{args.bip}:{args.broker_port}')
        self.socket = self.context.socket(zmq.REQ)
        # self.socket.setsockopt_string(zmq.IDENTITY, str(actor_id))
        self.socket.connect(f'tcp://{args.bip}:{5000+self.actor_id}')

    def process(self):
        log.info('Game %i started.', self.actor_id)
        env = GuanDanEnv(self.actor_id)
        rule = [ruleAgent(i) for i in range(4)]

        for rule_agent in rule:
            rule_agent.reset()

        while True:
            state = env.reset(new_start=True)
            while True:
                for rule_agent in rule:
                    rule_agent.reset()
                while not state.terminal:
                    stage = state.stage
                    if stage == 'tribute':
                        msg = env.get_message()
                        for rule_agent in rule:
                            rule_agent.decode(msg, tri_back_mode=True)
                        #print('playerid:', env.current_player_id)
                        action_index = rule[env.current_player_id].step(msg)
                        state = env.stepto(action_index)
                        self.send_learner(state)
                    elif stage == 'back':
                        msg = env.get_message()
                        for rule_agent in rule:
                            rule_agent.decode(msg)
                        #print('playerid:', env.current_player_id)
                        action_index = rule[env.current_player_id].step(msg)
                        state = env.stepto(action_index)
                        self.send_learner(state)
                    else:
                        assert stage == 'play'
                        action_index = self.send_learner(state)
                        state = env.stepto(action_index)

                self.send_learner(state)
                if not state.stage == 'gameOver':
                    state = env.reset(new_start=False)
                else:
                    break
                    pass

    def send_learner(self,state):
        p = pickle.dumps(state,protocol=pickle.HIGHEST_PROTOCOL)
        self.socket.send(p)
        p = self.socket.recv()
        return pickle.loads(p)

def run_one_actor(index, args):
    actor = Actor(index, args)
    # 初始化zmq
    actor.process()

def main():
    # 参数传递
    args, _ = parser.parse_known_args()
    clean_port(args)

    def exit_wrapper(index, *x, **kw):
        """Exit all actors on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_actor(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(actors):
                    if _i != index:
                        _p.terminate()

    actors = []
    for i in range(0,40):
        print(i)
        p = mp.Process(target=exit_wrapper, args=(i, args))
        p.start()
        time.sleep(0.5)
        actors.append(p)

    for actor in actors:
        actor.join()

if __name__ == '__main__':
    main()
