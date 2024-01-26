import json
import pickle
from ws4py.client.threadedclient import WebSocketClient
import zmq
import time
import multiprocess as mp
import redis
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--redisip', type=str, default='192.168.1.100',
                    help='redis server')
parser.add_argument('--envip', type=str, default='192.168.1.100',
                    help='env server')
parser.add_argument('--guandanip', type=str, default='127.0.0.1',
                    help='guandan server')

args, _ = parser.parse_known_args()

def getaport():
    r = redis.Redis(host=f'{args.redisip}', port=6379)
    rediskey = 'allports'
    while True:
        if r.exists(rediskey):
            portnum = r.rpop(rediskey)
            while portnum is None:
                time.sleep(0.5)
                portnum = r.rpop(rediskey)
            return pickle.loads(portnum)
        else:
            time.sleep(2)

class GDClient(WebSocketClient):
    def __init__(self, url, pos, envport, args):
        super().__init__(url)
        self.restCards = None              # 剩余的卡牌
        self.episode_rounds = 0            # 当前小局玩的回合数
        self.myPos = pos                  # agent所处的位置
        self.msg = []
        self.envport = envport
        self.context = zmq.Context()
        self.context.linger = 0
        self.request = self.context.socket(zmq.REQ)
        self.request.connect(f'tcp://{args.envip}:{self.envport}')

    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, message):
        # print("original:",message)
        message = json.loads(str(message))                                    # 先序列化收到的消息，转为Python中的字典
        # self.state.parse(message)                                             # 调用状态对象来解析状态

        # 如果是开头，将我们有的卡牌存下来，并获取座位号
        # message type : dict
        if message["type"] == "notify":
            if message['stage'] == 'beginning':
                self.myPos = message['myPos']
            self.msg.append({'pos': self.myPos, 'msg': message})
            #print('received notify message:', self.msg)

        # 小局结束，回合数清零，并记录一下结果
        if message["type"] == "act":
            self.msg.append({'pos': self.myPos, 'msg': message, 'indexRange': message['indexRange']})
            p = pickle.dumps(self.msg)
            self.request.send(p)
            p = pickle.loads(self.request.recv())
            self.send(json.dumps(p))
            #self.environment.store_message(self.msg, self.myPos,int(message['indexRange']))
            self.msg = []   # 需要测试要不要清空


def run_one_client(index, envport, args):
    client = GDClient(f'ws://{args.guandanip}:23456/game/client{index}', index, envport, args)
    client.connect()
    client.run_forever()

def main():
    # 参数传递
    clients = []
    #while port is None:
    port = getaport()    # 检测是否有env启动
    print('get port: ', port)

    def exit_wrapper(index, port, *x, **kw):
        """Exit all actors on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_client(index, port, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(clients):
                    if _i != index:
                        _p.terminate()

    for i in range(4):
        p = mp.Process(target=exit_wrapper, args=(i, port, args))
        p.start()
        time.sleep(0.2)
        clients.append(p)

    for client in clients:
        client.join()

if __name__ == '__main__':
    main()
