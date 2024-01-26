import zmq
import time
from collections import OrderedDict

context = zmq.Context.instance()
frontend = context.socket(zmq.ROUTER)
frontend.bind("tcp://*:12000")
backend = context.socket(zmq.ROUTER)
backend.bind("tcp://*:13000")
frontend.setsockopt(zmq.RCVHWM, 100)
backend.setsockopt(zmq.RCVHWM, 100)

workers = OrderedDict()
clients = {}
msg_cache = []
poll = zmq.Poller()

poll.register(backend, zmq.POLLIN)
poll.register(frontend, zmq.POLLIN)

if __name__ == '__main__':
    while True:
        socks = dict(poll.poll(30))
        now = time.time()
        # 接收后端消息
        if backend in socks and socks[backend] == zmq.POLLIN:
            # 接收后端地址、客户端地址、后端返回response  ps: 此处的worker_addr, client_addr, reply均是bytes类型
            worker_addr, client_addr, response = backend.recv_multipart()
            # 把后端存入workers
            workers[worker_addr] = time.time()
            if client_addr in clients:
                # 如果客户端地址存在,把返回的response转发给客户端,并删除客户端
                frontend.send_multipart([client_addr, response])
                clients.pop(client_addr)
            #else:
            # 客户端不存在
            #print(worker_addr, client_addr)
        # 处理所有未处理的消息
        while len(msg_cache) > 0 and len(workers) > 0:
            # 取出一个最近通信过的worker
            worker_addr, t = workers.popitem()
            # 判断是否心跳过期 过期则重新取worker
            if t - now > 1:
                continue
            msg = msg_cache.pop(0)
            # 转发缓存的消息
            backend.send_multipart([worker_addr, msg[0], msg[1]])
        # 接收前端消息
        if frontend in socks and socks[frontend] == zmq.POLLIN:
            # 获取客户端地址和请求内容  ps: 此处的client_addr, request均是bytes类型
            client_addr, request = frontend.recv_multipart()
            clients[client_addr] = 1
            while len(workers) > 0:
                # 取出一个最近通信过的worker
                worker_addr, t = workers.popitem()
                # 判断是否心跳过期 过期则重新取worker
                if t - now > 1:
                    continue
                # 转发消息
                backend.send_multipart([worker_addr, client_addr, request])
                break
            else:
                # while正常结束说明消息未被转发,存入缓存
                msg_cache.append([client_addr, request])