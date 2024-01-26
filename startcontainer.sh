#!/bin/bash
# learner docker
docker run -itd --gpus all  --network=guandanNet --ip 172.15.15.2 --name learner -v /data/guandan/:/data/guandan -w /data/guandan/ nvcr.io/nvidia/tensorflow:22.02-tf1-py3
#启动actor
for i in {3..42}
do
  docker run -itd --network=guandanNet --ip 172.15.15.$i --name gdgame_$i -v /data/guandan:/data/guandan -w /data/guandan  gdgame /bin/bash
done

#docker run -itd --network=guandanNet --ip 172.15.15.2 --name gdactor -v /data/guandan:/data/guandan -w /data/guandan  gdgame /bin/bash
