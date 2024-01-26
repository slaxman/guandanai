#!/bin/bash
cd /data/guandan
cd guandanai

ulimit -n 10000

#nohup python broker.py &
echo 'del allports' | redis-cli

nohup python train.py --extra_cpu_actor --train_actor  >> train.log &

for i in {3..42}
do
        docker exec gdgame_$i /bin/bash -c "/data/guandan/guandanai/game/start.sh"
        echo $i
done

./startactor.sh &
#docker exec gdactor /bin/bash -c "/data/guandan/guandanai/startactor.sh"
