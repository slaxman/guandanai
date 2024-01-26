#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
conda activate guandan
#pip install redis
nohup /data/guandan/guandan 10000000000 >/dev/null 2>&1 &
cd /data/guandan/guandanai
#nohup /data/guandan/guandan 10000000000 >> /data/guandan/game_out.log 2>&1 &
nohup python -u game/game.py >> /data/guandan/guandanai/process.log &
