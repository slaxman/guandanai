#!/bin/bash
#. /opt/conda/etc/profile.d/conda.sh
#conda activate guandan
#pip install redis
cd /data/guandan/guandanai
nohup python -u actor.py >> ./process.log &
