#!/bin/bash
#for i in {35..80}
#do
#    docker stop gdgame_$i
#    docker rm gdgame_$i
#done

docker stop learner
docker rm learner

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)