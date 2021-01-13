#!/bin/bash

docker rm tf_lab > /dev/null 2>&1

docker run --gpus=all --runtime=nvidia -it -p 8888:8888 -p 6006:6006 --name tf_lab --mount type=bind,source="$(pwd)/work",target="/home/work" \
 --mount type=bind,source="$(pwd)/data",target="/home/data" \
 --mount type=bind,source="$(pwd)/tf_logs",target="/home/tf_logs" \
 tf_lab:0.1.0 


# docker run -it -p 8888:8888 -p 6006:6006 --name tf_lab --mount type=bind,source="$(pwd)/work",target="/home/work" \
#  --mount type=bind,source="$(pwd)/data",target="/home/data" \
#  --mount type=bind,source="$(pwd)/tf_logs",target="/home/tf_logs" \
#  tf_lab:0.1.0 

docker rm tf_lab > /dev/null 2>&1
echo "Goodbye!"