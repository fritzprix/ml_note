#!/bin/bash

docker rm pytorch_lab > /dev/null 2>&1

docker run -it -p 8888:8888 -p 6006:6006 --name pytorch_lab --mount type=bind,source="$(pwd)/work",target="/home/work" \
 --mount type=bind,source="$(pwd)/data",target="/home/data" \
 --mount type=bind,source="$(pwd)/tf_logs",target="/home/tf_logs" \
 pytorch_lab:0.1.0 

docker rm pytorch_lab > /dev/null 2>&1
echo "Goodbye!"