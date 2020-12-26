#!/bin/sh

jt -t grade3
tensorboard --logdir /home/tf_logs &
jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root