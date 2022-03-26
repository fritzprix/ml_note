#!/bin/sh

jt -t grade3
tensorboard --logdir /home/tf_logs &
xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root