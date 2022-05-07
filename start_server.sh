#!/bin/sh

jt -t grade3
xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root