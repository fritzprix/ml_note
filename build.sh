#!/bin/bash
docker build -t pytorch_lab:0.1.0 .
mkdir data tf_logs work > /dev/null 2>&1