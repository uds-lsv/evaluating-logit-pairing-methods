#!/bin/bash

dataset='mnist'
data_dir='/tmp'
save_dir='/tmp'
log_dir='/tmp'
epochs=100

python train.py \
  --dataset=${dataset} \
  --data_dir=${data_dir} \
  --save_dir=${save_dir} \
  --log_dir=${log_dir} \
  --epochs=${epochs}