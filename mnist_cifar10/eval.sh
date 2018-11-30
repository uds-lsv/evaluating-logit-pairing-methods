#!/bin/bash

dataset='mnist'
data_dir='/tmp'
restore_path='/tmp'
log_dir='/tmp'
attack='PGD'
samples=1000
batch_size=1000
restarts=1

python eval.py \
  --dataset=${dataset} \
  --data_dir=${data_dir} \
  --restore_path=${restore_path} \
  --log_dir=${log_dir} \
  --attack=${attack} \
  --samples=${samples} \
  --batch_size=${batch_size} \
  --restarts=${restarts}

