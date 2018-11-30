#!/usr/bin/env bash
### Runs evaluation of a trained model


# Path to tf.records with Tiny ImageNet
data_path="/scratch/maksym/tiny-imagenet-tfrecord"

# Specify the model folder
model_name="../models_github/LSQ"

# Examples of evaluation methods: clean (only on test examples), pgdrnd_16_2_10 (PGD attack with a random target,
# eps=16, step_size=2, n_iters=10), pgdll_16_4_400 (PGD attack with a least-likely target, eps=16, step_size=4,
# n_iters=400)
adv_method=clean

# Number of random restarts of the PGD attack
n_restarts=10

# Number of test examples to evaluate on
num_examples=1000

export CUDA_VISIBLE_DEVICES=0  # Specify the GPU number
python eval.py \
  --train_dir=${model_name} \
  --num_examples=${num_examples} \
  --n_restarts=${n_restarts} \
  --adv_method=${adv_method} \
  --tiny_imagenet_data_dir=${data_path}

