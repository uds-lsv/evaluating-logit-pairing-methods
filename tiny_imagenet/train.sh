#!/usr/bin/env bash
### Runs synchronous training on 8 GPUs within the same server (starting from 0th GPU)


# Path to tf.records with Tiny ImageNet
data_path="/scratch/maksym/tiny-imagenet-tfrecord"

# 100 epochs with batch size=256 means 100000/256 * 100 = 39062 steps
max_steps=40000

# Note, that the model name matters defines the method used. If it contains 'Plain' substring, the plain loss will be
# used for training. If it contains '50% AT', adv. training with 50% adv. examples will be used. If it contains
# '100% AT', adv. training with 100% adv. examples will be used.
model_name="50% AT LL + ALP LL"

# How to generate adv. examples for adv. training: clean (only on training examples),
# pgdrnd_16_2_10 (PGD attack with a random target, eps=16, step_size=2, n_iters=10),
# pgdll_16_2_10 (PGD attack with a least-likely target, eps=16, step_size=2, n_iters=10)
adv_method=pgdll_16_2_10  #  clean, pgd_16_2_10, pgdll_16_2_10, pgdrnd_16_2_10

# strength of LSQ/CLP/ALP regularization
train_lp_weight=0.5

gpu_ids="0 1 2 3 4 5 6 7"

n_tasks=8  # total number of workers (should match to the number of `gpu_ids`)

# There are also other options, e.g.: --add_noise or --logit_squeezing. See `python train.py --helpfull` for details.

# First run a parameter server on CPU
export CUDA_VISIBLE_DEVICES=
nohup python train.py \
  --output_dir="models/${model_name}" \
  --task_ids="${gpu_ids}" \
  --job_name=ps \
  --task=0 \
  --worker_replicas=${n_tasks} \
  --replicas_to_aggregate=${n_tasks} &

# Then start 8 workers on 8 GPUs
# note that batch_size=32 is per GPU, so the effective batch size will be 32*8=256 images
for gpu_id in ${gpu_ids}; do
    task_id=$(( $gpu_id % $n_tasks ))
    echo $gpu_id $task_id
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    nohup python train.py \
      --max_steps=${max_steps} \
      --model_name="resnet_v2_50" \
      --hparams="train_adv_method=${adv_method},train_lp_weight=${train_lp_weight},batch_size=32" \
      --output_dir="models/${model_name}" \
      --tiny_imagenet_data_dir=${data_path} \
      --task_ids="${gpu_ids}" \
      --job_name=worker \
      --task=${task_id} \
      --worker_replicas=${n_tasks} \
      --replicas_to_aggregate=${n_tasks} &
done

