#!/usr/bin/env bash

eval_id=3
model_name=models_eval${eval_id}/baseline_nolabelsmooth_adam
short_model_name=`echo ${model_name} | cut -d'/' -f 2`
printf "${model_name} ${short_model_name}\n"


gpu=0
adv_method=clean  # clean, pgdll_16_2_10, pgdll_16_1_20, pgdll_16_0.5_400, pgdll_16_0.1_1000
n_restarts=1
num_examples=10000
export CUDA_VISIBLE_DEVICES=${gpu}
nohup python eval.py \
    --train_dir=${model_name} \
    --num_examples=${num_examples} \
    --n_restarts=${n_restarts} \
    --dataset=tiny_imagenet \
    --dataset_image_size=64 \
    --adv_method=${adv_method} \
    --hparams="eval_batch_size=10000" \
    --eval_once=True \
    --tiny_imagenet_data_dir="/scratch/maksym/tiny-imagenet-tfrecord" >> eval${eval_id}/${short_model_name}__${adv_method}.out &


gpu=1
adv_method=pgdrnd_16_2.0_10  # clean, pgdll_16_2_10, pgdll_16_1_20, pgdll_16_0.5_400, pgdll_16_0.1_1000
n_restarts=1
num_examples=1000
export CUDA_VISIBLE_DEVICES=${gpu}
nohup python eval.py \
    --train_dir=${model_name} \
    --num_examples=${num_examples} \
    --n_restarts=${n_restarts} \
    --dataset=tiny_imagenet \
    --dataset_image_size=64 \
    --adv_method=${adv_method} \
    --hparams="eval_batch_size=1000" \
    --eval_once=True \
    --tiny_imagenet_data_dir="/scratch/maksym/tiny-imagenet-tfrecord" >> eval${eval_id}/${short_model_name}__${adv_method}.out &


gpu=2
adv_method=pgdll_16_2.0_10  # clean, pgdll_16_2_10, pgdll_16_1_20, pgdll_16_0.5_400, pgdll_16_0.1_1000
n_restarts=1
num_examples=1000
export CUDA_VISIBLE_DEVICES=${gpu}
nohup python eval.py \
    --train_dir=${model_name} \
    --num_examples=${num_examples} \
    --n_restarts=${n_restarts} \
    --dataset=tiny_imagenet \
    --dataset_image_size=64 \
    --adv_method=${adv_method} \
    --hparams="eval_batch_size=1000" \
    --eval_once=True \
    --tiny_imagenet_data_dir="/scratch/maksym/tiny-imagenet-tfrecord" >> eval${eval_id}/${short_model_name}__${adv_method}.out &


gpu=3
adv_method=pgdrnd_16_4.0_400  # clean, pgdll_16_2_10, pgdll_16_1_20, pgdll_16_0.5_400, pgdll_16_0.1_1000
n_restarts=100
num_examples=1000
export CUDA_VISIBLE_DEVICES=${gpu}
nohup python eval.py \
    --train_dir=${model_name} \
    --num_examples=${num_examples} \
    --n_restarts=${n_restarts} \
    --dataset=tiny_imagenet \
    --dataset_image_size=64 \
    --adv_method=${adv_method} \
    --hparams="eval_batch_size=1000" \
    --eval_once=True \
    --tiny_imagenet_data_dir="/scratch/maksym/tiny-imagenet-tfrecord" >> eval${eval_id}/${short_model_name}__${adv_method}.out &


gpu=4
adv_method=pgdll_16_4.0_400  # clean, pgdll_16_2_10, pgdll_16_1_20, pgdll_16_0.5_400, pgdll_16_0.1_1000
n_restarts=100
num_examples=1000
export CUDA_VISIBLE_DEVICES=${gpu}
nohup python eval.py \
    --train_dir=${model_name} \
    --num_examples=${num_examples} \
    --n_restarts=${n_restarts} \
    --dataset=tiny_imagenet \
    --dataset_image_size=64 \
    --adv_method=${adv_method} \
    --hparams="eval_batch_size=1000" \
    --eval_once=True \
    --tiny_imagenet_data_dir="/scratch/maksym/tiny-imagenet-tfrecord" >> eval${eval_id}/${short_model_name}__${adv_method}.out &

