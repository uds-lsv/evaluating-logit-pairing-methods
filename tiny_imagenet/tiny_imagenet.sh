#!/usr/bin/env bash
### This script is used to download Tiny ImageNet dataset and convert it to tf-records format

# Download zip archive with TinyImagenet
curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip

# Extract archive
unzip tiny-imagenet-200.zip

# Convert dataset to TFRecord format
mkdir tiny-imagenet-tfrecord
python tiny_imagenet_converter/converter.py \
  --input_dir=tiny-imagenet-200 \
  --output_dir=tiny-imagenet-tfrecord


