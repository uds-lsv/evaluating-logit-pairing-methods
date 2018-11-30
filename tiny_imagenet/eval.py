# Copyright 2018 Google Inc. All Rights Reserved.
# Modifications copyright (C) 2018, Maksym Andriushchenko <m.andriushchenko@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Program which runs evaluation of Tiny Imagenet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow as tf

import adversarial_attack
import model_lib
import time
import numpy as np
from datasets import dataset_factory

FLAGS = flags.FLAGS


flags.DEFINE_string('train_dir', None,
                    'Directory with a checkpoint file to restore a model.')

flags.DEFINE_string('output_file', None,
                    'Name of output file. Used only in single evaluation mode.')

flags.DEFINE_string('eval_name', 'default', 'Name for eval subdirectory.')

flags.DEFINE_string('model_name', 'resnet_v2_50', 'Name of the model.')

flags.DEFINE_string('adv_method', 'clean',
                    'Method which is used to generate adversarial examples.')

flags.DEFINE_integer('n_restarts', 1,
                     'Number of restarts for the PGD attack.')

flags.DEFINE_string('hparams', '', 'Hyper parameters.')

flags.DEFINE_float('moving_average_decay', 0.9999,
                   'The decay to use for the moving average.')

flags.DEFINE_integer(
    'num_examples', -1,
    'The number of example to use for evaluation. Note that all of them should fit in 1 GPU.')


def get_latest_checkpoint(train_dir):
  ckpt_files = [train_dir + '/' + fname for fname in os.listdir(train_dir) if 'ckpt' in fname]
  last_mtime, last_fname = 0.0, ''
  for cur_fname in ckpt_files:
    cur_mtime = os.stat(cur_fname).st_mtime  # modification time
    if cur_mtime > last_mtime:
      last_mtime = cur_mtime
      last_fname = cur_fname
  last_fname = last_fname.replace('.meta', '').replace('.index', '').replace('.data-00000-of-00001', '')
  return last_fname


def main(_):
  num_examples = FLAGS.num_examples
  params = model_lib.default_hparams()
  params.parse(FLAGS.hparams)
  tf.logging.info('User provided hparams: %s', FLAGS.hparams)
  tf.logging.info('All hyper parameters: %s', params)
  graph = tf.Graph()
  with graph.as_default():
    img_bounds = (-1, 1) if 'Fine-tuned' in FLAGS.train_dir else (0, 1)
    dataset, total_num_examples, num_classes, bounds = dataset_factory.get_dataset(
        'tiny_imagenet',
        'validation',
        10000,
        64,
        is_training=False,
        bounds=img_bounds)
    dataset_iterator = dataset.make_one_shot_iterator()
    images, labels = dataset_iterator.get_next()
    images, labels = images[:num_examples], labels[:num_examples]

    # setup model
    global_step = tf.train.get_or_create_global_step()
    model_fn_two_args = model_lib.get_model(FLAGS.model_name, num_classes)
    model_fn = lambda x: model_fn_two_args(x, is_training=False)  # thin wrapper; args: images, is_training

    # clean first
    clean_logits = model_fn(images)

    labels_for_ae = tf.identity(labels)  # used only if 'rnd' in FLAGS.adv_method and FLAGS.n_restarts > 0
    if FLAGS.adv_method == 'clean':
      logits = clean_logits
    else:
      adv_examples = adversarial_attack.generate_adversarial_examples(
          images, labels_for_ae, bounds, model_fn, FLAGS.adv_method, n_restarts=FLAGS.n_restarts)
      adv_logits = model_fn(adv_examples)
      logits = adv_logits

    correct = tf.equal(tf.argmax(logits, 1), tf.to_int64(labels))
    # correct = tf.equal(tf.argmax(adv_logits, 1), tf.argmax(clean_logits, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Setup the moving averages
    variable_averages = tf.train.ExponentialMovingAverage(
      FLAGS.moving_average_decay, global_step)
    variables_to_restore = variable_averages.variables_to_restore(
      tf.contrib.framework.get_model_variables())
    variables_to_restore[global_step.op.name] = global_step

    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(graph=graph, config=config) as sess:
      saver.restore(sess, get_latest_checkpoint(FLAGS.train_dir))
      correct_vals = np.zeros([FLAGS.n_restarts, num_examples])
      time_start = time.time()
      # we select random classes only once, and fix them across multiple restarts
      labels_for_rnd_multrest = np.random.random_integers(0, num_classes-1, size=[num_examples, ])
      for i_restart in range(FLAGS.n_restarts):
        logits_val, correct_val, acc_val = sess.run([logits, correct, acc], feed_dict={labels_for_ae: labels_for_rnd_multrest})
        print('Accuracy: {:.2%}'.format(acc_val))
        correct_vals[i_restart, :] = correct_val

    print('[Elapsed {:.2f} min] avg_acc={:.2%}  min_acc={:.2%}  (n_restarts={})'.
          format((time.time() - time_start) / 60, correct_vals.mean(), correct_vals.min(axis=0).mean(), FLAGS.n_restarts))


if __name__ == '__main__':
  app.run(main)
