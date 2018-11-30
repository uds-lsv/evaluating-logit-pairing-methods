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

"""Program which train models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow as tf

import adversarial_attack
import model_lib
from datasets import dataset_factory

FLAGS = flags.FLAGS


flags.DEFINE_bool('add_noise', False,
                  'If true then add the gaussian noise of eps=16/255.')

flags.DEFINE_bool('logit_squeezing', False,
                  'If true then adde logit squeezing regularizer.')

flags.DEFINE_integer('max_steps', -1, 'Number of steps to stop at.')

flags.DEFINE_string('output_dir', None,
                    'Training directory where checkpoints will be saved.')

flags.DEFINE_integer('ps_tasks', 0, 'Number of parameter servers.')

flags.DEFINE_integer('task', 0, 'Task ID for running distributed training.')

flags.DEFINE_string('job_name', '', '[custom] job name: ps or worker.')

flags.DEFINE_string('task_ids', '', 'TF task ids.')

flags.DEFINE_string('model_name', 'resnet_v2_50', 'Name of the model.')

flags.DEFINE_integer('num_summary_images', 3,
                     'Number of images to display in Tensorboard.')

flags.DEFINE_integer(
  'save_summaries_steps', 100,
  'The frequency with which summaries are saved, in steps.')

flags.DEFINE_integer(
  'save_summaries_secs', None,
  'The frequency with which summaries are saved, in seconds.')

flags.DEFINE_integer(
  'save_model_steps', 2000,
  'The frequency with which the model is saved, in steps.')

flags.DEFINE_string('hparams', '', 'Hyper parameters.')

flags.DEFINE_integer('replicas_to_aggregate', 1,
                     'Number of gradients to collect before param updates.')

flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_float('moving_average_decay', 0.9999,
                   'The decay to use for the moving average.')

# Flags to control fine tuning
flags.DEFINE_string('finetune_checkpoint_path', None,
                    'Path to checkpoint for fine tuning. '
                    'If None then no fine tuning is done.')

flags.DEFINE_string('finetune_exclude_pretrained_scopes', '',
                    'Variable scopes to exclude when loading checkpoint for  '
                    'fine tuning.')

flags.DEFINE_string('finetune_trainable_scopes', None,
                    'If set then it defines list of variable scopes for '
                    'trainable variables.')


def _get_finetuning_init_fn(variable_averages):
  """Returns an init functions, used for fine tuning."""
  if not FLAGS.finetune_checkpoint_path:
    return None

  if tf.train.latest_checkpoint(FLAGS.output_dir):
    return None

  if tf.gfile.IsDirectory(FLAGS.finetune_checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.finetune_checkpoint_path)
  else:
    checkpoint_path = FLAGS.finetune_checkpoint_path

  if not checkpoint_path:
    tf.logging.warning('Not doing fine tuning, can not find checkpoint in %s',
                       FLAGS.finetune_checkpoint_path)
    return None

  tf.logging.info('Fine-tuning from %s', checkpoint_path)

  if FLAGS.finetune_exclude_pretrained_scopes:
    exclusions = {
      scope.strip()
      for scope in FLAGS.finetune_exclude_pretrained_scopes.split(',')
    }
  else:
    exclusions = set()

  filtered_model_variables = [
    v for v in tf.contrib.framework.get_model_variables()
    if not any([v.op.name.startswith(e) for e in exclusions])
  ]

  if variable_averages:
    variables_to_restore = {}
    for v in filtered_model_variables:
      # variables_to_restore[variable_averages.average_name(v)] = v
      if v in tf.trainable_variables():
        variables_to_restore[variable_averages.average_name(v)] = v
      else:
        variables_to_restore[v.op.name] = v
  else:
    variables_to_restore = {v.op.name: v for v in filtered_model_variables}

  assign_fn = tf.contrib.framework.assign_from_checkpoint_fn(
    checkpoint_path,
    variables_to_restore)
  if assign_fn:
    return lambda _, sess: assign_fn(sess)
  else:
    return None


def main(_):
  print(FLAGS.task_ids)
  task_ids = FLAGS.task_ids.split(' ')
  assert FLAGS.output_dir, '--output_dir has to be provided'
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  params = model_lib.default_hparams()
  params.parse(FLAGS.hparams)
  tf.logging.info('User provided hparams: %s', FLAGS.hparams)
  tf.logging.info('All hyper parameters: %s', params)
  batch_size = params.batch_size
  graph = tf.Graph()
  with graph.as_default():
    if FLAGS.job_name in ['ps', 'worker']:
      pss = ["localhost:2210"]
      workers = ["localhost:221" + str(i + 1) for i in range(len(task_ids))]
      print(workers)
      cluster = tf.train.ClusterSpec({"ps": pss, "worker": workers})
      server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task)

    if FLAGS.job_name == "ps":
      server.join()
    else:
      if FLAGS.job_name == "worker":
        device = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task,
                                                ps_device="/job:ps/task:0", cluster=cluster)
      else:
        device = '/device:GPU:0'

      with tf.device(device):
        img_bounds = (-1, 1) if 'finetune' in FLAGS.output_dir else (0, 1)
        dataset, examples_per_epoch, num_classes, bounds = (
          dataset_factory.get_dataset(
            'tiny_imagenet',
            'train',
            batch_size,
            64,
            is_training=True,
            bounds=img_bounds))
        dataset_iterator = dataset.make_one_shot_iterator()
        images, labels = dataset_iterator.get_next()
        one_hot_labels = tf.one_hot(labels, num_classes)

        # set up model
        global_step = tf.train.get_or_create_global_step()
        model_fn = model_lib.get_model(FLAGS.model_name, num_classes)

        if FLAGS.add_noise:
          images += tf.random_normal(images.shape, mean=0, stddev=16.0/255.0)

        if params.train_adv_method == 'clean':
          logits = model_fn(images, is_training=True)
          adv_examples = None
          correct = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
          train_acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        else:
          model_fn_eval_mode = lambda x: model_fn(x, is_training=False)
          adv_examples = adversarial_attack.generate_adversarial_examples(
            images, labels, bounds, model_fn_eval_mode, params.train_adv_method)
          all_examples = tf.concat([images, adv_examples], axis=0)
          logits = model_fn(all_examples, is_training=True)
          one_hot_labels = tf.concat([one_hot_labels, one_hot_labels], axis=0)
          correct = tf.equal(tf.argmax(logits[:batch_size], 1), tf.argmax(one_hot_labels[:batch_size], 1))
          train_acc = tf.reduce_mean(tf.cast(correct, tf.float32))
          correct = tf.equal(tf.argmax(logits[batch_size:], 1), tf.argmax(one_hot_labels[batch_size:], 1))
          adv_train_acc = tf.reduce_mean(tf.cast(correct, tf.float32))
          tf.summary.scalar('adv_train_acc', adv_train_acc)

        tf.summary.scalar('train_acc', train_acc)

        # update trainable variables if fine tuning is used
        model_lib.filter_trainable_variables(
          FLAGS.finetune_trainable_scopes)

        # set up losses
        # Note: batch_size is per-worker, not the global one
        if 'Plain' in FLAGS.output_dir:
          ce_labels, ce_logits = one_hot_labels[0:batch_size], logits[0:batch_size]
        elif '50% AT' in FLAGS.output_dir:
          ce_labels, ce_logits = one_hot_labels[0:2*batch_size], logits[0:2*batch_size]
        elif '100% AT' in FLAGS.output_dir:
          ce_labels, ce_logits = one_hot_labels[batch_size:2*batch_size], logits[batch_size:2*batch_size]
        else:
          ce_labels, ce_logits = one_hot_labels, logits

        total_loss = tf.losses.softmax_cross_entropy(
          onehot_labels=ce_labels,
          logits=ce_logits,
          label_smoothing=params.label_smoothing)
        tf.summary.scalar('loss_xent', total_loss)

        if params.train_lp_weight > 0:
          images1, images2 = tf.split(logits, 2)
          loss_lp = tf.losses.mean_squared_error(
            images1, images2, weights=params.train_lp_weight)
          tf.summary.scalar('loss_lp', loss_lp)
          total_loss += loss_lp

        if FLAGS.logit_squeezing > 0:
          logit_squeezing_loss = tf.reduce_mean(tf.norm(logits, ord='euclidean', axis=1))
          total_loss += 0.05 * logit_squeezing_loss
          tf.summary.scalar('logit_squeezing_loss', logit_squeezing_loss)

        if params.weight_decay > 0:
          loss_wd = (
              params.weight_decay
              * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
          )
          tf.summary.scalar('loss_wd', loss_wd)
          total_loss += loss_wd

        # Setup the moving averages:
        if FLAGS.moving_average_decay and (FLAGS.moving_average_decay > 0):
          with tf.name_scope('moving_average'):
            moving_average_variables = tf.contrib.framework.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, global_step)
        else:
          moving_average_variables = None
          variable_averages = None

        # set up optimizer and training op
        learning_rate, steps_per_epoch = model_lib.get_lr_schedule(
          params, examples_per_epoch, FLAGS.replicas_to_aggregate)

        optimizer = model_lib.get_optimizer(params, learning_rate)

        optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)

        train_op = tf.contrib.training.create_train_op(
          total_loss, optimizer,
          update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        tf.summary.image('images', images[0:FLAGS.num_summary_images])
        if adv_examples is not None:
          tf.summary.image('adv_images', adv_examples[0:FLAGS.num_summary_images])
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('current_epoch',
                          tf.to_double(global_step) / steps_per_epoch)

        # Training
        is_chief = FLAGS.task == 0

        scaffold = tf.train.Scaffold(
          init_fn=_get_finetuning_init_fn(variable_averages))
        hooks = [
          tf.train.NanTensorHook(total_loss),
        ]
        chief_only_hooks = [
          tf.train.LoggingTensorHook({'total_loss': total_loss,
                                      'global_step': global_step},
                                     every_n_iter=10),
          tf.train.SummarySaverHook(save_steps=FLAGS.save_summaries_steps,
                                    save_secs=FLAGS.save_summaries_secs,
                                    output_dir=FLAGS.output_dir,
                                    scaffold=scaffold),
          tf.train.CheckpointSaverHook(FLAGS.output_dir,
                                       save_steps=FLAGS.save_model_steps,
                                       scaffold=scaffold),
        ]

        if FLAGS.max_steps > 0:
          hooks.append(
            tf.train.StopAtStepHook(last_step=FLAGS.max_steps))

        # hook for sync replica training
        hooks.append(optimizer.make_session_run_hook(is_chief))

        if FLAGS.job_name == "worker":
          master = server.target
        else:
          master = None
        with tf.train.MonitoredTrainingSession(
            master=master,
            is_chief=is_chief,
            checkpoint_dir=FLAGS.output_dir,
            scaffold=scaffold,
            hooks=hooks,
            chief_only_hooks=chief_only_hooks,
            save_checkpoint_secs=None,
            save_summaries_steps=None,
            save_summaries_secs=None,
            config=tf.ConfigProto(allow_soft_placement=True)
        ) as session:
          while not session.should_stop():
            session.run([train_op])


if __name__ == '__main__':
  app.run(main)
