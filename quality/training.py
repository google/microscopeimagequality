"""Trains an Miq model.

Usage:
  Start training:
    python quality/miq_train.py --data_globs "/focus0/*,/focus1/*,/focus2/*, \
      /focus3/*,/focus4/*,/focus5/*,/focus6/*,/focus7/*,/focus8/*,/focus9/*, \
      /focus10/*" --train_log_dir <path_to_train_directory>

  View training progress:
    tensorboard --logdir=<path_to_train_directory>

    In web browser, go to localhost:6006.
"""
# Copyright 2017 Google Inc.
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

import os

import tensorflow
import tensorflow.contrib.slim

import data_provider
import dataset_creation
import miq

flags = tensorflow.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('data_globs', None,
                    'Comma-separated string of globs, one per class.')

flags.DEFINE_float('image_background_value', 0.0,
                   'Background value of images to subtract.')

flags.DEFINE_integer('patch_width', 84, 'The image patch width, in pixels.')

flags.DEFINE_string('train_log_dir', '/tmp/miq/',
                    'Directory where to write event logs.')

flags.DEFINE_float('learning_rate', .00003, 'The learning rate')

flags.DEFINE_integer(
    'save_summaries_secs', 15,
    'The frequency with which summaries are saved, in seconds.')

flags.DEFINE_integer('save_interval_secs', 60,
                     'The frequency with which the model is saved, in seconds.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

flags.DEFINE_integer('model_id', 0, 'Model ID.')

FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    if FLAGS.max_number_of_steps == 0:
        FLAGS.max_number_of_steps = None

    list_of_image_globs = FLAGS.data_globs.split(',')
    num_classes = len(list_of_image_globs)

    output_tfrecord_file_pattern = ('worker%g_' % FLAGS.task) + 'data_%s.tfrecord'

    image_size = dataset_creation.image_size_from_glob(list_of_image_globs[0],
                                                       FLAGS.patch_width)

    # Read images and convert to TFExamples in an TFRecord.
    dataset_creation.dataset_to_examples_in_tfrecord(
        list_of_image_globs,
        FLAGS.train_log_dir,
        output_tfrecord_file_pattern % 'train',
        num_classes,
        image_width=image_size.width,
        image_height=image_size.height,
        image_background_value=FLAGS.image_background_value)

    tfexamples_tfrecord_file_pattern = os.path.join(FLAGS.train_log_dir,
                                                    output_tfrecord_file_pattern)

    g = tensorflow.Graph()
    with g.as_default():

        # If ps_tasks is zero, the local device is used. When using multiple
        # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
        # across the different devices.
        with tensorflow.device(tensorflow.train.replica_device_setter(FLAGS.ps_tasks)):
            images, one_hot_labels, _, _ = data_provider.provide_data(
                tfexamples_tfrecord_file_pattern,
                split_name='train',
                batch_size=64,
                num_classes=num_classes,
                image_width=image_size.width,
                image_height=image_size.height,
                patch_width=FLAGS.patch_width)

            # Visualize the input
            tensorflow.summary.image('train_input', images)
            labels = tensorflow.argmax(one_hot_labels, 1)
            # slim.summaries.add_histogram_summaries([images, labels])


            # Define the model:
            logits = miq.miq_model(
                images,
                num_classes=num_classes,
                is_training=True,
                model_id=FLAGS.model_id)

            # Specify the loss function:
            miq.add_loss(logits, one_hot_labels, use_rank_loss=True)
            total_loss = tensorflow.losses.get_total_loss()
            tensorflow.summary.scalar('Total_Loss', total_loss)

            # Specify the optimization scheme:
            optimizer = tensorflow.train.AdamOptimizer(FLAGS.learning_rate)

            # Set up training.
            train_op = tensorflow.contrib.slim.learning.create_train_op(total_loss, optimizer)

            # Monitor model variables for debugging.
            # slim.summaries.add_histogram_summaries(slim.get_model_variables())

            # Run training.
            tensorflow.contrib.slim.learning.train(
                train_op=train_op,
                logdir=FLAGS.train_log_dir,
                is_chief=FLAGS.task == 0,
                number_of_steps=FLAGS.max_number_of_steps,
                save_summaries_secs=FLAGS.save_summaries_secs,
                save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
    tensorflow.app.run()
