"""Unittest for miq.py."""
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

import unittest

import tensorflow as tf
import tensorflow.contrib.slim as slim

from quality.quality import miq

FLAGS = tf.app.flags.FLAGS


class MiqTest(tf.test.TestCase):

  def test_add_loss_training_runs(self):
    with self.test_session():
      targets = tf.constant(
          [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
      inputs = tf.constant([[0.7, 0.3, 0.0], [0.9, 0.1, 0.0], [0.6, 0.4, 0.0],
                            [0.0, 0.4, 0.6]])

      predictions = tf.contrib.layers.fully_connected(inputs, 3)

      miq.add_loss(targets, predictions, use_rank_loss=True)

      total_loss = tf.losses.get_total_loss()

      tf.summary.scalar("Total Loss", total_loss)

      optimizer = tf.train.AdamOptimizer(0.000001)

      # Set up training.
      train_op = slim.learning.create_train_op(total_loss, optimizer)

      # Run training.
      slim.learning.train(
          train_op, None, number_of_steps=5, log_every_n_steps=5)


if __name__ == "__main__":
  unittest.main()
