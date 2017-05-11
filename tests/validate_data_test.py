"""Unittest for validate_data.py."""
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
import unittest

import tensorflow as tf

from quality.quality import validate_data

FLAGS = tf.app.flags.FLAGS


class ValidateDataTest(unittest.TestCase):

  def setUp(self):
    self.input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__))
,"data")
    self.input_image_path = os.path.join(
        self.input_directory, 'BBBC006_z_aligned__a01__s1__w1_10.png')

  def testCheckDuplicateImageNameRuns(self):
    validate_data.check_duplicate_image_name(['/a/b.c', '/d/e.f'])

  def testCheckDuplicateImageNameSameName(self):
    with self.assertRaises(ValueError):
      validate_data.check_duplicate_image_name(['/a/b.c', '/a/b.c'])

  def testCheckDuplicateImageNameDifferentPathAndExtension(self):
    with self.assertRaises(ValueError):
      validate_data.check_duplicate_image_name(['/a/b.c', '/d/b.f'])

  def testCheckImageDimensionsRuns(self):
    validate_data.check_image_dimensions([self.input_image_path], 10, 10)

  def testCheckImageDimensionsImageTooSmall(self):
    with self.assertRaises(ValueError):
      validate_data.check_image_dimensions([self.input_image_path], 1e4, 1e4)


if __name__ == '__main__':
  unittest.main()
