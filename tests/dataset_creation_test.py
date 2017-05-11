"""Unittest for dataset_creation.py."""
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
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from quality.quality import dataset_creation

FLAGS = tf.app.flags.FLAGS


class DatasetCreationTest(unittest.TestCase):

  def setUp(self):
    self.input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__))
,"data")
    self.test_dir = tempfile.mkdtemp()    
    self.input_image_path = os.path.join(
        self.input_directory, 'BBBC006_z_aligned__a01__s1__w1_10.png')
    self.input_image_path_tif = os.path.join(self.input_directory,
                                             '00_mcf-z-stacks-03212011_k06_s2_w12667264a-6432-4f7e-bf58-625a1319a1c9.tif')
    self.glob_images = os.path.join(self.input_directory, 'images_for_glob_test/*')

    self.list_of_class_globs = []
    self.num_classes = 3

    for _ in range(self.num_classes):
      self.list_of_class_globs.append(self.glob_images)

    self.empty_directory = os.path.join(self.test_dir, 'empty')
    self.image_width = 520
    self.image_height = 520

  def testDatasetRandomizeRuns(self):
    dataset = dataset_creation.Dataset(
        np.zeros((2, 2)), ['a', 'b'], self.image_width, self.image_height)
    dataset.randomize()

  def testDatsetSubsampleForShard(self):
    labels = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    image_paths = ['path'] * labels.shape[0]
    dataset = dataset_creation.Dataset(labels, image_paths, self.image_width,
                                       self.image_height)
    dataset.subsample_for_shard(0, 2)
    np.testing.assert_array_equal(np.array([[0, 1], [4, 5]]), dataset.labels)

  def testDatasetGetSample(self):
    dataset = dataset_creation.Dataset(
        np.zeros((2, 2)), [self.input_image_path, self.input_image_path],
        self.image_width, self.image_height)
    _, _, image_path = dataset.get_sample(0, True)
    self.assertEquals(self.input_image_path, image_path)

  def testDatasetToExamplesInTfrecordRuns(self):
    dataset_creation.dataset_to_examples_in_tfrecord(
        self.list_of_class_globs,
        self.test_dir,
        output_tfrecord_filename='data_train.tfrecord',
        num_classes=self.num_classes,
        image_width=self.image_width,
        image_height=self.image_height)

  def testConvertToExamplesRuns(self):
    labels = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    image_paths = [self.input_image_path] * 3
    dataset_creation.convert_to_examples(
        dataset_creation.Dataset(labels, image_paths, self.image_width,
                                 self.image_height),
        output_directory=self.test_dir,
        output_tfrecord_filename='data_train.tfrecord')

  def testGetPreprocesssedImageRuns(self):
    image = dataset_creation.get_preprocessed_image(
        self.input_image_path,
        image_background_value=0.0,
        image_brightness_scale=1.0,
        image_width=self.image_width,
        image_height=self.image_height,
        normalize=True)
    self.assertEqual((520, 520), image.shape)

  def testNormalizeImage(self):
    image = dataset_creation.read_16_bit_greyscale(self.input_image_path)
    image_normalized = dataset_creation.normalize_image(image)
    expected_mean = np.mean(
        image) * 496.283426445 * dataset_creation._FOREGROUND_MEAN
    self.assertLess(np.abs(expected_mean - np.mean(image_normalized)), 1e-6)

  def testNormalizeImageNoForeground(self):
    image = np.zeros((100, 100), dtype=np.float32)
    image_normalized = dataset_creation.normalize_image(image)
    self.assertEquals(0.0, np.mean(image_normalized))

  def testGenerateTfExampleRuns(self):
    image = np.ones((100, 100), dtype=np.float32)
    label = np.array([0.0, 1.0], dtype=np.float32)
    image_path = 'directory/filename.extension'
    _ = dataset_creation.generate_tf_example(image, label, image_path)

  def testRead16BitGreyscalePng(self):
    image = dataset_creation.read_16_bit_greyscale(self.input_image_path)
    self.assertEquals(image.shape, (520, 696))
    self.assertAlmostEquals(np.max(image), 3252.0 / 65535)
    self.assertEquals(image.dtype, np.float32)

  def testRead16BitGreyscaleTif(self):
    image = dataset_creation.read_16_bit_greyscale(self.input_image_path_tif)
    self.assertEquals(image.shape, (520, 696))
    self.assertAlmostEquals(np.max(image), 1135.0 / 65535)
    self.assertEquals(image.dtype, np.float32)

  def testGetImagePaths(self):
    directory_images = self.input_directory

    paths = dataset_creation.get_image_paths(
        os.path.join(self.input_directory, 'images_for_glob_test'), 100)
    for path in paths:
      extension = os.path.splitext(path)[1]
      assert extension == '.png' or extension == '.tif', 'path is %s' % path
    self.assertEquals(24, len(paths))

  def testImageSizeFromGlob(self):
    image_size = dataset_creation.image_size_from_glob(self.input_image_path,
                                                       84)
    self.assertEquals(504, image_size.height)
    self.assertEquals(672, image_size.width)

  def testGetImagesFromGlob(self):
    paths = dataset_creation.get_images_from_glob(self.glob_images, 100)
    for path in paths:
      assert os.path.splitext(path)[1] == '.png' or os.path.splitext(path)[
          1] == '.tif', 'path is %s' % path
    self.assertEquals(24, len(paths))

  def testReadLabeledDatasetWithoutPatches(self):
    max_images = 3
    dataset = dataset_creation.read_labeled_dataset(self.list_of_class_globs,
                                                    max_images,
                                                    self.num_classes,
                                                    self.image_width,
                                                    self.image_height)

    num_images_expected = (max_images * self.num_classes)

    self.assertEquals(dataset.labels.shape,
                      (num_images_expected, self.num_classes))
    self.assertEquals(num_images_expected, len(dataset.image_paths))

  def testReadUnlabeledDataset(self):
    max_images = 3
    num_classes = 5
    dataset = dataset_creation.read_unlabeled_dataset([self.glob_images],
                                                      max_images, num_classes,
                                                      self.image_width,
                                                      self.image_height)

    num_images_expected = max_images

    self.assertEquals(dataset.labels.shape, (num_images_expected, num_classes))
    self.assertEquals(num_images_expected, len(dataset.image_paths))


if __name__ == '__main__':
  unittest.main()
