"""Unittest for data_provider.py."""
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

import numpy as np
import png

import tempfile
from tensorflow.contrib.slim import dataset_data_provider
import tensorflow as tf

import unittest

from quality import data_provider

FLAGS = tf.app.flags.FLAGS

TFRECORD_NUM_ENTRIES = 33
TFRECORD_NUM_CLASSES = 3
TFRECORD_LABEL_ORDERING = [
    1, 1, 1, 1, 1, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 1, 0, 2, 0, 1, 2, 0, 2, 2, 0,
    1, 0, 1, 1, 2, 0, 0, 1
]


class DataProviderTest(unittest.TestCase):

  def setUp(self):
    self.input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__))
,"testdata")
    self.test_dir = tempfile.mkdtemp()    
    self.batch_size = TFRECORD_NUM_ENTRIES
    # For a patch size of 28, we have 324 patches per image in this tfrecord.
    self.patches_per_image = 324
    self.num_classes = TFRECORD_NUM_CLASSES
    self.tfrecord_file_pattern = os.path.join(self.input_directory,
                                              'data_%s.tfrecord')
    self.image_width = 520
    self.image_height = 520

  def testGetFilenameNumRecords(self):
    tf_record_path = '/folder/filename.tfrecord'
    path = data_provider.get_filename_num_records(tf_record_path)
    expected_path = '/folder/filename.num_records'
    self.assertEquals(expected_path, path)

  def testGetNumRecords(self):
    tf_record_path = os.path.join(self.input_directory, 'data_train.tfrecord')
    num_records = data_provider.get_num_records(tf_record_path)
    expected_num_records = TFRECORD_NUM_ENTRIES
    self.assertEquals(expected_num_records, num_records)

  def Save16BitPng(self, filename, im):
    path = os.path.join(self.test_dir, filename)
    with open(path, 'w') as f:
      writer = png.Writer(
          width=im.shape[1], height=im.shape[0], bitdepth=16, greyscale=True)
      writer.write(f, im.tolist())

  def GetTFSession(self, graph):
    sv = tf.train.Supervisor(logdir=os.path.join(self.test_dir, 'tmp_logs/'))
    sess = sv.PrepareSession('')
    sv.StartQueueRunners(sess, graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
    return sess

  def GetDataFromTfrecord(self):
    """Helper function that gets image, label tensors from tfrecord."""
    split_name = 'train'
    num_records = data_provider.get_num_records(self.tfrecord_file_pattern %
                                                split_name)
    self.assertEquals(TFRECORD_NUM_ENTRIES, num_records)
    dataset = data_provider.get_split(
        split_name,
        self.tfrecord_file_pattern,
        num_classes=self.num_classes,
        image_width=self.image_width,
        image_height=self.image_height)
    provider = dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=2 * self.batch_size,
        common_queue_min=self.batch_size,
        shuffle=False)
    [image, label, image_path] = provider.get([
        data_provider.FEATURE_IMAGE, data_provider.FEATURE_IMAGE_CLASS,
        data_provider.FEATURE_IMAGE_PATH
    ])
    return image, label, image_path

  def testGetSplit(self):
    g = tf.Graph()
    with g.as_default():
      image, label, image_path = self.GetDataFromTfrecord()

      sess = self.GetTFSession(g)

      # Check that the tensor shapes are as expected.
      np_image, np_label, np_image_path = sess.run([image, label, image_path])
      self.assertListEqual(
          list(np_image.shape),
          [data_provider.IMAGE_WIDTH, data_provider.IMAGE_WIDTH, 1])
      self.assertListEqual(list(np_label.shape), [self.num_classes])
      self.assertListEqual([1], list(np_image_path.shape))
      self.assertEquals(22, len(np_image_path[0]))

      # Write the image for viewing.
      im = (np.squeeze(np_image) * 65535).astype(np.uint16)
      image_class = np.argmax(np_label)
      self.Save16BitPng('single_im_from_tfrecord_%g.png' % image_class, im)

  def testBatching(self):

    g = tf.Graph()
    with g.as_default():
      image, label, image_path = self.GetDataFromTfrecord()

      # Expand since get_batches() requires a larger dimension tensor.
      expanded_label = tf.expand_dims(label, dim=0)
      expanded_image = tf.expand_dims(image, dim=0)
      expanded_image_path = tf.expand_dims(image_path, dim=0)

      images, labels, image_paths = data_provider.get_batches(
          expanded_image,
          expanded_label,
          expanded_image_path,
          batch_size=self.batch_size,
          num_threads=1)

      sess = self.GetTFSession(g)

      [np_images, np_labels, np_image_paths] = sess.run(
          [images, labels, image_paths])

      # Check the number of images and shape is as expected.
      self.assertListEqual(
          list(np_images.shape), [
              self.batch_size, data_provider.IMAGE_WIDTH,
              data_provider.IMAGE_WIDTH, 1
          ])
      self.assertListEqual([self.batch_size, 1], list(np_image_paths.shape))
      self.assertEquals(1, len(np_image_paths[0]))
      self.assertEquals('image_000',
                        os.path.basename(np_image_paths[0][0]))

      # Check the ordering of labels in a single batch (which is preserved
      # since we used num_threads=1).
      image_classes = np.argmax(np_labels, axis=1).tolist()

      self.assertListEqual(image_classes, TFRECORD_LABEL_ORDERING)

  def testGetImagePatchTensor(self):
    patch_width = 280
    g = tf.Graph()
    with g.as_default():
      image, label, image_path = self.GetDataFromTfrecord()
      patch, label, image_path = data_provider.get_image_patch_tensor(
          image, label, image_path, patch_width=patch_width)

      sess = self.GetTFSession(g)

      [np_patch, np_label, np_image_path] = sess.run([patch, label, image_path])

      # Check that the tensor shapes are as expected.
      self.assertListEqual(
          list(np_patch.shape), [1, patch_width, patch_width, 1])
      self.assertListEqual(list(np_label.shape), [1, self.num_classes])
      self.assertListEqual([1, 1], list(np_image_path.shape))

      # Write the image for viewing.
      im = (np.squeeze(np_patch) * 65535).astype(np.uint16)
      self.Save16BitPng('single_random_patch_from_tfrecord.png', im)

  def testApplyRandomBrightnessAdjust(self):
    g = tf.Graph()
    with g.as_default():
      image, _, _ = self.GetDataFromTfrecord()
      factor = 2.0
      patch = data_provider.apply_random_brightness_adjust(image, factor,
                                                           factor)

      sess = self.GetTFSession(g)

      [np_patch, np_image] = sess.run([patch, image])

      self.assertListEqual(list(np_patch.shape), list(np_image.shape))
      np.testing.assert_array_equal(np_image * factor, np_patch)

  def testGetImageTilesTensor(self):
    patch_width = 100
    g = tf.Graph()
    with g.as_default():
      image, label, image_path = self.GetDataFromTfrecord()
      tiles, labels, image_paths = data_provider.get_image_tiles_tensor(
          image, label, image_path, patch_width=patch_width)

      sess = self.GetTFSession(g)

      [np_tiles, np_labels, np_image_paths] = sess.run(
          [tiles, labels, image_paths])

      # Check that the tensor shapes are as expected.
      num_tiles_expected = 25
      self.assertListEqual(
          list(np_tiles.shape),
          [num_tiles_expected, patch_width, patch_width, 1])
      self.assertListEqual(
          list(np_labels.shape), [num_tiles_expected, self.num_classes])
      self.assertListEqual([num_tiles_expected, 1], list(np_image_paths.shape))

  def testGetImageTilesTensorNonSquare(self):
    patch_width = 100
    g = tf.Graph()
    with g.as_default():
      image = tf.zeros([patch_width * 4, patch_width * 3, 1])
      label = tf.constant([0, 0, 1])
      image_path = tf.constant(['path'])
      tiles, labels, image_paths = data_provider.get_image_tiles_tensor(
          image, label, image_path, patch_width=patch_width)

      sess = self.GetTFSession(g)

      [np_tiles, np_labels, np_image_paths] = sess.run(
          [tiles, labels, image_paths])

      # Check that the tensor shapes are as expected.
      num_tiles_expected = 12
      self.assertListEqual(
          list(np_tiles.shape),
          [num_tiles_expected, patch_width, patch_width, 1])
      self.assertListEqual(
          list(np_labels.shape), [num_tiles_expected, self.num_classes])
      self.assertListEqual([num_tiles_expected, 1], list(np_image_paths.shape))

  def testProvideDataWithRandomPatches(self):
    images, one_hot_labels, image_paths, _ = data_provider.provide_data(
        self.tfrecord_file_pattern,
        split_name='train',
        batch_size=self.batch_size,
        num_classes=self.num_classes,
        image_width=self.image_width,
        image_height=self.image_height,
        patch_width=28,
        randomize=True)

    self.assertEquals(images.get_shape().as_list(), [self.batch_size, 28, 28, 1])
    self.assertEquals(one_hot_labels.get_shape().as_list(),
                      [self.batch_size, self.num_classes])
    self.assertEquals([self.batch_size, 1], image_paths.get_shape().as_list())

  def testProvideDataImagePath(self):
    g = tf.Graph()
    with g.as_default():

      _, _, image_paths, _ = data_provider.provide_data(
          self.tfrecord_file_pattern,
          split_name='train',
          batch_size=self.patches_per_image,
          num_classes=3,
          image_width=self.image_width,
          image_height=self.image_height,
          patch_width=28,
          randomize=False,
          num_threads=1)

      sess = self.GetTFSession(g)

      [np_image_paths] = sess.run([image_paths])

      filename_expected = 'image_000'
      self.assertEquals(1, len(np_image_paths[0]))
      self.assertEquals(filename_expected,
                        os.path.basename(np_image_paths[0][0]))

  def testProvideDataUniformTiles(self):
    g = tf.Graph()
    with g.as_default():

      images, one_hot_labels, _, _ = data_provider.provide_data(
          self.tfrecord_file_pattern,
          split_name='train',
          batch_size=self.patches_per_image,
          num_classes=self.num_classes,
          image_width=self.image_width,
          image_height=self.image_height,
          patch_width=28,
          randomize=False)

      num_tiles_expected = self.patches_per_image
      self.assertEquals(images.get_shape().as_list(), [num_tiles_expected, 28, 28, 1])
      self.assertEquals(one_hot_labels.get_shape().as_list(),
                        [num_tiles_expected, self.num_classes])

      sess = self.GetTFSession(g)

      [np_images, np_labels] = sess.run([images, one_hot_labels])
      self.assertEquals(np_labels.shape, (num_tiles_expected, self.num_classes))

      im = (np.squeeze(np_images[0, :, :, :]) * 65535).astype(np.uint16)
      self.Save16BitPng('first_tile_single_batch.png', im)

  def testProvideDataWithDeterministicOrdering(self):
    # Use patches larger to speed up test, otherwise it will timeout.
    patch_size_factor = 3
    batch_size = self.patches_per_image / patch_size_factor**2
    g = tf.Graph()
    with g.as_default():

      images, one_hot_labels, image_paths, _ = data_provider.provide_data(
          self.tfrecord_file_pattern,
          split_name='train',
          batch_size=batch_size,
          num_classes=self.num_classes,
          image_width=self.image_width,
          image_height=self.image_height,
          patch_width=28 * patch_size_factor,
          randomize=False,
          num_threads=1)

      sess = self.GetTFSession(g)

      # Here, we are looking at the first label across many batches, rather than
      # the ordering of labels in one batch, as in testBatching(). We check to
      # ensure the ordering is deterministic for num_threads = 1.
      image_classes = []
      num_batches_tested = min(20, TFRECORD_NUM_ENTRIES)
      for i in range(num_batches_tested):
        [np_images, np_labels, np_image_paths] = sess.run(
            [images, one_hot_labels, image_paths])

        self.assertEquals(np_labels.shape, (batch_size, self.num_classes))

        # All class labels should be identical within this batch.
        image_class = np.argmax(np_labels, axis=1)
        self.assertTrue(np.all(image_class[0] == image_class))
        self.assertTrue(np.all(np_image_paths[0] == np_image_paths))
        image_classes.append(image_class[0])

        im = (np.squeeze(np_images[0, :, :, :]) * 65535).astype(np.uint16)
        self.Save16BitPng('first_tile_per_batch_%g.png' % i, im)

      self.assertListEqual(image_classes,
                           TFRECORD_LABEL_ORDERING[0:num_batches_tested])


if __name__ == '__main__':
  unittest.main()
