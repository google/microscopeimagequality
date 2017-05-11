import os
import tempfile
import unittest

import numpy
import tensorflow

import quality.dataset_creation

FLAGS = tensorflow.app.flags.FLAGS


class DatasetCreationTest(unittest.TestCase):
    def setUp(self):
        self.input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__))
                                            , "data")
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
        dataset = quality.dataset_creation.Dataset(
            numpy.zeros((2, 2)), ['a', 'b'], self.image_width, self.image_height)
        dataset.randomize()

    def testDatsetSubsampleForShard(self):
        labels = numpy.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        image_paths = ['path'] * labels.shape[0]
        dataset = quality.dataset_creation.Dataset(labels, image_paths, self.image_width,
                                                   self.image_height)
        dataset.subsample_for_shard(0, 2)
        numpy.testing.assert_array_equal(numpy.array([[0, 1], [4, 5]]), dataset.labels)

    def testDatasetGetSample(self):
        dataset = quality.dataset_creation.Dataset(
            numpy.zeros((2, 2)), [self.input_image_path, self.input_image_path],
            self.image_width, self.image_height)
        _, _, image_path = dataset.get_sample(0, True)
        self.assertEquals(self.input_image_path, image_path)

    def testDatasetToExamplesInTfrecordRuns(self):
        quality.dataset_creation.dataset_to_examples_in_tfrecord(
            self.list_of_class_globs,
            self.test_dir,
            output_tfrecord_filename='data_train.tfrecord',
            num_classes=self.num_classes,
            image_width=self.image_width,
            image_height=self.image_height)

    def testConvertToExamplesRuns(self):
        labels = numpy.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=numpy.float32)
        image_paths = [self.input_image_path] * 3
        quality.dataset_creation.convert_to_examples(
            quality.dataset_creation.Dataset(labels, image_paths, self.image_width,
                                             self.image_height),
            output_directory=self.test_dir,
            output_tfrecord_filename='data_train.tfrecord')

    def testGetPreprocesssedImageRuns(self):
        image = quality.dataset_creation.get_preprocessed_image(
            self.input_image_path,
            image_background_value=0.0,
            image_brightness_scale=1.0,
            image_width=self.image_width,
            image_height=self.image_height,
            normalize=True)
        self.assertEqual((520, 520), image.shape)

    def testNormalizeImage(self):
        image = quality.dataset_creation.read_16_bit_greyscale(self.input_image_path)
        image_normalized = quality.dataset_creation.normalize_image(image)
        expected_mean = numpy.mean(
            image) * 496.283426445 * quality.dataset_creation._FOREGROUND_MEAN
        self.assertLess(numpy.abs(expected_mean - numpy.mean(image_normalized)), 1e-6)

    def testNormalizeImageNoForeground(self):
        image = numpy.zeros((100, 100), dtype=numpy.float32)
        image_normalized = quality.dataset_creation.normalize_image(image)
        self.assertEquals(0.0, numpy.mean(image_normalized))

    def testGenerateTfExampleRuns(self):
        image = numpy.ones((100, 100), dtype=numpy.float32)
        label = numpy.array([0.0, 1.0], dtype=numpy.float32)
        image_path = 'directory/filename.extension'
        _ = quality.dataset_creation.generate_tf_example(image, label, image_path)

    def testRead16BitGreyscalePng(self):
        image = quality.dataset_creation.read_16_bit_greyscale(self.input_image_path)
        self.assertEquals(image.shape, (520, 696))
        self.assertAlmostEquals(numpy.max(image), 3252.0 / 65535)
        self.assertEquals(image.dtype, numpy.float32)

    def testRead16BitGreyscaleTif(self):
        image = quality.dataset_creation.read_16_bit_greyscale(self.input_image_path_tif)
        self.assertEquals(image.shape, (520, 696))
        self.assertAlmostEquals(numpy.max(image), 1135.0 / 65535)
        self.assertEquals(image.dtype, numpy.float32)

    def testGetImagePaths(self):
        directory_images = self.input_directory

        paths = quality.dataset_creation.get_image_paths(
            os.path.join(self.input_directory, 'images_for_glob_test'), 100)
        for path in paths:
            extension = os.path.splitext(path)[1]
            assert extension == '.png' or extension == '.tif', 'path is %s' % path
        self.assertEquals(24, len(paths))

    def testImageSizeFromGlob(self):
        image_size = quality.dataset_creation.image_size_from_glob(self.input_image_path,
                                                                   84)
        self.assertEquals(504, image_size.height)
        self.assertEquals(672, image_size.width)

    def testGetImagesFromGlob(self):
        paths = quality.dataset_creation.get_images_from_glob(self.glob_images, 100)
        for path in paths:
            assert os.path.splitext(path)[1] == '.png' or os.path.splitext(path)[
                                                              1] == '.tif', 'path is %s' % path
        self.assertEquals(24, len(paths))

    def testReadLabeledDatasetWithoutPatches(self):
        max_images = 3
        dataset = quality.dataset_creation.read_labeled_dataset(self.list_of_class_globs,
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
        dataset = quality.dataset_creation.read_unlabeled_dataset([self.glob_images],
                                                                  max_images, num_classes,
                                                                  self.image_width,
                                                                  self.image_height)

        num_images_expected = max_images

        self.assertEquals(dataset.labels.shape, (num_images_expected, num_classes))
        self.assertEquals(num_images_expected, len(dataset.image_paths))
