import os
import unittest

import tensorflow

import quality.validation

FLAGS = tensorflow.app.flags.FLAGS


class ValidateDataTest(unittest.TestCase):
    def setUp(self):
        self.input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

        self.input_image_path = os.path.join(self.input_directory, 'BBBC006_z_aligned__a01__s1__w1_10.png')

    def testCheckDuplicateImageNameRuns(self):
        quality.validation.check_duplicate_image_name(['/a/b.c', '/d/e.f'])

    def testCheckDuplicateImageNameSameName(self):
        with self.assertRaises(ValueError):
            quality.validation.check_duplicate_image_name(['/a/b.c', '/a/b.c'])

    def testCheckDuplicateImageNameDifferentPathAndExtension(self):
        with self.assertRaises(ValueError):
            quality.validation.check_duplicate_image_name(['/a/b.c', '/d/b.f'])

    def testCheckImageDimensionsRuns(self):
        quality.validation.check_image_dimensions([self.input_image_path], 10, 10)

    def testCheckImageDimensionsImageTooSmall(self):
        with self.assertRaises(ValueError):
            quality.validation.check_image_dimensions([self.input_image_path], 1e4, 1e4)
