import logging
import os
import tempfile
import unittest
import numpy as np
import tensorflow as tf
from PIL import Image
from quality import constants
from quality import data_provider
from quality import evaluation
from quality import inference

FLAGS = tf.app.flags.FLAGS


class RunInferenceTest(tf.test.TestCase):
    def setUp(self):
        self.input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__))
                                            , "data")
        self.test_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__))
                                                , "data")
        self.test_dir = tempfile.mkdtemp()
        self.glob_images = os.path.join(self.input_directory, 'images_for_glob_test/*')

        self.patch_width = 84
        self.num_classes = 11

    def testPatchValuesToMask(self):
        values = np.round(
            np.array([[0.2, 0.4, 0.5], [1.0, 0.0, 0.3]]) *
            np.iinfo(np.uint16).max).astype(np.uint16)
        mask = inference.patch_values_to_mask(values, self.patch_width)
        self.assertEquals((168, 252), mask.shape)
        self.assertEquals(np.iinfo(np.uint16).max, np.max(mask))

    def testSaveMasksAndAnnotatedVisualization(self):
        test_filename = 'BBBC006_z_aligned__a01__s1__w1_10.png'
        orig_name = os.path.join(self.test_data_directory, test_filename)
        prediction = 1
        certainties = {name: 0.3 for name in evaluation.CERTAINTY_NAMES}
        num_patches = 4
        np_images = np.ones((num_patches, self.patch_width, self.patch_width, 1))
        np_probabilities = np.ones(
            (num_patches, self.num_classes)) / self.num_classes
        np_probabilities[0, :] = 0
        np_probabilities[0, 1] = 1.0
        np_probabilities[1, :] = 0
        np_probabilities[1, 2] = 0.4
        np_probabilities[1, -1] = 0.6
        np_labels = 2 * np.ones(num_patches)
        image_height = int(np.sqrt(num_patches)) * self.patch_width
        image_width = image_height

        inference.save_masks_and_annotated_visualization(
            orig_name, self.test_dir, prediction, certainties, np_images,
            np_probabilities, np_labels, self.patch_width, image_height,
            image_width)

        # Check that output has been generated and is the correct shape.
        expected_size = Image.open(orig_name, 'r').size
        expected_visualization_path = os.path.join(
            self.test_dir,
            'actual2_pred1_mean_certainty=0.300orig_name=%s' % test_filename)
        expected_predictions_path = os.path.join(self.test_dir,
                                                 constants.PREDICTIONS_MASK_FORMAT %
                                                 test_filename)
        expected_certainties_path = os.path.join(self.test_dir,
                                                 constants.CERTAINTY_MASK_FORMAT %
                                                 test_filename)
        expected_valid_path = os.path.join(self.test_dir,
                                           constants.VALID_MASK_FORMAT %
                                           test_filename)

        img = Image.open(expected_visualization_path, 'r')
        self.assertEquals(expected_size, img.size)

        img = Image.open(expected_predictions_path, 'r')
        self.assertEquals(expected_size, img.size)

        img = Image.open(expected_certainties_path, 'r')
        self.assertEquals(expected_size, img.size)

        img = Image.open(expected_valid_path, 'r')
        self.assertEquals(expected_size, img.size)

    def testSaveMasksAndAnnotatedVisualizationTif(self):
        test_filename = ('00_mcf-z-stacks-03212011_k06_s2_w12667264a'
                         '-6432-4f7e-bf58-625a1319a1c9.tif')
        orig_name = os.path.join(self.test_data_directory, test_filename)
        prediction = 1
        certainties = {name: 0.3 for name in evaluation.CERTAINTY_NAMES}
        num_patches = 4
        np_images = np.ones((num_patches, self.patch_width, self.patch_width, 1))
        np_probabilities = np.ones(
            (num_patches, self.num_classes)) / self.num_classes
        image_height = int(np.sqrt(num_patches)) * self.patch_width
        image_width = image_height

        np_labels = 2 * np.ones(num_patches)

        inference.save_masks_and_annotated_visualization(
            orig_name, self.test_dir, prediction, certainties, np_images,
            np_probabilities, np_labels, self.patch_width, image_height,
            image_width)

        mask_formats = [
            constants.CERTAINTY_MASK_FORMAT, constants.PREDICTIONS_MASK_FORMAT,
            constants.VALID_MASK_FORMAT
        ]
        for mask_format in mask_formats:
            orig_name_png = os.path.splitext(os.path.basename(orig_name))[0] + '.png'
            expected_file = os.path.join(self.test_dir,
                                         mask_format % orig_name_png)
            self.assertTrue(os.path.isfile(expected_file))

    def testRunModelInferenceFirstHalfRuns(self):
        batch_size = 1
        num_classes = 11
        model_patch_width = 84
        image_width = 84
        image_height = 84

        tfexamples_tfrecord = inference.build_tfrecord_from_pngs(
            [self.glob_images],
            use_unlabeled_data=True,
            num_classes=num_classes,
            eval_directory=self.test_dir,
            image_background_value=0,
            image_brightness_scale=1,
            shard_num=0,
            num_shards=1,
            image_width=image_width,
            image_height=image_height)

        num_samples = data_provider.get_num_records(tfexamples_tfrecord %
                                                    inference._SPLIT_NAME)

        logging.info('TFRecord has %g samples.', num_samples)

        g = tf.Graph()
        with g.as_default():
            images, one_hot_labels, _, _ = data_provider.provide_data(
                tfexamples_tfrecord,
                split_name=inference._SPLIT_NAME,
                batch_size=batch_size,
                num_classes=num_classes,
                image_width=84,
                image_height=84,
                patch_width=model_patch_width,
                randomize=False,
                num_threads=1)

            labels = evaluation.get_model_and_metrics(
                images,
                num_classes=num_classes,
                one_hot_labels=one_hot_labels,
                is_training=False).labels

            self.assertEquals(batch_size, labels.get_shape())


if __name__ == '__main__':
    unittest.main()
