"""Unittest for degrade.py."""
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

import numpy as np

import tensorflow as tf

import unittest

from quality import dataset_creation

import degrade

FLAGS = tf.app.flags.FLAGS


class DegradeTest(unittest.TestCase):
  """Tests degrade.py."""

  def setUp(self):
    """Set up paths to test data."""
    self.test_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__))
,"testdata")
    self.test_dir = tempfile.mkdtemp()
    tf.logging.info("Loaded test data")
    self.degrader = degrade.ImageDegrader(
        random_seed=0,
        photoelectron_factor=65535.0,
        sensor_offset_in_photoelectrons=0.0)

  def get_test_image(self, name):
    path = os.path.join(self.test_data_directory, name)
    return dataset_creation.read_16_bit_greyscale(path)

  def testSetExposureGolden(self):
    exposure_factor = 100.0
    image = self.get_test_image("cell_image.tiff")
    exposure_adjusted_image = self.degrader.set_exposure(image, exposure_factor)

    # Check image is saturated.
    self.assertEquals(1.0, np.max(exposure_adjusted_image))

    expected_image = self.get_test_image("cell_image_saturated.png")
    np.testing.assert_almost_equal(expected_image, exposure_adjusted_image, 4)

  def testSetExposureGolden2(self):
    exposure_factor = 0.0001
    image = self.get_test_image("cell_image.tiff")
    exposure_adjusted_image = self.degrader.set_exposure(image, exposure_factor)

    np.testing.assert_almost_equal(exposure_factor,
                                   np.max(exposure_adjusted_image) /
                                   np.max(image), 4)

  def testSetExposureWithOffsetGolden(self):
    exposure_factor = 100.0
    image = self.get_test_image("cell_image.tiff")
    degrader = degrade.ImageDegrader(
        random_seed=0,
        photoelectron_factor=65535.0,
        sensor_offset_in_photoelectrons=100.0)
    exposure_adjusted_image = degrader.set_exposure(image, exposure_factor)

    # Check image is saturated.
    self.assertEquals(1.0, np.max(exposure_adjusted_image))

    expected_image = self.get_test_image("cell_image_saturated_with_offset.png")
    np.testing.assert_almost_equal(expected_image, exposure_adjusted_image, 4)

  def testSetExposureNoExposureChange(self):
    exposure_factor = 1.0
    image = self.get_test_image("cell_image.tiff")
    exposure_adjusted_image = self.degrader.set_exposure(image, exposure_factor)

    np.testing.assert_almost_equal(image, exposure_adjusted_image, 4)

  def testApplyPoissonNoise(self):
    image = self.get_test_image("cell_image.tiff")
    noisy_image = self.degrader.apply_poisson_noise(image)

    expected_image = self.get_test_image("cell_image_poisson_noise_py.png")

    np.testing.assert_almost_equal(expected_image, noisy_image)

  def testGetAiryPsf(self):
    image = self.get_test_image("cell_image.tiff")
    psf = self.get_test_image("psf.png")
    blurred_image = self.degrader.apply_blur_kernel(image, psf)
    expected_image = self.get_test_image("cell_image_airy_blurred.png")
    np.testing.assert_almost_equal(expected_image, blurred_image, 4)

  def testEvaluateAiryPsfAtPoint(self):
    psf_value = degrade.get_airy_psf(1, 1e-6, 0.0, 500e-9, 0.5, 1.0, False)[0]
    np.testing.assert_almost_equal(psf_value, .25, 5)

    psf_value = degrade.get_airy_psf(1, 1e-6, 1e-6, 500e-9, 0.5, 1.0, False)[0]
    np.testing.assert_almost_equal(psf_value, .20264, 5)

    psf_value = degrade.get_airy_psf(3, 3e-6, 0.0, 500e-9, 0.5, 1.0,
                                     False)[0, 1]
    np.testing.assert_almost_equal(psf_value, .00114255, 7)

  def testGetAiryPsfGolden(self):
    psf = degrade.get_airy_psf(21, 5e-6, 4.0e-6, 500e-9, 0.5, 1.0)
    expected_psf = self.get_test_image("psf.png")
    np.testing.assert_almost_equal(expected_psf, psf, 4)
    
  def testGetAiryPsfGoldenZeroDepth(self):    
    psf = degrade.get_airy_psf(5, 5e-6, 0.0e-6, 500e-9, 0.5, 1.0)
    # This should be a delta function for large enough pixel sizes.
    expected_psf = np.zeros((5, 5))
    expected_psf[2, 2] = 1.0
    np.testing.assert_almost_equal(expected_psf, psf, 2)

  def testWritePngRuns(self):
    psf = self.get_test_image("psf.png")
    degrade.write_png(psf, os.path.join(self.test_dir, "psf.png"))

  def testReadWritePng(self):
    image = self.get_test_image("cell_image.tiff")
    output_path = os.path.join(self.test_dir, "cell_image2.png")
    degrade.write_png(image, output_path)
    image2 = dataset_creation.read_16_bit_greyscale(output_path)

    np.testing.assert_almost_equal(image, image2, 4)

  def testDegradeImages(self):
    glob = os.path.join(self.test_data_directory, "cell_image.tiff*")
    output_path = self.test_dir
    degrade.degrade_images(
        glob,
        output_path,
        20e-6,
        1.0,
        0,
        65535,
        0,
        psf_width_pixels=21,
        pixel_size_meters=5e-6/21)
    degraded_image = dataset_creation.read_16_bit_greyscale(
        os.path.join(output_path, "cell_image.png"))
    expected_image = self.get_test_image("cell_image_degraded.png")
    np.testing.assert_almost_equal(expected_image, degraded_image, 4)

  def testDegradeImagesNoChange(self):
    glob = os.path.join(self.test_data_directory, "cell_image.tiff*")
    output_path = os.path.join(self.test_dir, 'no_change')
    degrade.degrade_images(
        glob,
        output_path,
        0e-6,
        1.0,
        0,
        65535,
        0,
        psf_width_pixels=21,
        pixel_size_meters=40e-6/21,
        skip_apply_poisson_noise=True)
    degraded_image = dataset_creation.read_16_bit_greyscale(
        os.path.join(output_path, "cell_image.png"))
    expected_image = self.get_test_image("cell_image.tiff")
    np.testing.assert_almost_equal(expected_image, degraded_image, 4)


if __name__ == "__main__":
  unittest.main()
