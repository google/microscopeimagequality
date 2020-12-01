import os
import tempfile

import numpy
import skimage.io
import tensorflow

import microscopeimagequality.dataset_creation
import microscopeimagequality.degrade

test_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

test_dir = tempfile.mkdtemp()

tensorflow.logging.info("Loaded test data")

degrader = microscopeimagequality.degrade.ImageDegrader(random_seed=0, photoelectron_factor=65535.0, sensor_offset_in_photoelectrons=0.0)


def get_test_image(name):
    path = os.path.join(test_data_directory, name)

    return microscopeimagequality.dataset_creation.read_16_bit_greyscale(path)


def test_set_exposure_golden():
    exposure_factor = 100.0

    image = get_test_image("cell_image.tiff")

    exposure_adjusted_image = degrader.set_exposure(image, exposure_factor)

    # Check image is saturated.
    assert 1.0 == numpy.max(exposure_adjusted_image)

    expected_image = get_test_image("cell_image_saturated.png")

    numpy.testing.assert_almost_equal(expected_image, exposure_adjusted_image, 4)


def test_set_exposure_golden2():
    exposure_factor = 0.0001

    image = get_test_image("cell_image.tiff")

    exposure_adjusted_image = degrader.set_exposure(image, exposure_factor)

    numpy.testing.assert_almost_equal(exposure_factor, numpy.max(exposure_adjusted_image) / numpy.max(image), 4)


def test_set_exposure_with_offset_golden():
    exposure_factor = 100.0

    image = get_test_image("cell_image.tiff")

    degrader = microscopeimagequality.degrade.ImageDegrader(random_seed=0, photoelectron_factor=65535.0, sensor_offset_in_photoelectrons=100.0)

    exposure_adjusted_image = degrader.set_exposure(image, exposure_factor)

    # Check image is saturated.
    assert 1.0 == numpy.max(exposure_adjusted_image)

    expected_image = get_test_image("cell_image_saturated_with_offset.png")

    numpy.testing.assert_almost_equal(expected_image, exposure_adjusted_image, 4)


def test_set_exposure_no_exposure_change():
    exposure_factor = 1.0

    image = get_test_image("cell_image.tiff")

    exposure_adjusted_image = degrader.set_exposure(image, exposure_factor)

    numpy.testing.assert_almost_equal(image, exposure_adjusted_image, 4)


def test_apply_poisson_noise():
    image = get_test_image("cell_image.tiff")

    noisy_image = degrader.random_noise(image)

    expected_image = get_test_image("cell_image_poisson_noise_py.png")

    numpy.testing.assert_almost_equal(expected_image, noisy_image)


def test_get_airy_psf():
    image = get_test_image("cell_image.tiff")

    psf = get_test_image("psf.png")

    blurred_image = degrader.apply_blur_kernel(image, psf)

    expected_image = get_test_image("cell_image_airy_blurred.png")

    numpy.testing.assert_almost_equal(expected_image, blurred_image, 4)


def test_evaluate_airy_psf_at_point():
    psf_value = microscopeimagequality.degrade.get_airy_psf(1, 1e-6, 0.0, 500e-9, 0.5, 1.0, False)[0]

    numpy.testing.assert_almost_equal(psf_value, .25, 5)

    psf_value = microscopeimagequality.degrade.get_airy_psf(1, 1e-6, 1e-6, 500e-9, 0.5, 1.0, False)[0]

    numpy.testing.assert_almost_equal(psf_value, .20264, 5)

    psf_value = microscopeimagequality.degrade.get_airy_psf(3, 3e-6, 0.0, 500e-9, 0.5, 1.0, False)[0, 1]

    numpy.testing.assert_almost_equal(psf_value, .00114255, 7)


def test_get_airy_psf_golden():
    psf = microscopeimagequality.degrade.get_airy_psf(21, 5e-6, 4.0e-6, 500e-9, 0.5, 1.0)

    expected_psf = get_test_image("psf.png")

    numpy.testing.assert_almost_equal(expected_psf, psf, 4)


def test_get_airy_psf_golden_zero_depth():
    psf = microscopeimagequality.degrade.get_airy_psf(5, 5e-6, 0.0e-6, 500e-9, 0.5, 1.0)

    # This should be a delta function for large enough pixel sizes.
    expected_psf = numpy.zeros((5, 5))

    expected_psf[2, 2] = 1.0

    numpy.testing.assert_almost_equal(expected_psf, psf, 2)


def test_read_write_png():
    image = get_test_image("cell_image.tiff")
    output_path = os.path.join(test_dir, "cell_image2.png")

    skimage.io.imsave(output_path, image, "pil")

    image2 = microscopeimagequality.dataset_creation.read_16_bit_greyscale(output_path)

    numpy.testing.assert_almost_equal(image, image2, 4)


def test_degrade_images():
    glob = os.path.join(test_data_directory, "cell_image.tiff*")

    output_path = test_dir

    microscopeimagequality.degrade.degrade_images(glob, output_path, 20e-6, 1.0, 0, 65535, 0, psf_width_pixels=21, pixel_size_meters=5e-6 / 21)

    degraded_image = microscopeimagequality.dataset_creation.read_16_bit_greyscale(os.path.join(output_path, "cell_image.png"))

    expected_image = get_test_image("cell_image_degraded.png")

    numpy.testing.assert_almost_equal(expected_image, degraded_image, 4)


def test_degrade_images_no_change():
    glob = os.path.join(test_data_directory, "cell_image.tiff*")

    output_path = os.path.join(test_dir, 'no_change')

    microscopeimagequality.degrade.degrade_images(glob, output_path, 0e-6, 1.0, 0, 65535, 0, psf_width_pixels=21, pixel_size_meters=40e-6 / 21, skip_apply_poisson_noise=True)

    degraded_image = microscopeimagequality.dataset_creation.read_16_bit_greyscale(os.path.join(output_path, "cell_image.png"))

    expected_image = get_test_image("cell_image.tiff")

    numpy.testing.assert_almost_equal(expected_image, degraded_image, 4)
