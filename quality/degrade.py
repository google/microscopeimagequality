"""
Tool for simulating microscope image degradations.

Example usage:
  To simulate defocus at a depth of 2 microns (for the default imaging
    parameters):

  from quality import degrade
  degrade.degrade_images('/path_clean_images/*',
                   '/degraded_image_output/'
                   z_depth_meters=2e-6,
                   exposure_factor=1.0,
                   random_seed=0,
                   photoelectron_factor=65535,
                   sensor_offset_in_photoelectrons=100)
"""

import os

import numpy
import past.builtins
import scipy.integrate
import scipy.signal
import scipy.special
import skimage.io

import quality.dataset_creation


class ImageDegrader(object):
    """
    Holds image sensor parameters for degrading images.

    Attributes:
        _random_generator: np.random.RandomState for generating noise.
        _photoelectron_factor: Float, factor to convert pixel values in range [0.0, 1.0] to units photoelectrons.
        _sensor_offset_in_photoelectrons: Float, image sensor offset (black level), in units of photoelectrons.
    """
    def __init__(self, random_seed=0, photoelectron_factor=65535.0, sensor_offset_in_photoelectrons=100.0):
        """
        Initialize with image sensor parameters.

        Args:
            random_seed: Integer, the random seed.
            photoelectron_factor: Float, factor to convert to photoelectrons.
            sensor_offset_in_photoelectrons: Float, image sensor offset (black level), in terms of photoelectrons.
        """
        self._photoelectron_factor = photoelectron_factor
        self._sensor_offset_in_photoelectrons = sensor_offset_in_photoelectrons
        self._random_generator = numpy.random.RandomState(random_seed)

    def random_noise(self, image):
        """
        Applies per-pixel Poisson noise to an image.

        Pixel values are converted to units of photoelectrons before noise is applied.

        Args:
            image: A 2D numpy float array in [0.0, 1.0], the image to apply noise to.

        Returns:
            A 2D numpy float array of same shape as 'image', in [0.0, 1.0].
        """
        image_photoelectrons = numpy.maximum(0.0, image * self._photoelectron_factor - self._sensor_offset_in_photoelectrons)

        noisy_image_photoelectrons = self._random_generator.poisson(image_photoelectrons).astype(numpy.float64)

        noisy_image = (noisy_image_photoelectrons + self._sensor_offset_in_photoelectrons) / self._photoelectron_factor

        clipped_image = numpy.minimum(1.0, noisy_image)

        return clipped_image

    @staticmethod
    def apply_blur_kernel(image, psf):
        """
        Applies a blur kernel to the image after normalizing the kernel.

        A symmetric boundary is used to handle the image borders.

        Args:
            image: A 2D numpy float array in [0.0, 1.0], the image to blur.
            psf: A 2D numpy float array, the kernel to blur the image with.

        Returns:
            A 2D numpy float array of same shape as 'image', in [0.0, 1.0].
        """
        psf_normalized = psf / numpy.sum(psf)

        return scipy.signal.convolve2d(image, psf_normalized, 'same', boundary='symm')

    def set_exposure(self, image, exposure_factor):
        """
        Adjusts the image exposure.

        Args:
            image: A 2D numpy float array in [0.0, 1.0], the image to adjust exposure in.
            exposure_factor: A non-negative float, the factor to adjust exposure by.

        Returns:
            A 2D numpy float array of same shape as 'image', in [0.0, 1.0].
        """

        image_without_offset = numpy.maximum(0.0, (image * self._photoelectron_factor - self._sensor_offset_in_photoelectrons))

        adjusted_without_offset = image_without_offset * exposure_factor

        adjusted = ((adjusted_without_offset + self._sensor_offset_in_photoelectrons) / self._photoelectron_factor)

        clipped_image = numpy.minimum(1.0, adjusted)

        return clipped_image


def get_airy_psf(psf_width_pixels, psf_width_meters, z, wavelength, numerical_aperture, refractive_index, normalize=True):
    """
    Generate Airy point spread function (psf) kernel from optical parameters.

    Args:
        psf_width_pixels: Integer, the width of the psf, in pixels. Must be odd. If this is even, testGetAiryPsfGoldenZeroDepth() will fail.
        psf_width_meters: Float, the width of the psf, in meters.
        z: Float, z-coordinate relative to the focal plane, in meters.
        wavelength: Float, wavelength of light in meters.
        numerical_aperture: Float, numerical aperture of the imaging lens.
        refractive_index: Float, refractive index of the imaging medium.
        normalize: Boolean, whether to normalize psf to max value.

    Returns:
        The psf kernel, a numpy float 2D array.

    Raises:
        ValueError: If psf_width_pixels is not an odd number.
    """
    if psf_width_pixels % 2 == 0:
        raise ValueError(
            'psf_width_pixels must be an odd number, but is %d.' % psf_width_pixels)

    meters_per_pixel = psf_width_meters / psf_width_pixels
    psf = numpy.zeros((psf_width_pixels, psf_width_pixels), dtype=numpy.float64)
    for i in past.builtins.xrange(psf_width_pixels):
        for j in past.builtins.xrange(psf_width_pixels):
            x = (i - (psf_width_pixels - 1.0) / 2.0) * meters_per_pixel
            y = (j - (psf_width_pixels - 1.0) / 2.0) * meters_per_pixel
            psf[i, j] = _evaluate_airy_function_at_point(
                x, y, z, wavelength, numerical_aperture, refractive_index)

    # Normalize PSF to max value.
    if normalize:
        return psf / numpy.max(psf)
    return psf


def _evaluate_airy_function_at_point(x, y, z, wavelength, numerical_aperture, refractive_index):
    """
    Evaluates the Airy point spread function at a point.

    Args:
        x: Float, x coordinate, in meters.
        y: Float, y coordinate, in meters.
        z: Float, z coordinate, in meters.
        wavelength: Float, wavelength of light in meters.
        numerical_aperture: Float, numerical aperture of the imaging lens.
        refractive_index: Float, refractive index of the imaging medium.

    Returns:
        A real float, the value of the Airy point spread function at the coordinate.
    """
    k = 2 * numpy.pi / wavelength
    na = numerical_aperture  # pylint: disable=invalid-name
    n = refractive_index

    def function_to_integrate(rho):
        bessel_arg = k * na / n * numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2)) * rho
        return scipy.special.j0(bessel_arg) * numpy.exp(-1.0 / 2.0 * 1j * k * numpy.power(
            rho, 2) * z * numpy.power(na / n, 2)) * rho

    integral_result = _integrate_numerical(function_to_integrate, 0.0, 1.0)
    return float(numpy.real(integral_result * numpy.conj(integral_result)))


def _integrate_numerical(function_to_integrate, start, end):
    """
    Numerically integrate a complex function with real end points.

    Args:
        function_to_integrate: Function to integrate.
        start: Float, real starting point.
        end: Float, real ending point.

    Returns:
        Complex float, the value of the numerical integration.
    """

    def real_function(x):
        return numpy.real(function_to_integrate(x))

    def imag_function(x):
        return numpy.imag(function_to_integrate(x))

    real_result = scipy.integrate.quad(real_function, start, end)[0]

    imag_result = scipy.integrate.quad(imag_function, start, end)[0]

    return real_result + 1j * imag_result


def degrade_images(images, output_path, z_depth_meters, exposure_factor, random_seed, photoelectron_factor, sensor_offset_in_photoelectrons, wavelength=500e-9, numerical_aperture=0.5, refractive_index=1.0, psf_width_pixels=51, pixel_size_meters=0.65e-6, skip_apply_poisson_noise=False):
    """
    Create a PSF and degrade all specified images.

    Args:
        images: String, glob for input images, either .png, .tif or .tiff.
        output_path: String, path to save degraded images.
        z_depth_meters: Z-coordinate, in meters, distance relative to focal plane.
        exposure_factor: A non-negative float, the factor to adjust exposure by.
        random_seed: Integer, the random seed.
        photoelectron_factor: Float, factor to convert to photoelectrons.
        sensor_offset_in_photoelectrons: Float, image sensor offset (black level), in terms of photoelectrons.
        wavelength: Float, wavelength of light in meters.
        numerical_aperture: Float, numerical aperture of the imaging lens.
        refractive_index: Float, refractive index of the imaging medium.
        psf_width_pixels: Integer, the width of the psf, in pixels. Must be odd.
        pixel_size_meters: Float, width of each image pixel in meters. This is the magnified camera pixel size.
        skip_apply_poisson_noise: Boolean, skip application of Poisson noise.

    Raises:
        ValueError: If no images are found by the specified glob.
    """
    psf_width_meters = psf_width_pixels * pixel_size_meters

    psf = get_airy_psf(psf_width_pixels, psf_width_meters, z_depth_meters, wavelength, numerical_aperture, refractive_index)

    degrader = ImageDegrader(random_seed, photoelectron_factor, sensor_offset_in_photoelectrons)

    image_paths = quality.dataset_creation.get_images_from_glob(images, max_images=1e7)

    if not image_paths:
        raise ValueError('No images found with glob %s.' % images)

    for path in image_paths:
        image = quality.dataset_creation.read_16_bit_greyscale(path)
        blurred_image = degrader.apply_blur_kernel(image, psf)
        exposure_adjusted_image = degrader.set_exposure(blurred_image, exposure_factor)

        if skip_apply_poisson_noise:
            noisy_image = exposure_adjusted_image
        else:
            noisy_image = degrader.random_noise(exposure_adjusted_image)

        output_filename = os.path.join(output_path, '%s.png' % os.path.splitext(os.path.basename(path))[0])

        output_dir = os.path.dirname(output_filename)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        skimage.io.imsave(output_filename, noisy_image)
