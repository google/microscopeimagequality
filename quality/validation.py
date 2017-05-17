from __future__ import print_function

import logging
import os
import sys

import quality.dataset_creation

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def check_duplicate_image_name(image_paths):
    """
    Check that there are no duplicate names (without path or extension).

    Args:
    image_paths: List of strings, paths to images.

    Raises:
    ValueError: If there is a duplicate image name.
    """
    image_names = [os.path.basename(os.path.splitext(p)[0]) for p in image_paths]

    num_images = len(image_names)

    num_unique = len(set(image_names))

    if num_images != num_unique:
        raise ValueError('Found %d duplicate images.' % (num_images - num_unique))

    logging.info('Found no duplicates in %d images.', num_images)


def check_image_dimensions(image_paths, image_height, image_width):
    """
    Check that the image dimensions are valid.

    A valid image has height and width no smaller than the specified height, width.

    Args:
    image_paths: List of strings, paths to images.
    image_height: Integer, height of image.
    image_width: Integer, width of image.

    Raises:
    ValueError: If there is an invalid image dimension
    """
    logging.info('Using image height, width %s', str((image_height, image_width)))

    bad_images = []

    for path in image_paths:
        logging.info('Trying to read image %s', path)
        image = quality.dataset_creation.read_16_bit_greyscale(path)

        if image.shape[0] < image_height or image.shape[1] < image_width:
            bad_images.append(path)
            logging.info('Image %s dimension %s is too small.', path, str(image.shape))

    logging.info('Done checking images')

    logging.info('Found %d bad images.', len(bad_images))

    if bad_images:
        raise ValueError('Found %d bad images! \n %s' % (len(bad_images), '\n'.join(bad_images)))
