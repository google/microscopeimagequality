from __future__ import print_function

"""Check that images specified by glob are valid for image quality inference.

Example usage:
  python validata_data.py \
    --image_globs_list "/tmp/folder_of_pngs_and_tifs/*" \
    --image_height=100 --image_width=100
"""
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

import logging
import os
import sys

import tensorflow

import dataset_creation

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

flags = tensorflow.app.flags

flags.DEFINE_string('image_globs_list', None, 'Comma separated list of globs to'
                                              'images for validating.')
flags.DEFINE_integer(
    'image_width', None,
    'Integer, width to crop to. Must be multiple of model_patch_width.')
flags.DEFINE_integer(
    'image_height', None,
    'Integer, height to crop to. Must be multiple of model_patch_width.')
flags.DEFINE_integer('model_patch_width', 84,
                     'The image patch width, in pixels, for model input.')
FLAGS = flags.FLAGS
_MAX_IMAGES_TO_VALIDATE = 1e6


def check_duplicate_image_name(image_paths):
    """Check that there are no duplicate names (without path or extension).

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
    """Check that the image dimensions are valid.

  A valid image has height and width no smaller than the specified height,
  width.

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
        image = dataset_creation.read_16_bit_greyscale(path)

        if image.shape[0] < image_height or image.shape[1] < image_width:
            bad_images.append(path)
            logging.info('Image %s dimension %s is too small.', path,
                         str(image.shape))

    logging.info('Done checking images')
    logging.info('Found %d bad images.', len(bad_images))
    if bad_images:
        raise ValueError('Found %d bad images! \n %s' %
                         (len(bad_images), '\n'.join(bad_images)))


def main(_):
    image_paths = []
    for glob in FLAGS.image_globs_list.split(','):
        print(glob)
        image_paths += dataset_creation.get_images_from_glob(
            glob, _MAX_IMAGES_TO_VALIDATE)
    logging.info('Found %d paths', len(image_paths))
    if len(image_paths) == 0:
        raise ValueError('No images found.')

    check_duplicate_image_name(image_paths)

    if FLAGS.image_width is None or FLAGS.image_height is None:
        height, width = dataset_creation.image_size_from_glob(
            FLAGS.image_globs_list[0], FLAGS.model_patch_width)
    else:
        height, width = FLAGS.image_height, FLAGS.image_width

    check_image_dimensions(image_paths, height, width)


if __name__ == '__main__':
    tensorflow.app.run()
