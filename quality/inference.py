"""Run model inference to annotate input images with patch/image predictions.

Example usage:
  python quality/run_inference.py \
    --eval_directory /tmp/ \
    --model_ckpt_file /path/model.ckpt \
    --image_globs_list "/images/*"

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

import PIL
import numpy as np
import png
import skimage.external.tifffile
import tensorflow

import data_provider
import constants
import dataset_creation
import evaluation

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

flags = tensorflow.app.flags

flags.DEFINE_string('eval_directory', None,
                    'Output directory to save results to.')
flags.DEFINE_string('model_ckpt_file', None,
                    'Path to Tensorflow model .ckpt file.')
flags.DEFINE_string('image_globs_list', None, 'Comma separated list of string'
                    'globs to images for inference.')
flags.DEFINE_float('image_brightness_scale', 1.0,
                   'Multiplicative exposure value.')
flags.DEFINE_integer(
    'image_width', None,
    'Integer, width to crop to. Must be multiple of model_patch_width.')
flags.DEFINE_integer(
    'image_height', None,
    'Integer, height to crop to. Must be multiple of model_patch_width.')
flags.DEFINE_integer('model_patch_width', 84,
                     'The image patch width, in pixels, for model input.')
flags.DEFINE_integer('num_classes', 11, 'Number of model classes')
flags.DEFINE_boolean('show_plots', False, 'Whether to show plots')
flags.DEFINE_integer('shard_num', 1, 'Job task number')
flags.DEFINE_integer('num_shards', 1, 'Total number of job tasks')
flags.DEFINE_string('probability_aggregation_method', evaluation.METHOD_AVERAGE,
                    'Method for aggregating probabilities.')
flags.DEFINE_integer('inference_model_id', 0, 'Model ID.')

FLAGS = flags.FLAGS

_SPLIT_NAME = 'test'
_TFRECORD_FILE_PATTERN = 'data_%s-%05d-of-%05d.tfrecord'


def patch_values_to_mask(values, patch_width):
  """Construct a mask from an array of patch values.

  Args:
    values: A uint16 2D numpy array.
    patch_width: Width in pixels of each patch.

  Returns:
    The  mask, a uint16 numpy array of width patch_width *
    values.shape[0].

  Raises:
    ValueError: If the input values are invalid.
  """
  if values.dtype != np.uint16 or len(values.shape) != 2:
    logging.info('dtype: %s shape: %s', values.dtype, values.shape)
    raise ValueError('Input must be a 2D np.uint16 array.')

  patches_per_column = values.shape[0]
  patches_per_row = values.shape[1]

  mask = np.zeros(
      (patches_per_column * patch_width, patches_per_row * patch_width),
      dtype=np.uint16)

  for i in range(patches_per_column):
    for j in range(patches_per_row):
      ymin = i * patch_width
      xmin = j * patch_width
      mask[ymin:ymin + patch_width, xmin:xmin + patch_width] = values[i, j]

  return mask


def save_masks_and_annotated_visualization(orig_name,
                                           output_directory,
                                           prediction,
                                           certainties,
                                           np_images,
                                           np_probabilities,
                                           np_labels,
                                           patch_width,
                                           image_height,
                                           image_width,
                                           show_plots=False):
  """For a prediction on a single image, save the output masks and images.

  Args:
    orig_name: String, full path to original input image.
    output_directory: String, path to directory for outputs.
    prediction: Integer, index of predicted class.
    certainties: Dictionary mapping certainty type (string) to float value.
    np_images: Numpy array of patches of shape (num_patches, width, width, 1).
    np_probabilities: Numpy array of shape (num_patches, num_classes), the
      probabilities predicted by the model for each class.
    np_labels: Integer numpy array of shape (num_patches) indicating true class.
      The true class must be the same for all patches.
    patch_width: Integer, width of image patches.
    image_height: Integer, the image height.
    image_width: Integer, the image width.
    show_plots: Whether to show plots (use with Colab).

  Raises:
    ValueError: If the image to annotate cannot be found or opened.
  """

  if not os.path.isfile(orig_name):
    raise ValueError('File for annotating does not exist: %s.' % orig_name)
  file_extension = os.path.splitext(orig_name)[1]
  if file_extension == '.png':
    output_width, output_height = PIL.Image.open(orig_name, 'r').size
  elif file_extension == '.tif':
    output_height, output_width = skimage.external.tifffile.TiffFile(orig_name, 'r').asarray().shape
  else:
    raise ValueError('Unsupported file extension %s', file_extension)
  logging.info('Original image size %d x %d', output_height, output_width)

  def pad_and_save_image(im, image_output_path):
    """Pad a 2D or 3D image (numpy array) to match the original and save."""
    # The image is either a greyscale 16-bit mask, or 8-bit RGB color.
    is_greyscale_mask = len(im.shape) == 2

    y_pad = output_height - im.shape[0]
    x_pad = output_width - im.shape[1]
    pad_size = ((0, y_pad), (0, x_pad)) if is_greyscale_mask else (
        (0, y_pad), (0, x_pad), (0, 0))
    im_padded = np.pad(im, pad_size, 'constant')

    if not is_greyscale_mask:
      img = PIL.Image.fromarray(im_padded)
      img.save(image_output_path)
    else:
      with open(image_output_path, 'w') as f:
        writer = png.Writer(
            width=im_padded.shape[1],
            height=im_padded.shape[0],
            bitdepth=16,
            greyscale=True)
        writer.write(f, im_padded.tolist())

  orig_name_png = os.path.splitext(os.path.basename(orig_name))[0] + '.png'
  visualized_image_name = ('actual%g_pred%g_mean_certainty=%0.3f' +
                           (constants.ORIG_IMAGE_FORMAT % orig_name_png))
  output_path = (os.path.join(output_directory, visualized_image_name) %
                 (np_labels[0], prediction, certainties['mean']))

  annotated_visualization = np.squeeze(
      evaluation.visualize_image_predictions(
          np_images,
          np_probabilities,
          np_labels,
          image_height,
          image_width,
          show_plot=show_plots,
          output_path=None))

  # Pad and save visualization.
  pad_and_save_image(annotated_visualization, output_path)

  def save_mask_from_patch_values(values, mask_format):
    """Convert patch values to mask, pad and save."""
    if np.min(values) < 0 or np.max(values) > np.iinfo(np.uint16):
      raise ValueError('Mask value out of bounds.')
    values = values.astype(np.uint16)
    reshaped_values = values.reshape((image_height / patch_width,
                                      image_width / patch_width))
    mask = patch_values_to_mask(reshaped_values, patch_width)
    pad_and_save_image(mask,
                       os.path.join(output_directory,
                                    mask_format % orig_name_png))

  # Create, pad and save masks.
  certainties = evaluation.certainties_from_probabilities(np_probabilities)
  certainties = np.round(certainties *
                         np.iinfo(np.uint16).max).astype(np.uint16)
  save_mask_from_patch_values(certainties, constants.CERTAINTY_MASK_FORMAT)

  predictions = np.argmax(np_probabilities, 1)
  save_mask_from_patch_values(predictions, constants.PREDICTIONS_MASK_FORMAT)

  valid_pixel_regions = np.ones(
      predictions.shape, dtype=np.uint16) * np.iinfo(np.uint16).max
  save_mask_from_patch_values(valid_pixel_regions, constants.VALID_MASK_FORMAT)


def run_model_inference(model_ckpt_file, probabilities, labels, images,
                        output_directory, image_paths, num_samples,
                        image_height, image_width, show_plots, shard_num,
                        num_shards, patch_width, aggregation_method):
  """Run a previously trained model on images."""
  logging.info('Running inference and writing inference results to \n%s',
               os.path.dirname(output_directory))

  if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

  aggregate_labels = []
  patch_labels = []

  model_directory = os.path.dirname(model_ckpt_file)
  if not os.path.isdir(model_directory):
    logging.fatal('Model checkpoint directory does not exist.')

  saver = tensorflow.train.Saver()
  with tensorflow.Session() as sess:
    logging.info('Restoring checkpoint %s', model_ckpt_file)
    saver.restore(sess, model_ckpt_file)
    coord = tensorflow.train.Coordinator()
    threads = tensorflow.train.start_queue_runners(sess=sess, coord=coord)
    logging.info('Started queue_runners.')

    for i in range(num_samples):
      logging.info('Running inference on sample  %d.', i)
      [np_probabilities, np_labels, np_images,
       np_image_paths] = sess.run([probabilities, labels, images, image_paths])

      (prediction, certainties,
       probabilities_i) = evaluation.aggregate_prediction_from_probabilities(
           np_probabilities, aggregation_method)

      # Each name must be unique since all workers write to same directory.
      orig_name = np_image_paths[0][0] if np_image_paths[0][0] else (
          'not_available_%03d_%07d.png' % shard_num, i)

      save_masks_and_annotated_visualization(
          orig_name, output_directory, prediction, certainties, np_images,
          np_probabilities, np_labels, patch_width, image_height, image_width,
          show_plots)

      if i == 0:
        patch_probabilities = np_probabilities
        aggregate_probabilities = np.expand_dims(probabilities_i, 0)
        orig_names = []
        all_certainties = {}
        for k in evaluation.CERTAINTY_TYPES.values():
          all_certainties[k] = []
      else:
        patch_probabilities = np.concatenate((patch_probabilities,
                                              np_probabilities), 0)
        aggregate_probabilities = np.concatenate(
            (aggregate_probabilities, np.expand_dims(probabilities_i, 0)))
      orig_names.append(orig_name)
      for k, v in certainties.items():
        all_certainties[k].append(v)

      aggregate_labels.append(np_labels[0])
      patch_labels += list(np_labels)

    aggregate_predictions = list(np.argmax(aggregate_probabilities, 1))
    logging.info('Inference output to %s.', output_directory)

    logging.info('Done evaluating model.')

    output_file = (os.path.join(output_directory, 'results-%05d-of-%05d.csv') %
                   (shard_num, num_shards))
    evaluation.save_inference_results(aggregate_probabilities, aggregate_labels,
                                      all_certainties, orig_names,
                                      aggregate_predictions, output_file)

    # If we're not sharding, save out accuracy statistics.
    if num_shards == 1:
      save_confusion = not np.any(aggregate_labels < 0)
      evaluation.save_result_plots(aggregate_probabilities, aggregate_labels,
                                   save_confusion, output_directory,
                                   patch_probabilities, patch_labels)
    logging.info('Stopping threads')
    coord.request_stop()
    coord.join(threads)
    logging.info('Threads stopped')


def build_tfrecord_from_pngs(image_globs_list, use_unlabeled_data, num_classes,
                             eval_directory, image_background_value,
                             image_brightness_scale, shard_num, num_shards,
                             image_width, image_height):
  """Build a TFRecord from pngs either from synthetic images or a directory."""

  # Generate a local TFRecord
  tfrecord_file_pattern = _TFRECORD_FILE_PATTERN % ('%s', shard_num, num_shards)

  num_samples_converted = dataset_creation.dataset_to_examples_in_tfrecord(
      image_globs_list,
      eval_directory,
      tfrecord_file_pattern % _SPLIT_NAME,
      num_classes,
      image_width=image_width,
      image_height=image_height,
      max_images=1e6,
      randomize=False,
      image_background_value=image_background_value,
      image_brightness_scale=image_brightness_scale,
      shard_num=shard_num,
      num_shards=num_shards,
      normalize=False,
      use_unlabeled_data=use_unlabeled_data)
  logging.info('Created TFRecord with %g examples.', num_samples_converted)

  return os.path.join(eval_directory, tfrecord_file_pattern)


def main(_):
  if FLAGS.eval_directory is None:
    logging.fatal('Eval directory required.')
  if FLAGS.model_ckpt_file is None:
    logging.fatal('Model checkpoint file required.')
  if FLAGS.image_globs_list is None:
    logging.fatal('Must provide image globs list.')

  if not os.path.isdir(FLAGS.eval_directory):
    os.makedirs(FLAGS.eval_directory)

  image_globs_list = FLAGS.image_globs_list.split(',')
  use_unlabeled_data = True

  # Input images will be cropped to image_height x image_width.
  image_size = dataset_creation.image_size_from_glob(image_globs_list[0],
                                                     FLAGS.model_patch_width)
  if FLAGS.image_width is not None and FLAGS.image_height is not None:
    image_width = int(FLAGS.model_patch_width * np.floor(
        FLAGS.image_width / FLAGS.model_patch_width))
    image_height = int(FLAGS.model_patch_width * np.floor(
        FLAGS.image_height / FLAGS.model_patch_width))

    if image_width > image_size.width or image_height > image_size.height:
      raise ValueError(
          'Specified (image_width, image_height) = (%d, %d) exceeds valid '
          'dimensions (%d, %d).' % (image_width, image_height, image_size.width,
                                    image_size.height))
  else:
    image_width = image_size.width
    image_height = image_size.height

  # All patches evaluated in a batch correspond to one single input image.
  batch_size = int(image_width * image_height / (FLAGS.model_patch_width**2))

  logging.info(('Using batch_size=%d for image_width=%d, '
                'image_height=%d, model_patch_width=%d'), batch_size,
               image_width, image_height, FLAGS.model_patch_width)

  tfexamples_tfrecord = build_tfrecord_from_pngs(
      image_globs_list, use_unlabeled_data, FLAGS.num_classes,
      FLAGS.eval_directory, FLAGS.image_background_value,
      FLAGS.image_brightness_scale, FLAGS.shard_num, FLAGS.num_shards,
      image_width, image_height)

  num_samples = data_provider.get_num_records(tfexamples_tfrecord % _SPLIT_NAME)
  logging.info('TFRecord has %g samples.', num_samples)

  g = tensorflow.Graph()
  with g.as_default():
    images, one_hot_labels, image_paths, _ = data_provider.provide_data(
        tfexamples_tfrecord,
        split_name=_SPLIT_NAME,
        batch_size=batch_size,
        num_classes=FLAGS.num_classes,
        image_width=image_width,
        image_height=image_height,
        patch_width=FLAGS.model_patch_width,
        randomize=False,
        num_threads=1)

    model_metrics = evaluation.get_model_and_metrics(
        images,
        num_classes=FLAGS.num_classes,
        one_hot_labels=one_hot_labels,
        is_training=False,
        model_id=FLAGS.inference_model_id)

    run_model_inference(
        FLAGS.model_ckpt_file,
        model_metrics.probabilities,
        model_metrics.labels,
        images,
        os.path.join(FLAGS.eval_directory, 'miq_result_images'),
        image_paths,
        num_samples,
        image_height,
        image_width,
        FLAGS.show_plots,
        shard_num=FLAGS.shard_num,
        num_shards=FLAGS.num_shards,
        patch_width=FLAGS.model_patch_width,
        aggregation_method=FLAGS.probability_aggregation_method)

  # Delete TFRecord to save disk space.
  tfrecord_path = tfexamples_tfrecord % _SPLIT_NAME
  os.remove(tfrecord_path)
  logging.info('Deleted %s', tfrecord_path)


if __name__ == '__main__':
  tensorflow.app.run()
