r"""Aggregate and summarize model inference results.

Requires the prediction-annotated .png images and .csv files from
run_inference.py. The output is in a 'summary' subdirectory, and includes an
aggregated .csv file and various summary images.

Example usage:
  python summarize_inference.py \
     --experiment_directory <path_to_eval_directory>
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

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from quality import constants
from quality import miq_eval

flags = tf.app.flags

flags.DEFINE_string('experiment_directory', None,
                    'Directory of inference .csv and .png outputs')

FLAGS = flags.FLAGS

# Thickness of prediction border annotation, as fraction of image height.
_BORDER_FRACTION = 0.08

_FIG_WIDTH = 60


def check_image_count_matches(experiment_path, num_images_expected):
  """Check the number of inference .png files is as expected.

  Args:
    experiment_path: String, path to experiment folder (e.g.
      path/to/miq_result_images).
    num_images_expected: Integer, number of expected images.
  """
  filenames = os.listdir(experiment_path)
  filenames_png = [f for f in filenames if '.png' in f and 'actual' in f]
  logging.info('num expected: %g, num png files: %g', num_images_expected,
               len(filenames_png))
  assert num_images_expected == len(filenames_png)


def _plot_histogram(values, xlabel, ylabel, save_path, bins=10):
  """Plot histogram for values in [0.0, 1.0].

  Args:
    values: List of floats.
    xlabel: String, x-axis label.
    ylabel: String, y-axis label.
    save_path: String, path to save the figure.
    bins: Integer, number of histogram bins.

  Raises:
    ValueError: If input values are out of range.
  """
  if np.min(values) < 0.0 or np.max(values) > 1.0:
    raise ValueError('Input values out of range.')
  plt.figure()
  _, _, patches = plt.hist(values, bins=bins, range=(0.0, 1.0), color='gray')

  alpha_index = np.array(range(1, bins)).astype(np.float32) / (bins - 1)
  for a, p in zip(alpha_index, patches):
    plt.setp(p, 'alpha', a)

  plt.xlim(0.0, 1.0)

  plt.tick_params(bottom=False, left=False, top=False, right=False)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid('off')
  plt.savefig(save_path, bbox_inches='tight')


def _make_scatter_subplot(num_classes, predictions, certainties1, certainties2,
                          label1, label2):
  """Make a single scatter subplot.

  Args:
    num_classes: Integer, total number of possible predicted classes.
    predictions: List of integers in [0, num_classes).
    certainties1: List of floats in [0.0, 1.0].
    certainties2: List of floats in [0.0, 1.0].
    label1: String, text axes label for certainties1.
    label2: String, text axes label for certainties2.
  """
  for c in range(num_classes):
    mask = predictions == c
    if np.any(mask):
      color = plt.cm.hsv(float(c) / num_classes)
      alpha = _get_alpha(np.sum(mask))
      logging.info('class %d, alpha %g counts %d', c, alpha, np.sum(mask))
      plot_scatter(
          np.array(certainties1)[mask],
          np.array(certainties2)[mask], label1, label2, color, alpha)


def plot_scatter(x, y, xlabel, ylabel, color, alpha):
  """Plot scatter plot."""
  plt.scatter(x, y, alpha=alpha, s=2.5, c=color, linewidths=0)

  plt.grid('off')
  plt.tick_params(
      labelbottom=False,
      labelleft=False,
      bottom=False,
      left=False,
      top=False,
      right=False)
  plt.ylim([0.0, 1.0])
  plt.xlim([0.0, 1.0])
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)


def _get_alpha(count):
  """Optimal scatter plot alpha for a given number of points."""
  # These were empirically determined.
  if count >= 1e4:
    alpha = 0.03
  if count < 1e4:
    alpha = 0.08
  if count < 1e3:
    alpha = 0.3
  if count < 5e2:
    alpha = 0.5
  if count < 1e2:
    alpha = 0.9
  if count < 1e1:
    alpha = 1.0
  return alpha


def plot_certainties(certainties, predictions, num_classes, save_path):
  """Generate scatter plots of certainties.

  Args:
    certainties: Dictionary mapping string certainty type to list of float
      certainty values in [0.0, 1.0].
    predictions: List of integer predictions in [0, num_classes).
    num_classes: Integer, total number of possible predicted classes.
    save_path: String, path to save the figure.
  """
  keys = sorted(certainties.keys())
  num_keys = len(keys)
  fig_width = int(2.5 * len(certainties.keys()))
  plt.figure(figsize=(fig_width, fig_width))
  for i, k1 in enumerate(keys):
    for j, k2 in enumerate(keys):
      if i > j:
        plt.subplot(num_keys, num_keys, 1 + i * num_keys + j)
        _make_scatter_subplot(num_classes, predictions, certainties[k2],
                              certainties[k1], k2
                              if i == num_keys - 1 else '', k1
                              if j == 0 else '')
    logging.info('Certainty %s has min %g, mean %g, max %g.', k1,
                 np.min(certainties[k1]),
                 np.mean(certainties[k1]), np.max(certainties[k1]))
  plt.subplots_adjust(hspace=0.05, wspace=0.05)
  plt.savefig(save_path, bbox_inches='tight', dpi=600)


def _read_valid_part_of_annotated_image(experiment_path, orig_name):
  """Reads in an image and returns the valid region.

  The valid region defines the pixels over which the inference has been done.

  Args:
    experiment_path: String, path to inference annotated output images.
    orig_name: Original filename without path and extension of image to be
        found.

  Returns:
    An image as a numpy array, with the valid region only if a mask file
      exists.

  Raises:
      ValueError: If the image is not found.
  """
  filename_index = None
  all_files = os.listdir(experiment_path)
  # Find the annotated image file. There is exactly one.
  for index, name in enumerate(all_files):
    # Exclude all masks from search.
    if (constants.ORIG_IMAGE_FORMAT + '.png') % orig_name in name:
      filename_index = index
  if filename_index is None:
    raise ValueError('File %s not found' % orig_name)
  annotated_filename = all_files[filename_index]

  image = np.array(
      Image.open(os.path.join(experiment_path, annotated_filename)))
  mask_path = os.path.join(experiment_path,
                           constants.VALID_MASK_FORMAT % orig_name + '.png')
  if not os.path.isdir(mask_path):
    logging.info('No mask found at %s', mask_path)
  else:
    mask = np.array(Image.open(mask_path))
    # Get the upper-left crop that is valid (where mask > 0).
    max_valid_row = np.argwhere(np.sum(mask, 1))[-1]
    max_valid_column = np.argwhere(np.sum(mask, 0))[-1]
    image = image[:max_valid_row, :max_valid_column]

  return image


def _save_color_legend(num_classes, path):
  """Save a legend for interpreting the predicted class colors.

  This produces an image with a color bar denoting the colors of each of the
  predicted classes.

  Args:
    num_classes: Integer, the number of classes in the prediction task.
    path: Path to png file to save the figure.
  """

  probabilities = np.identity(num_classes, dtype=np.float32)
  probabilities = np.tile(probabilities, [num_classes, 1])
  patch_width = miq_eval.BORDER_SIZE / 2
  patches = np.zeros(
      (num_classes**2, patch_width, patch_width, 1), dtype=np.float32)
  # Make up some dummy labels.
  labels = [0] * num_classes**2
  image_shape = (num_classes * patch_width, num_classes * patch_width)

  image = miq_eval.get_rgb_image(1.0, patches, probabilities, labels,
                                 image_shape)
  image = image[miq_eval.BORDER_SIZE:miq_eval.BORDER_SIZE + patch_width, :]
  plt.figure()
  plt.imshow(image, interpolation='nearest')
  plt.grid('off')
  plt.axis('off')
  plt.savefig(path, bbox_inches='tight')
  plt.close()


def save_histograms_scatter_plots_and_csv(probabilities,
                                          labels,
                                          certainties,
                                          orig_names,
                                          predictions,
                                          output_path,
                                          output_path_all_plots=None):
  """Visualize and save various summary plots and an aggregated .csv file.

  Args:
    probabilities: Numpy float array of shape [num_samples x num_classes].
    labels: List of integers, the actual classes, length num_samples.
    certainties: Dict of lists of floats, the certainties, each length
      num_samples.
    orig_names: List of strings, the original names, length num_samples.
    predictions: List of integers, the predicted classes, length
      num_samples.
    output_path: String, path to folder to save summary results.
    output_path_all_plots: String, path to folder to save less useful results.
  """
  if output_path_all_plots is None:
    output_path_all_plots = output_path

  logging.info('Saving inference results in single .csv file.')
  miq_eval.save_inference_results(probabilities, labels, certainties,
                                  orig_names, predictions,
                                  os.path.join(output_path, 'results_all.csv'))

  logging.info('Generating simple result plot.')
  save_confusion = not np.any(np.array(labels) < 0)
  miq_eval.save_result_plots(probabilities, labels, save_confusion,
                             output_path_all_plots)

  predictions = np.array(predictions)
  num_classes = probabilities.shape[1]

  _save_color_legend(num_classes, os.path.join(output_path, 'color_legend.png'))

  plot_certainties(certainties, predictions, num_classes,
                   os.path.join(output_path_all_plots,
                                'certainty_scatter_plot_all_certainties.png'))

  certainties_subset = {k: certainties[k] for k in ['mean', 'aggregate']}
  plot_certainties(certainties_subset, predictions, num_classes,
                   os.path.join(output_path, 'certainty_scatter_plot.png'))

  # Generate and save histograms for predictions and certainties.

  miq_eval.save_prediction_histogram(
      predictions,
      os.path.join(output_path, 'histogram_predictions.jpg'), num_classes)
  miq_eval.save_prediction_histogram(
      predictions,
      os.path.join(output_path, 'histogram_predictions_log.jpg'),
      num_classes,
      log=True)

  for kind in miq_eval.CERTAINTY_TYPES.values():
    if kind == 'aggregate':
      path = output_path
    else:
      path = output_path_all_plots
    _plot_histogram(certainties[kind], '%s prediction certainty' % kind,
                    'image count',
                    os.path.join(path, 'histogram_%s_certainty.jpg' % kind))

  logging.info('Done summarizing results')


def _adjust_image_annotation(image, label_intensity):
  """Adjusts the annotation at the bottom of the image."""
  # Change the intensity of the bottom border.
  image[-1 * miq_eval.BORDER_SIZE:, :, :] = (
      image[-1 * miq_eval.BORDER_SIZE:, :, :].astype(np.float32) *
      label_intensity).astype(image.dtype)

  # Make bottom border larger.
  border_size = max(miq_eval.BORDER_SIZE,
                    int(_BORDER_FRACTION * image.shape[0]))
  image[-1 * border_size:, :, :] = np.tile(image[-1:, :, :], (border_size, 1,
                                                              1))
  return image


def _rank_examples(indices, rank_method, certainties, predictions,
                   num_plots_in_row, predicted_class):
  """Rank the examples based on a ranking method.

  Args:
    indices: 1D numpy array of indices to rank.
    rank_method: String, the ranking method.
    certainties: List of floats, the certainties.
    predictions: 1D numpy array of the predicted class indices.
    num_plots_in_row: Int, number of plots in each row.
    predicted_class: Integer, the predicted class.

  Returns:
    The ranked indices as a 1D numpy array.

  Raises:
    ValueError: If the certainty rank method is invalid.
  """
  if rank_method == 'random':
    np.random.shuffle(indices)
  elif 'certainty' in rank_method:
    class_certainties = np.array(certainties)[predictions == predicted_class]
    indices = indices[np.argsort(class_certainties)]
    if 'certainty_most' in rank_method:
      indices = indices[::-1]
    elif 'certainty_least_to_most' in rank_method:
      stride = indices.shape[0] / num_plots_in_row
      indices = indices[:stride * num_plots_in_row:stride]
    elif 'certainty_least' in rank_method:
      pass
    else:
      raise ValueError('Invalid certainty rank method %s' % rank_method)
  else:
    raise ValueError('Invalid rank_method %s' % rank_method)
  return indices


def save_summary_montages(probabilities,
                          certainties,
                          orig_names,
                          predictions,
                          experiment_path,
                          output_path,
                          output_path_all_plots=None):
  """Visualize and save summary montage images.

  Args:
    probabilities: Numpy float array of shape [num_samples x num_classes].
    certainties: Dict of lists of floats, the certainties, each length
      num_samples.
    orig_names: List of strings, the original names, length num_samples.
    predictions: List of integers, the predicted classes, length
      num_samples.
    experiment_path: String, path to folder containing results.
    output_path: String, path to folder to save summary results.
    output_path_all_plots: String, path to folder to save less useful results.
  """
  if output_path_all_plots is None:
    output_path_all_plots = output_path

  predictions = np.array(predictions)
  num_samples, num_classes = probabilities.shape

  with open(
      os.path.join(output_path_all_plots, 'montage_image_paths.txt'), 'w') as f:

    f.write(('# This text file maps subplots in each summary image with the \n'
             '# original image path. Subplots are denoted by 0-indexed row \n'
             '# and column from upper left.\n\n'))

    def plot_image(index, label_intensity=1.0):
      """Read and plot inference image."""

      orig_name = os.path.splitext(os.path.basename(orig_names[index]))[0]
      f.write('%s\n' % orig_names[index])
      image = _read_valid_part_of_annotated_image(experiment_path, orig_name)

      image = _adjust_image_annotation(image, label_intensity)

      plt.imshow(image)
      plt.tick_params(labelbottom=False, labelleft=False)
      plt.grid('off')
      plt.axis('off')

    def subplot(nrows, ncols, num):
      """Makes a subplot and logs the (row, column) with 0-indexing."""
      plt.subplot(nrows, ncols, num)
      f.write('%d, %d ' % ((num - 1) / ncols, (num - 1) % ncols))

    def savefig(path):
      """Saves figure and logs the path."""
      plt.subplots_adjust(hspace=0.01, wspace=0.01)
      plt.savefig(path, bbox_inches='tight')
      plt.close()
      f.write('%s\n\n' % path)

    def setup_new_montage_figure(nrows, ncols):
      """New figure with blank subplot at corners to fix figure shape."""
      plt.figure(figsize=(_FIG_WIDTH, _FIG_WIDTH))
      plt.subplot(nrows, ncols, 1)
      plt.axis('off')
      plt.subplot(nrows, ncols, nrows * ncols)
      plt.axis('off')

    def montage_by_class_rank(rank_method, certainties, num_per_class=10):
      """Montage select images per class ranked by a particular method."""
      setup_new_montage_figure(num_classes, num_per_class)
      for i in range(num_classes):
        class_indices = np.array(range(num_samples))[predictions == i]
        num_plots_in_row = min(class_indices.shape[0], num_per_class)
        if num_plots_in_row == 0:
          continue
        class_indices = _rank_examples(class_indices, rank_method, certainties,
                                       predictions, num_plots_in_row, i)
        for j in range(num_plots_in_row):
          subplot(num_classes, num_per_class, 1 + i * num_per_class + j)
          plot_image(class_indices[j], certainties[class_indices[j]])
      savefig(os.path.join(output_path_all_plots, 'rank_%s.jpg' % rank_method))

    def montage_by_class_bin(rank_method, certainties, bins_per_class=10):
      """Montage one image per certainty bin for each class."""
      boundaries = np.linspace(0.0, 1.0, bins_per_class + 1)
      setup_new_montage_figure(num_classes, bins_per_class)
      for i in range(num_classes):
        for j in range(bins_per_class):
          mask = (predictions == i) & (certainties >= boundaries[j]) & (
              certainties < boundaries[j + 1])
          bin_indices = np.array(range(num_samples))[mask]
          bin_certainties = np.array(certainties)[mask]
          if bin_indices.shape[0] == 0:
            continue
          # Use the approximate median value in the bin.
          bin_indices = bin_indices[np.argsort(bin_certainties)]
          index = bin_indices[len(bin_indices) / 2]
          subplot(num_classes, bins_per_class, 1 + i * bins_per_class + j)
          plot_image(index, certainties[index])
      if rank_method == 'aggregate_certainty_least_to_most':
        path = output_path
      else:
        path = output_path_all_plots
      savefig(os.path.join(path, 'bin_%s.jpg' % rank_method))

    def montage_by_certainty(certainties, kind):
      montage_by_class_bin('%s_certainty_least_to_most' % kind, certainties)
      montage_by_class_rank('%s_certainty_least' % kind, certainties)
      montage_by_class_rank('%s_certainty_most' % kind, certainties)
      montage_by_class_rank('%s_certainty_least_to_most' % kind, certainties)

    def montage_first_several(num_subplots, sorted_indices, name):
      """Montages the first num_subplots^2 images."""
      setup_new_montage_figure(num_subplots, num_subplots)
      for i in range(num_subplots):
        for j in range(num_subplots):
          if i * num_subplots + j < len(sorted_indices):
            subplot(num_subplots, num_subplots, 1 + i * num_subplots + j)
            plot_image(sorted_indices[i * num_subplots + j])
      savefig(os.path.join(output_path_all_plots, '%s.jpg' % name))

    def plot_most_least_certain(certainties, kind):
      indices = np.argsort(certainties)
      width = min(len(certainties), 8)
      montage_first_several(width, indices, 'least_%s_certainty' % kind)
      montage_first_several(width, indices[::-1], 'most_%s_certainty' % kind)

    # Now actually generate the montages.
    montage_by_class_rank('random', certainties['mean'])
    for certainty in miq_eval.CERTAINTY_TYPES.values():
      logging.info('Generating montages for certainty type: %s.', certainty)
      montage_by_certainty(certainties[certainty], certainty)
      plot_most_least_certain(certainties[certainty], certainty)

  logging.info('Done saving summary montages.')


def main(_):
  if FLAGS.experiment_directory is None:
    logging.fatal('Experiment directory required.')

  (probabilities, labels, certainties, orig_names,
   predictions) = miq_eval.load_inference_results(FLAGS.experiment_directory)

  if not predictions:
    logging.fatal('No inference output found at %s.',
                  FLAGS.experiment_directory)

  check_image_count_matches(FLAGS.experiment_directory, len(predictions))

  output_path = os.path.join(FLAGS.experiment_directory, 'summary')
  if not os.path.isdir(output_path):
    os.makedirs(output_path)

  # Less useful plots go here.
  output_path_all_plots = os.path.join(output_path, 'additional_plots')
  if not os.path.isdir(output_path_all_plots):
    os.makedirs(output_path_all_plots)

  save_histograms_scatter_plots_and_csv(probabilities, labels, certainties,
                                        orig_names, predictions, output_path,
                                        output_path_all_plots)

  save_summary_montages(probabilities, certainties, orig_names, predictions,
                        FLAGS.experiment_directory, output_path,
                        output_path_all_plots)
  logging.info('Done summarizing results at %s', output_path)


if __name__ == '__main__':
  tf.app.run()
