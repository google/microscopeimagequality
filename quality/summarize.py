r"""Aggregate and summarize model inference results.

Requires the prediction-annotated .png images and .csv files from
`quality predict`. The output is in a 'summary' subdirectory, and includes an
aggregated .csv file and various summary images.

Example usage:
  quality summarize <path_to_eval_directory>
"""

import logging
import os
import sys

import matplotlib
import matplotlib.pyplot
import numpy
import skimage.io

import quality.constants
import quality.evaluation

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# matplotlib.use('Agg')

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
    if numpy.min(values) < 0.0 or numpy.max(values) > 1.0:
        raise ValueError('Input values out of range.')
    matplotlib.pyplot.figure()
    _, _, patches = matplotlib.pyplot.hist(values, bins=bins, range=(0.0, 1.0), color='gray')

    alpha_index = numpy.array(range(1, bins)).astype(numpy.float32) / (bins - 1)
    for a, p in zip(alpha_index, patches):
        matplotlib.pyplot.setp(p, 'alpha', a)

    matplotlib.pyplot.xlim(0.0, 1.0)

    matplotlib.pyplot.tick_params(bottom=False, left=False, top=False, right=False)
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel(ylabel)
    matplotlib.pyplot.grid('off')
    matplotlib.pyplot.savefig(save_path, bbox_inches='tight')


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
        if numpy.any(mask):
            color = matplotlib.pyplot.cm.hsv(float(c) / num_classes)
            alpha = _get_alpha(numpy.sum(mask))
            logging.info('class %d, alpha %g counts %d', c, alpha, numpy.sum(mask))
            plot_scatter(
                numpy.array(certainties1)[mask],
                numpy.array(certainties2)[mask], label1, label2, color, alpha)


def plot_scatter(x, y, xlabel, ylabel, color, alpha):
    """Plot scatter plot."""
    matplotlib.pyplot.scatter(x, y, alpha=alpha, s=2.5, c=color, linewidths=0)

    matplotlib.pyplot.grid('off')
    matplotlib.pyplot.tick_params(
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
        top=False,
        right=False)
    matplotlib.pyplot.ylim([0.0, 1.0])
    matplotlib.pyplot.xlim([0.0, 1.0])
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel(ylabel)


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
    matplotlib.pyplot.figure(figsize=(fig_width, fig_width))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if i > j:
                matplotlib.pyplot.subplot(num_keys, num_keys, 1 + i * num_keys + j)
                _make_scatter_subplot(num_classes, predictions, certainties[k2],
                                      certainties[k1], k2
                                      if i == num_keys - 1 else '', k1
                                      if j == 0 else '')
        logging.info('Certainty %s has min %g, mean %g, max %g.', k1,
                     numpy.min(certainties[k1]),
                     numpy.mean(certainties[k1]), numpy.max(certainties[k1]))
    matplotlib.pyplot.subplots_adjust(hspace=0.05, wspace=0.05)
    matplotlib.pyplot.savefig(save_path, bbox_inches='tight', dpi=600)


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
        if (quality.constants.ORIG_IMAGE_FORMAT + '.png') % orig_name in name:
            filename_index = index
    if filename_index is None:
        raise ValueError('File %s not found' % orig_name)
    annotated_filename = all_files[filename_index]

    image = skimage.io.imread(os.path.join(experiment_path, annotated_filename))

    mask_path = os.path.join(experiment_path, quality.constants.VALID_MASK_FORMAT % orig_name + '.png')

    # if not os.path.isdir(mask_path):
    #     logging.info('No mask found at %s', mask_path)
    # else:
    mask = skimage.io.imread(mask_path)
    # Get the upper-left crop that is valid (where mask > 0).
    max_valid_row = numpy.argwhere(numpy.sum(mask, 1))[-1][0]
    max_valid_column = numpy.argwhere(numpy.sum(mask, 0))[-1][0]
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

    probabilities = numpy.identity(num_classes, dtype=numpy.float32)
    probabilities = numpy.tile(probabilities, [num_classes, 1])
    patch_width = quality.evaluation.BORDER_SIZE // 2
    patches = numpy.zeros((num_classes ** 2, patch_width, patch_width, 1), dtype=numpy.float32)
    # Make up some dummy labels.
    labels = [0] * num_classes ** 2
    image_shape = (num_classes * patch_width, num_classes * patch_width)

    image = quality.evaluation.get_rgb_image(1.0, patches, probabilities, labels,
                                     image_shape)
    image = image[quality.evaluation.BORDER_SIZE:quality.evaluation.BORDER_SIZE + patch_width, :]
    matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(image, interpolation='nearest')
    matplotlib.pyplot.grid('off')
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.savefig(path, bbox_inches='tight')
    matplotlib.pyplot.close()


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
    quality.evaluation.save_inference_results(probabilities, labels, certainties,
                                      orig_names, predictions,
                                      os.path.join(output_path, 'results_all.csv'))

    logging.info('Generating simple result plot.')
    save_confusion = not numpy.any(numpy.array(labels) < 0)
    quality.evaluation.save_result_plots(probabilities, labels, save_confusion,
                                 output_path_all_plots)

    predictions = numpy.array(predictions)
    num_classes = probabilities.shape[1]

    _save_color_legend(num_classes, os.path.join(output_path, 'color_legend.png'))

    plot_certainties(certainties, predictions, num_classes,
                     os.path.join(output_path_all_plots,
                                  'certainty_scatter_plot_all_certainties.png'))

    certainties_subset = {k: certainties[k] for k in ['mean', 'aggregate']}
    plot_certainties(certainties_subset, predictions, num_classes,
                     os.path.join(output_path, 'certainty_scatter_plot.png'))

    # Generate and save histograms for predictions and certainties.

    quality.evaluation.save_prediction_histogram(
        predictions,
        os.path.join(output_path, 'histogram_predictions.jpg'), num_classes)
    quality.evaluation.save_prediction_histogram(
        predictions,
        os.path.join(output_path, 'histogram_predictions_log.jpg'),
        num_classes,
        log=True)

    for kind in quality.evaluation.CERTAINTY_TYPES.values():
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
    image[-1 * quality.evaluation.BORDER_SIZE:, :, :] = (
        image[-1 * quality.evaluation.BORDER_SIZE:, :, :].astype(numpy.float32) *
        label_intensity).astype(image.dtype)

    # Make bottom border larger.
    border_size = max(quality.evaluation.BORDER_SIZE,
                      int(_BORDER_FRACTION * image.shape[0]))
    image[-1 * border_size:, :, :] = numpy.tile(image[-1:, :, :], (border_size, 1,
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
        numpy.random.shuffle(indices)
    elif 'certainty' in rank_method:
        class_certainties = numpy.array(certainties)[predictions == predicted_class]
        indices = indices[numpy.argsort(class_certainties)]
        if 'certainty_most' in rank_method:
            indices = indices[::-1]
        elif 'certainty_least_to_most' in rank_method:
            stride = indices.shape[0] // num_plots_in_row
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

    predictions = numpy.array(predictions)
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

            matplotlib.pyplot.imshow(image)
            matplotlib.pyplot.tick_params(labelbottom=False, labelleft=False)
            matplotlib.pyplot.grid('off')
            matplotlib.pyplot.axis('off')

        def subplot(nrows, ncols, num):
            """Makes a subplot and logs the (row, column) with 0-indexing."""
            matplotlib.pyplot.subplot(nrows, ncols, num)
            f.write('%d, %d ' % ((num - 1) / ncols, (num - 1) % ncols))

        def savefig(path):
            """Saves figure and logs the path."""
            matplotlib.pyplot.subplots_adjust(hspace=0.01, wspace=0.01)
            matplotlib.pyplot.savefig(path, bbox_inches='tight')
            matplotlib.pyplot.close()
            f.write('%s\n\n' % path)

        def setup_new_montage_figure(nrows, ncols):
            """New figure with blank subplot at corners to fix figure shape."""
            matplotlib.pyplot.figure(figsize=(_FIG_WIDTH, _FIG_WIDTH))
            matplotlib.pyplot.subplot(nrows, ncols, 1)
            matplotlib.pyplot.axis('off')
            matplotlib.pyplot.subplot(nrows, ncols, nrows * ncols)
            matplotlib.pyplot.axis('off')

        def montage_by_class_rank(rank_method, certainties, num_per_class=10):
            """Montage select images per class ranked by a particular method."""
            setup_new_montage_figure(num_classes, num_per_class)
            for i in range(num_classes):
                class_indices = numpy.array(range(num_samples))[predictions == i]
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
            boundaries = numpy.linspace(0.0, 1.0, bins_per_class + 1)
            setup_new_montage_figure(num_classes, bins_per_class)
            for i in range(num_classes):
                for j in range(bins_per_class):
                    mask = (predictions == i) & (certainties >= boundaries[j]) & (
                        certainties < boundaries[j + 1])
                    bin_indices = numpy.array(range(num_samples))[mask]
                    bin_certainties = numpy.array(certainties)[mask]
                    if bin_indices.shape[0] == 0:
                        continue
                    # Use the approximate median value in the bin.
                    bin_indices = bin_indices[numpy.argsort(bin_certainties)]
                    index = bin_indices[len(bin_indices) // 2]
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
            indices = numpy.argsort(certainties)
            width = min(len(certainties), 8)
            montage_first_several(width, indices, 'least_%s_certainty' % kind)
            montage_first_several(width, indices[::-1], 'most_%s_certainty' % kind)

        # Now actually generate the montages.
        montage_by_class_rank('random', certainties['mean'])
        for certainty in quality.evaluation.CERTAINTY_TYPES.values():
            logging.info('Generating montages for certainty type: %s.', certainty)
            montage_by_certainty(certainties[certainty], certainty)
            plot_most_least_certain(certainties[certainty], certainty)

    logging.info('Done saving summary montages.')
