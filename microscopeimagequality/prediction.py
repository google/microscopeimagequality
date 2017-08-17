"""
Run model inference to annotate input images with patch/image predictions.

Example usage:
  quality predict \
    --checkpoint /path/model.ckpt \
    --output /tmp/ \
    "/images/*"

"""

import logging
import os
import sys

import numpy
import skimage.io
import tensorflow

import microscopeimagequality.constants
import microscopeimagequality.dataset_creation
import microscopeimagequality.evaluation

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

_SPLIT_NAME = 'test'

_TFRECORD_FILE_PATTERN = 'data_%s-%05d-of-%05d.tfrecord'

class ImageQualityClassifier(object):
  """Object for running image quality model inference.

  Attributes:
    graph: TensorFlow graph.
  """

  def __init__(self,
               model_ckpt,
               model_patch_side_length,
               num_classes,
               graph=None):
    """Initialize the model from a checkpoint.

    Args:
      model_ckpt: String, path to TensorFlow model checkpoint to load.
      model_patch_side_length: Integer, the side length in pixels of the square
        image passed to the model.
      num_classes: Integer, the number of classes the model predicts.
      graph: TensorFlow graph. If None, one will be created.
    """
    self._model_patch_side_length = model_patch_side_length
    self._num_classes = num_classes

    if graph is None:
      graph = tensorflow.Graph()
    self.graph = graph

    with self.graph.as_default():
      self._image_placeholder = tensorflow.placeholder(
          tensorflow.float32, shape=[None, None, 1])

      self._probabilities = self._probabilities_from_image(
          self._image_placeholder, model_patch_side_length, num_classes)

      self._sess = tensorflow.Session()
      saver = tensorflow.train.Saver()

      saver.restore(self._sess, model_ckpt)
    logging.info('Model restored from %s.', model_ckpt)

  def __del__(self):
    self._sess.close()

  def _probabilities_from_image(self, image_placeholder,
                                model_patch_side_length, num_classes):
    """Get probabilities tensor from input image tensor.

    Args:
      image_placeholder: Float32 tensor, placeholder for input image.
      model_patch_side_length: Integer, the side length in pixels of the square
        image passed to the model.
      num_classes: Integer, the number of classes the model predicts.

    Returns:
      Probabilities tensor, shape [num_classes] representing the predicted
      probabilities for each class.
    """
    labels_fake = tensorflow.zeros([self._num_classes])

    image_path_fake = tensorflow.constant(['unused'])
    tiles, labels, _ = _get_image_tiles_tensor(
        image_placeholder, labels_fake, image_path_fake,
        model_patch_side_length)

    model_metrics = microscopeimagequality.evaluation.get_model_and_metrics(
        tiles,
        num_classes=num_classes,
        one_hot_labels=labels,
        is_training=False)

    return model_metrics.probabilities

  def predict(self, image):
    """Run inference on an image.

    Args:
      image: Numpy float array, two-dimensional.

    Returns:
      A evaluation.WholeImagePrediction object.
    """
    feed_dict = {self._image_placeholder: numpy.expand_dims(image, 2)}
    [np_probabilities] = self._sess.run(
        [self._probabilities], feed_dict=feed_dict)

    return microscopeimagequality.evaluation.aggregate_prediction_from_probabilities(
        np_probabilities, microscopeimagequality.evaluation.METHOD_AVERAGE)
  
  def get_patch_predictions(self,  image):
    """Run inference on each patch in an image, returning each patch score.

    Args:
      image: Numpy float array, of shape (height, width).

    Returns:
      List of tuples, with (upper_left_row, upper_left_col, height, width
      evaluation.WholeImagePrediction) which denote the patch location,
      dimensions and predition result.
    """
    results = []
    w = microscopeimagequality.constants.PATCH_SIDE_LENGTH
    for i in range(0, image.shape[0] - w, w):
      for j in range(0, image.shape[1] - w, w):
        results.append((i, j, w, w, self.predict(image[i:i+w, j:j+w])))
    return results               
    
  def get_annotated_prediction(self, image):
    """Run inference to annotate the input image with patch predictions.

    Args:
      image: Numpy float array, two-dimensional.

    Returns:
      RGB image as uint8 numpy array of shape (image_height, image_width, 3),
      representing the upper left crop of the input image, where:
         image_height = floor(image.shape[0] / model_patch_side_length)
         image_width = floor(image.shape[1] / model_patch_side_length)
    """

    feed_dict = {self._image_placeholder: numpy.expand_dims(image, 2)}

    with self.graph.as_default():
      patches = _get_image_tiles_tensor(
          self._image_placeholder,
          tensorflow.constant([0]),
          tensorflow.constant([0]),
          patch_width=self._model_patch_side_length)[0]
      [np_probabilities, np_patches] = self._sess.run(
          [self._probabilities, patches], feed_dict=feed_dict)

    # We use '-1' to denote no true label exists.
    np_labels = -1 * numpy.ones((np_patches.shape[0]))
    return numpy.squeeze(
        microscopeimagequality.evaluation.visualize_image_predictions(
            np_patches,
            np_probabilities,
            np_labels,
            image.shape[0],
            image.shape[1],
            show_plot=False,
            output_path=None))

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
    if values.dtype != numpy.uint16 or len(values.shape) != 2:
        logging.info('dtype: %s shape: %s', values.dtype, values.shape)
        raise ValueError('Input must be a 2D np.uint16 array.')

    patches_per_column = values.shape[0]
    patches_per_row = values.shape[1]

    mask = numpy.zeros(
        (patches_per_column * patch_width, patches_per_row * patch_width),
        dtype=numpy.uint16)

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

    orig_name = orig_name.decode("utf-8")

    if not os.path.isfile(orig_name):
        raise ValueError('File for annotating does not exist: %s.' % orig_name)

    output_height, output_width = skimage.io.imread(orig_name).shape

    logging.info('Original image size %d x %d', output_height, output_width)

    def pad_and_save_image(im, image_output_path):
        """Pad a 2D or 3D image (numpy array) to match the original and save."""
        # The image is either a greyscale 16-bit mask, or 8-bit RGB color.
        is_greyscale_mask = len(im.shape) == 2

        y_pad = output_height - im.shape[0]
        x_pad = output_width - im.shape[1]
        pad_size = ((0, y_pad), (0, x_pad)) if is_greyscale_mask else (
            (0, y_pad), (0, x_pad), (0, 0))
        im_padded = numpy.pad(im, pad_size, 'constant')

        skimage.io.imsave(image_output_path, im_padded)

    orig_name_png = os.path.splitext(os.path.basename(orig_name))[0] + '.png'
    visualized_image_name = ('actual%g_pred%g_mean_certainty=%0.3f' +
                             (microscopeimagequality.constants.ORIG_IMAGE_FORMAT % orig_name_png))
    output_path = (os.path.join(output_directory, visualized_image_name) %
                   (np_labels[0], prediction, certainties['mean']))

    annotated_visualization = numpy.squeeze(
        microscopeimagequality.evaluation.visualize_image_predictions(
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
        if numpy.min(values) < 0 or numpy.max(values) > numpy.iinfo(numpy.uint16).max:
            raise ValueError('Mask value out of bounds.')
        values = values.astype(numpy.uint16)
        reshaped_values = values.reshape((image_height // patch_width, image_width // patch_width))
        mask = patch_values_to_mask(reshaped_values, patch_width)
        pad_and_save_image(mask, os.path.join(output_directory, mask_format % orig_name_png))

    # Create, pad and save masks.
    certainties = microscopeimagequality.evaluation.certainties_from_probabilities(np_probabilities)
    certainties = numpy.round(certainties *
                              numpy.iinfo(numpy.uint16).max).astype(numpy.uint16)
    save_mask_from_patch_values(certainties, microscopeimagequality.constants.CERTAINTY_MASK_FORMAT)

    predictions = numpy.argmax(np_probabilities, 1)
    save_mask_from_patch_values(predictions, microscopeimagequality.constants.PREDICTIONS_MASK_FORMAT)

    valid_pixel_regions = numpy.ones(
        predictions.shape, dtype=numpy.uint16) * numpy.iinfo(numpy.uint16).max
    save_mask_from_patch_values(valid_pixel_regions, microscopeimagequality.constants.VALID_MASK_FORMAT)

def _get_image_tiles_tensor(image, label, image_path, patch_width):
  """Gets patches that tile the input image, starting at upper left.

  Args:
    image: Input image tensor, size [height x width x 1].
    label: Input label tensor, size [num_classes].
    image_path: Input image path tensor, size [1].
    patch_width: Integer representing width of image patch.

  Returns:
    Tensors tiles, size [num_tiles x patch_width x patch_width x 1], labels,
    size [num_tiles x num_classes], and image_paths, size [num_tiles x 1].
  """
  tiles_before_reshape = tensorflow.extract_image_patches(
      tensorflow.expand_dims(image, dim=0), [1, patch_width, patch_width, 1],
      [1, patch_width, patch_width, 1], [1, 1, 1, 1], 'VALID')
  tiles = tensorflow.reshape(tiles_before_reshape, [-1, patch_width, patch_width, 1])

  labels = tensorflow.tile(tensorflow.expand_dims(label, dim=0), [tensorflow.shape(tiles)[0], 1])
  image_paths = tensorflow.tile(
      tensorflow.expand_dims(image_path, dim=0), [tensorflow.shape(tiles)[0], 1])
  return tiles, labels, image_paths

def run_model_inference( model_ckpt_file, probabilities, labels, images,
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

            [np_probabilities, np_labels, np_images, np_image_paths] = sess.run([probabilities, labels, images, image_paths])

            (prediction, certainties, probabilities_i) = microscopeimagequality.evaluation.aggregate_prediction_from_probabilities(np_probabilities, aggregation_method)

            # Each name must be unique since all workers write to same directory.
            orig_name = np_image_paths[0][0] if np_image_paths[0][0] else ('not_available_%03d_%07d.png' % shard_num, i)

            save_masks_and_annotated_visualization(orig_name, output_directory, prediction, certainties, np_images, np_probabilities, np_labels, patch_width, image_height, image_width, show_plots)

            if i == 0:
                patch_probabilities = np_probabilities
                aggregate_probabilities = numpy.expand_dims(probabilities_i, 0)
                orig_names = []
                all_certainties = {}
                for k in microscopeimagequality.evaluation.CERTAINTY_TYPES.values():
                    all_certainties[k] = []
            else:
                patch_probabilities = numpy.concatenate((patch_probabilities,
                                                         np_probabilities), 0)
                aggregate_probabilities = numpy.concatenate(
                    (aggregate_probabilities, numpy.expand_dims(probabilities_i, 0)))

            orig_names.append(orig_name)

            for k, v in certainties.items():
                all_certainties[k].append(v)

            aggregate_labels.append(np_labels[0])

            patch_labels += list(np_labels)

        aggregate_predictions = list(numpy.argmax(aggregate_probabilities, 1))

        logging.info('Inference output to %s.', output_directory)

        logging.info('Done evaluating model.')

        output_file = (os.path.join(output_directory, 'results-%05d-of-%05d.csv') % (shard_num, num_shards))

        microscopeimagequality.evaluation.save_inference_results(aggregate_probabilities, aggregate_labels, all_certainties, orig_names, aggregate_predictions, output_file)

        # If we're not sharding, save out accuracy statistics.
        if num_shards == 1:
            save_confusion = not numpy.any(numpy.asarray(aggregate_labels) < 0)

            microscopeimagequality.evaluation.save_result_plots(aggregate_probabilities, aggregate_labels, save_confusion, output_directory, patch_probabilities, patch_labels)

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

    num_samples_converted = microscopeimagequality.dataset_creation.dataset_to_examples_in_tfrecord(
        list_of_image_globs=image_globs_list,
        output_directory=eval_directory,
        output_tfrecord_filename=tfrecord_file_pattern % _SPLIT_NAME,
        num_classes=num_classes,
        image_width=image_width,
        image_height=image_height,
        max_images=1e6,
        randomize=False,
        image_background_value=image_background_value,
        image_brightness_scale=image_brightness_scale,
        shard_num=shard_num,
        num_shards=num_shards,
        normalize=False,
        use_unlabeled_data=use_unlabeled_data
    )

    logging.info('Created TFRecord with %g examples.', num_samples_converted)

    return os.path.join(eval_directory, tfrecord_file_pattern)
