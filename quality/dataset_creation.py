"""Functions for reading data and saving as TFExamples in TFRecord format.
"""

import collections
import glob
import logging
import os

import numpy
import skimage.io
import tensorflow
import tensorflow.core.example

import quality.data_provider

# Threshold for foreground objects (after background subtraction).
_FOREGROUND_THRESHOLD = 100.0 / 65535
# Minimum fraction of nonzero pixels to be considered foreground image.
_FOREGROUND_AREA_THRESHOLD = 0.0001
# Somewhat arbitrary mean foreground value all images are normalized to.
_FOREGROUND_MEAN = 200.0 / 65535

_SUPPORTED_EXTENSIONS = ['.tif', '.tiff', '.png']


class Dataset(object):
    """Holds the image data before training examples are created.

  The actual images are only read when samples are retrieved.

  Attributes:
    labels: Float32 numpy array [num_images x num_classes].
    image_paths: List of image paths (strings).
    num_examples: Integer, number of examples in dataset.
    image_background_value: Float, background value of images in dataset.
    image_brightness_scale: Float, multiplicative exposure factor.
    image_width: Integer, width of image size to be cropped.
    image_height: Integer, height of image size to be cropped.
  """

    def __init__(self,
                 labels,
                 image_paths,
                 image_width,
                 image_height,
                 image_background_value=0.0,
                 image_brightness_scale=1.0):

        assert len(labels.shape) == 2
        assert len(image_paths) == labels.shape[0]
        assert image_background_value < 1.0
        assert 0 < image_brightness_scale
        self.labels = labels
        self.image_paths = image_paths
        self.num_examples = len(image_paths)
        self.image_background_value = image_background_value
        self.image_brightness_scale = image_brightness_scale
        self.image_width = image_width
        self.image_height = image_height
        assert self.num_examples > 0
        self.subsampled = False
        logging.info('Created dataset with background=%g, brightness_scale=%g.',
                     self.image_background_value, self.image_brightness_scale)

    def randomize(self):
        """Randomize the ordering of images and labels."""
        ordering = numpy.random.permutation(range(self.num_examples))
        self.labels = self.labels[ordering, :]
        self.image_paths = list(numpy.array(self.image_paths)[ordering])

    def subsample_for_shard(self, shard_num, num_shards):
        """Subsample the data based on the shard."""
        if self.subsampled:
            logging.fatal('Dataset has already been subsampled.')
        if num_shards > self.num_examples:
            logging.fatal('num_shards exceeded num_examples')
        self.labels = self.labels[shard_num::num_shards, :]
        self.image_paths = self.image_paths[shard_num::num_shards]
        self.num_examples = len(self.image_paths)
        self.subsampled = True

    def get_sample(self, index, normalize):
        """Get a single sample from the dataset.

    Args:
      index: Integer, index within dataset for the sample.
      normalize: Boolean, whether to brightness normalize the image.

    Returns:
      Tuple of image, a 2D numpy float array, label, a 1D numpy array, and
      image_path, a string path to the image.

    Raises:
      ValueError: If the image pixel values are invalid.
    """
        assert index < self.num_examples
        image_path = self.image_paths[index]
        label = self.labels[index, :]

        # Read image from disk.
        image = get_preprocessed_image(image_path, self.image_background_value,
                                       self.image_brightness_scale,
                                       self.image_width, self.image_height,
                                       normalize)

        assert len(image.shape) == 2
        assert image.dtype == numpy.float32
        assert image.shape[0] == self.image_height
        assert image.shape[1] == self.image_width

        # Check that image pixel values are valid.
        if numpy.any(numpy.isnan(image)):
            raise ValueError('NaNs found in image from %s' % image_path)
        if numpy.min(image) < 0.0 or numpy.max(image) > 1.0:
            raise ValueError('Image values exceed range [0,1.0]: [%g,%g]' %
                             (numpy.min(image), numpy.max(image)))

        return image, label, image_path


def dataset_to_examples_in_tfrecord(list_of_image_globs,
                                    output_directory,
                                    output_tfrecord_filename,
                                    num_classes,
                                    image_width,
                                    image_height,
                                    max_images=100000,
                                    randomize=True,
                                    image_background_value=0.0,
                                    image_brightness_scale=1.0,
                                    shard_num=None,
                                    num_shards=None,
                                    normalize=True,
                                    use_unlabeled_data=False):
    """Reads dataset and saves as TFExamples in a TFRecord.

  Args:
    list_of_image_globs: List of strings, each a glob. If use_unlabeled_data is
      False, the number of globs must equal num_classes (the images for the ith
      glob  will take the true label for class i -- this is used for training
      and evaluation).
    output_directory: String, path to output direcotry.
    output_tfrecord_filename: String, name for output TFRecord.
    num_classes: Integer, number of classes of defocus.
    image_width: Integer, width of image size to be cropped.
    image_height: Integer, height of image size to be cropped.
    max_images: Integer, max number of images to read per class.
    randomize: Boolean, whether to randomly permute the data ordering.
    image_background_value: Float, background value of images in dataset.
    image_brightness_scale: Float, multiplicative exposure factor.
    shard_num: Integer, if sharding, borg task number.
    num_shards: Integer, if sharding, total number of borg tasks.
    normalize: Boolean, whether to brightness normalize the image.
    use_unlabeled_data: Boolean, whether there does not exist true labels.

  Returns:
    Number of converted example images.

  Raises:
    ValueError: If the input image directories are invalid.
  """
    # Get the image paths and labels. Patches will be extracted in data_provider.
    if not use_unlabeled_data:
        if len(list_of_image_globs) != num_classes:
            raise ValueError('%d globs specified, but for labeled data, must be %d' %
                             (len(list_of_image_globs), num_classes))
        dataset = read_labeled_dataset(list_of_image_globs, max_images, num_classes,
                                       image_width, image_height,
                                       image_background_value,
                                       image_brightness_scale)
    else:
        dataset = read_unlabeled_dataset(list_of_image_globs, max_images,
                                         num_classes, image_width, image_height,
                                         image_background_value,
                                         image_brightness_scale)

    if dataset.num_examples == 0:
        raise ValueError('No images found from globs.')
    # Optionally subsample if sharding.
    if shard_num is not None and num_shards is not None and num_shards > 1:
        dataset.subsample_for_shard(shard_num, num_shards)

    # Convert to Examples and write the result to an TFRecord.
    num_examples = convert_to_examples(dataset, output_directory,
                                       output_tfrecord_filename, randomize,
                                       normalize)
    return num_examples


def convert_to_examples(dataset,
                        output_directory,
                        output_tfrecord_filename,
                        randomize=True,
                        normalize=True):
    """Save images and labels into TF Example protos in TFRecord.

  The number of examples is also saved as a .num_examples file.

  Args:
    dataset: Dataset object to convert to examples.
    output_directory: String, path to output directory.
    output_tfrecord_filename: String, name for output TFRecord.
    randomize: Boolean, whether to randomly permute the data ordering.
      normalize: Boolean, whether to brightness normalize the image.

  Returns:
    Number of converted example images.

  Raises:
    ValueError: If dataset contains no examples.
  """
    if dataset.num_examples == 0:
        raise ValueError('No examples found')

    if randomize:
        dataset.randomize()

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, output_tfrecord_filename)

    # Write the actual TFRecord.
    with tensorflow.python_io.TFRecordWriter(output_path) as writer:
        for index in range(dataset.num_examples):
            image, label, image_path = dataset.get_sample(index, normalize)
            example = generate_tf_example(image, label, image_path)
            writer.write(example.SerializeToString())
            if index % 100 == 0:
                logging.info('Saved to TFRecord %g of %g', index, dataset.num_examples)
        logging.info('Wrote %s examples to a TFRecord, with image shape %gx%g.',
                     dataset.num_examples, image.shape[0], image.shape[1])

    # Write the number of examples as a separate file.
    with open(
            quality.data_provider.get_filename_num_records(
                os.path.join(output_directory, output_tfrecord_filename)), 'w') as f:
        f.write(str(dataset.num_examples))

    return dataset.num_examples


def get_preprocessed_image(path,
                           image_background_value,
                           image_brightness_scale,
                           image_width,
                           image_height,
                           normalize=True):
    """Read the a tif or png image, background subtract, crop and normalize.

  Args:
    path: Path to 16-bit tif or png image to be read.
    image_background_value: Float value, between 0.0 and 1.0, indicating
      background value to be subtracted from all pixels. This value should be
      empirically determined for each camera, and should represent the mean
      pixel value with zero incident photons. Note that many background pixel
      values will be clipped to zero.
    image_brightness_scale: Float value, greater than one, indicating
      value to scale all images by (prior to normalization). This is primarily
      for evaluating the performance of the automatic normalization.
    image_width: Integer, width of image size to be cropped.
    image_height: Integer, height of image size to be cropped.
    normalize: Boolean, whether to normalize the image based on the mean
      foreground pixel values. Note that the noise amplitude will be affected by
      this operation, but until the model is trained on data spanning the
      entire 16-bits of dynamic range, this is the best approach toward
      handling the large dynamic range.

  Returns:
    The preprocessed image as a 2D float numpy array.

  Raises:
    ValueError: if image is too small.
  """
    image = read_16_bit_greyscale(path)

    if image.shape[0] < image_height or image.shape[1] < image_width:
        logging.info('Image path %s', path)
        logging.info('Image shape %s', str(image.shape))
        logging.info('image_height, image_width %d,%d', image_height, image_width)
        raise ValueError('Image is too small')

    # Background subtraction
    image_without_background = numpy.clip((
        (image - image_background_value) * image_brightness_scale), 0.0, 1.0)

    cropped_image = image_without_background[0:image_height, 0:image_width]

    # Normalize by the mean of the foreground_pixels.
    if normalize:
        logging.info('Normalizing image brightness')
        preprocessed_image = normalize_image(cropped_image)
    else:
        logging.info('Skipping image brightness normalization')
        preprocessed_image = cropped_image

    return preprocessed_image


def normalize_image(image):
    foreground_mask = image > _FOREGROUND_THRESHOLD
    image_has_foreground = (numpy.sum(foreground_mask) > _FOREGROUND_AREA_THRESHOLD *
                            foreground_mask.shape[0] * foreground_mask.shape[1])
    if image_has_foreground:
        foreground_mean = numpy.sum(image[foreground_mask]) / numpy.sum(foreground_mask)
        return numpy.clip(image / foreground_mean * _FOREGROUND_MEAN, 0.0, 1.0)
    return image


def generate_tf_example(image, label, image_path):
    """Generates a single TF example from an image and label.

  Args:
    image: Float32 numpy array of shape [height x width] containing greyscale
      image.
    label: Float32 numpy array of length [num_classes], a one-hot encoding of
      the class.
    image_path: String, the original path to the image.
  Returns:
    TensorFlow Example.
  """
    example = tensorflow.train.Example()
    features = example.features

    image_expanded = numpy.expand_dims(image, axis=2)
    features.feature[quality.data_provider.FEATURE_IMAGE].float_list.value.extend(
        (image_expanded.flatten().tolist()))

    features.feature[quality.data_provider.FEATURE_IMAGE_CLASS].float_list.value.extend(
        (label.flatten().tolist()))

    features.feature[quality.data_provider.FEATURE_IMAGE_PATH].bytes_list.value.append(
        str.encode(image_path)
    )

    return example


def read_16_bit_greyscale(path):
    """Reads a 16-bit png or tif into a numpy array.

  Args:
    path: String indicating path to .png or .tif file to read.
  Returns:
    A float32 numpy array of the greyscale image, where [0, 65535] is mapped to
    [0, 1].
  """

    file_extension = os.path.splitext(path)[1]

    assert (file_extension in _SUPPORTED_EXTENSIONS), 'path is %s' % path

    greyscale_map = skimage.io.imread(path)

    # Normalize to float in range [0, 1]
    assert numpy.max(greyscale_map) <= 65535
    greyscale_map_normalized = greyscale_map.astype(numpy.float32) / 65535
    return greyscale_map_normalized


def get_image_paths(input_directory, max_images):
    """Gets PNG and TIF image paths within a given directory.

  Args:
    input_directory: String name of input directory, without trailing '/'.
    max_images: Integer, max number of images paths to return.

  Returns:
    List of strings of image paths.
  """
    # os.walk might require path to directory without trailing '/'.
    assert input_directory
    assert input_directory[-1] != '/'
    paths = []
    logging.info('Searching %s for PNG and TIF files.', input_directory)

    for directory, _, files in os.walk(input_directory):
        for f in files:
            path = os.path.join(directory, f)
            if os.path.splitext(path)[1] in _SUPPORTED_EXTENSIONS:
                paths.append(path)
    if not paths:
        logging.info('No images found in directory.')
        return []
    num_images = min(len(paths), max_images)
    if num_images < len(paths):
        logging.info(
            'Using only max_images=%d images of %d images found in directory.',
            max_images, len(paths))
    else:
        logging.info('Found %g images.', len(paths))
    return paths[0:num_images]


def image_size_from_glob(glob, patch_width):
    """Infer image size from glob specifying images.

  Args:
    glob: String, glob specifying at least one image.
    patch_width: Integer, the width in pixels of the model patch size.

  Returns:
    A tuple of height, width, both integers.

  Raises:
    ValueError: If the input glob returns no images.
  """
    image_paths = get_images_from_glob(glob, max_images=1)
    if not image_paths:
        raise ValueError('No input images found in the first glob: %s.' % glob)
    image = read_16_bit_greyscale(image_paths[0])
    image_width = int(patch_width * numpy.floor(image.shape[1] / patch_width))
    image_height = int(patch_width * numpy.floor(image.shape[0] / patch_width))

    image_size = collections.namedtuple('image_size', ['height', 'width'])
    return image_size(image_height, image_width)


def get_images_from_glob(pathnames, max_images):
    """Gets PNG and TIF image paths specified by the glob.

  Args:
    pathnames: String, glob for input images.
    max_images: Integer, max number of images paths to return. Useful for
      restricting the dataset for testing.

  Returns:
    List of string of image paths.
  """
    logging.info('Finding files for glob %s', pathnames)
    paths = glob.glob(pathnames)

    # Filter out paths that are not PNG or TIF.
    filtered_paths = []
    for path in paths:
        if os.path.splitext(path)[1] in _SUPPORTED_EXTENSIONS:
            filtered_paths.append(path)
        else:
            # Ignore all other file types.
            logging.info('Excluding path %s', path)

    num_images = min(len(filtered_paths), max_images)
    if num_images < len(filtered_paths):
        logging.info(
            'Using only max_images=%d images of %d images found in directory.',
            max_images, len(filtered_paths))
    else:
        logging.info('Found %g images.', len(filtered_paths))
    return filtered_paths[0:num_images]


def read_labeled_dataset(list_of_globs,
                         max_images,
                         num_classes,
                         image_width,
                         image_height,
                         image_background_value=0.0,
                         image_brightness_scale=1.0):
    """Gets image paths from disk and create one-hot-encoded labels.

  Use for training and evaluation, where true class labels are known.

  Args:
    list_of_globs: List of strings with length equal to num_classes, each a
      glob. The images for the ith glob will take the true label for
      class i.
    max_images: Integer, max number of images to read per class.
    num_classes: Integer, number of classes of defocus.
    image_width: Integer, width of image size to be cropped.
    image_height: Integer, height of image size to be cropped.
    image_background_value: Float, background value of images in dataset.
    image_brightness_scale: Float, multiplicative exposure factor.

  Returns:
    Dataset object.
  """
    class_labels = range(num_classes)
    labels = numpy.zeros((0, num_classes), dtype=numpy.float32)
    image_paths = []

    for glob, class_label in zip(list_of_globs, class_labels):
        image_paths_i = get_images_from_glob(glob, max_images)

        if image_paths_i is None:
            logging.fatal('No images found for %s', glob)
            continue

        image_paths.extend(image_paths_i)

        labels_class_i = numpy.zeros(
            (len(image_paths_i), num_classes), dtype=numpy.float32)
        labels_class_i[:, class_label] = 1.0
        logging.info('Assigning class label %d for images from %s', class_label,
                     glob)

        labels = numpy.concatenate((labels, labels_class_i))
    if labels.shape[0] == 0:
        logging.fatal('No images found in %s', str(list_of_globs))

    return Dataset(labels, image_paths, image_width, image_height,
                   image_background_value, image_brightness_scale)


def read_unlabeled_dataset(list_of_globs,
                           max_images,
                           num_classes,
                           image_width,
                           image_height,
                           image_background_value=0.0,
                           image_brightness_scale=1.0):
    """Gets image paths from disk for unlabeled data.

  Use for inference (not for training or evaluation). The one-hot encoded label
  vector will be all zeros.

  Args:
    list_of_globs: List of strings, globs for images (unlabeled).
    max_images: Integer, max number of images to read per class.
    num_classes: Integer, number of classes of defocus.
    image_width: Integer, width of image size to be cropped.
    image_height: Integer, height of image size to be cropped.
    image_background_value: Float, background value of images in dataset.
    image_brightness_scale: Float, multiplicative exposure factor.


  Returns:
    Dataset object.
  """

    image_paths = []
    for glob in list_of_globs:
        image_paths_i = get_images_from_glob(glob, max_images)
        if image_paths_i is None:
            logging.fatal('No images found for %s', glob)
            continue
        image_paths.extend(image_paths_i)

    if image_paths is None:
        logging.fatal('No images found in %s', str(list_of_globs))
    logging.info('Using unlabeled image dataset of %d examples from %s',
                 len(image_paths), list_of_globs)

    labels = numpy.zeros((len(image_paths), num_classes), dtype=numpy.float32)

    return Dataset(labels, image_paths, image_width, image_height,
                   image_background_value, image_brightness_scale)
