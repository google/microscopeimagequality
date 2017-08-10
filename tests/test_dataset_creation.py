import os
import tempfile

import numpy
import pytest

import microscopeimagequality.dataset_creation

input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

test_dir = tempfile.mkdtemp()

input_image_path = os.path.join(input_directory, "BBBC006_z_aligned__a01__s1__w1_10.png")

input_image_path_tif = os.path.join(input_directory, "00_mcf-z-stacks-03212011_k06_s2_w12667264a-6432-4f7e-bf58-625a1319a1c9.tif")

glob_images = os.path.join(input_directory, "images_for_glob_test/*")

list_of_class_globs = []

num_classes = 3

empty_directory = os.path.join(test_dir, "empty")

image_width = 520

image_height = 520

for _ in range(num_classes):
    list_of_class_globs.append(glob_images)


def test_dataset_randomize_runs():
    dataset = microscopeimagequality.dataset_creation.Dataset(numpy.zeros((2, 2)), ["a", "b"], image_width, image_height)

    dataset.randomize()


def test_datset_subsample_for_shard():
    labels = numpy.array([[0, 1], [2, 3], [4, 5], [6, 7]])

    image_paths = ["path"] * labels.shape[0]

    dataset = microscopeimagequality.dataset_creation.Dataset(labels, image_paths, image_width, image_height)

    dataset.subsample_for_shard(0, 2)

    numpy.testing.assert_array_equal(numpy.array([[0, 1], [4, 5]]), dataset.labels)


def test_dataset_get_sample():
    dataset = microscopeimagequality.dataset_creation.Dataset(numpy.zeros((2, 2)), [input_image_path, input_image_path], image_width, image_height)

    _, _, image_path = dataset.get_sample(0, True)

    assert input_image_path == image_path


def test_dataset_to_examples_in_tfrecord_runs():
    microscopeimagequality.dataset_creation.dataset_to_examples_in_tfrecord(
        list_of_class_globs,
        test_dir,
        output_tfrecord_filename="data_train.tfrecord",
        num_classes=num_classes,
        image_width=image_width,
        image_height=image_height
    )


def test_convert_to_examples_runs():
    labels = numpy.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=numpy.float32)
    image_paths = [input_image_path] * 3
    microscopeimagequality.dataset_creation.convert_to_examples(
        microscopeimagequality.dataset_creation.Dataset(labels, image_paths, image_width, image_height),
        output_directory=test_dir,
        output_tfrecord_filename="data_train.tfrecord"
    )


def test_get_preprocesssed_image_runs():
    image = microscopeimagequality.dataset_creation.get_preprocessed_image(
        input_image_path,
        image_background_value=0.0,
        image_brightness_scale=1.0,
        image_width=image_width,
        image_height=image_height,
        normalize=True
    )

    pytest.approx((520, 520), image.shape)


def test_normalize_image():
    image = microscopeimagequality.dataset_creation.read_16_bit_greyscale(input_image_path)

    image_normalized = microscopeimagequality.dataset_creation.normalize_image(image)

    expected_mean = numpy.mean(image) * 496.283426445 * microscopeimagequality.dataset_creation._FOREGROUND_MEAN

    assert numpy.abs(expected_mean - numpy.mean(image_normalized)) < 1e-6


def test_normalize_image_no_foreground():
    image = numpy.zeros((100, 100), dtype=numpy.float32)

    image_normalized = microscopeimagequality.dataset_creation.normalize_image(image)

    assert 0.0 == numpy.mean(image_normalized)


def test_generate_tf_example_runs():
    image = numpy.ones((100, 100), dtype=numpy.float32)

    label = numpy.array([0.0, 1.0], dtype=numpy.float32)

    image_path = "directory/filename.extension"

    _ = microscopeimagequality.dataset_creation.generate_tf_example(image, label, image_path)


def test_read16_bit_greyscale_png():
    image = microscopeimagequality.dataset_creation.read_16_bit_greyscale(input_image_path)

    assert image.shape, (520 == 696)

    pytest.approx(numpy.max(image), 3252.0 / 65535)

    assert image.dtype == numpy.float32


def test_read16_bit_greyscale_tif():
    image = microscopeimagequality.dataset_creation.read_16_bit_greyscale(input_image_path_tif)

    assert image.shape, (520 == 696)

    pytest.approx(numpy.max(image), 1135.0 / 65535)

    assert image.dtype == numpy.float32


def test_get_image_paths():
    paths = microscopeimagequality.dataset_creation.get_image_paths(os.path.join(input_directory, "images_for_glob_test"), 100)

    for path in paths:
        extension = os.path.splitext(path)[1]

        assert extension == ".png" or extension == ".tif", "path is %s" % path

    assert 24 == len(paths)


def test_image_size_from_glob():
    image_size = microscopeimagequality.dataset_creation.image_size_from_glob(input_image_path, 84)

    assert 504 == image_size.height

    assert 672 == image_size.width


def test_get_images_from_glob():
    paths = microscopeimagequality.dataset_creation.get_images_from_glob(glob_images, 100)

    for path in paths:
        assert os.path.splitext(path)[1] == ".png" or os.path.splitext(path)[1] == ".tif", "path is %s" % path

    assert 24 == len(paths)


def test_read_labeled_dataset_without_patches():
    max_images = 3

    dataset = microscopeimagequality.dataset_creation.read_labeled_dataset(list_of_class_globs, max_images, num_classes, image_width, image_height)

    num_images_expected = (max_images * num_classes)

    assert dataset.labels.shape, (num_images_expected == num_classes)

    assert num_images_expected == len(dataset.image_paths)


def test_read_unlabeled_dataset():
    max_images = 3

    num_classes = 5

    dataset = microscopeimagequality.dataset_creation.read_unlabeled_dataset([glob_images], max_images, num_classes, image_width, image_height)

    num_images_expected = max_images

    assert dataset.labels.shape, (num_images_expected == num_classes)

    assert num_images_expected == len(dataset.image_paths)
