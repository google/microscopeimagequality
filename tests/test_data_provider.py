import os
import tempfile

import numpy
import numpy.testing
import skimage.io
import tensorflow
import tensorflow.contrib.slim

import microscopeimagequality.data_provider

TFRECORD_NUM_ENTRIES = 33

TFRECORD_NUM_CLASSES = 3

TFRECORD_LABEL_ORDERING = [1, 1, 1, 1, 1, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 1, 0, 2, 0, 1, 2, 0, 2, 2, 0, 1, 0, 1, 1, 2, 0, 0, 1]

input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

test_dir = tempfile.mkdtemp()

batch_size = TFRECORD_NUM_ENTRIES

# For a patch size of 28, we have 324 patches per image in this tfrecord.
patches_per_image = 324

num_classes = TFRECORD_NUM_CLASSES

tfrecord_file_pattern = os.path.join(input_directory, "data_%s.tfrecord")

image_width = 520

image_height = 520


def test_get_filename_num_records():
    tf_record_path = "/folder/filename.tfrecord"
    path = microscopeimagequality.data_provider.get_filename_num_records(tf_record_path)
    expected_path = "/folder/filename.num_records"
    assert expected_path == path


def test_get_num_records():
    tf_record_path = os.path.join(input_directory, "data_train.tfrecord")
    num_records = microscopeimagequality.data_provider.get_num_records(tf_record_path)
    expected_num_records = TFRECORD_NUM_ENTRIES
    assert expected_num_records == num_records


def save16_bit_png(filename, im):
    path = os.path.join(test_dir, filename)
    skimage.io.imsave(path, im, "pil")


def get_tf_session(graph):
    sv = tensorflow.train.Supervisor(logdir=os.path.join(test_dir, "tmp_logs/"))
    sess = sv.PrepareSession("")
    sv.StartQueueRunners(sess, graph.get_collection(tensorflow.GraphKeys.QUEUE_RUNNERS))
    return sess


def get_data_from_tfrecord():
    """Helper function that gets image, label tensors from tfrecord."""
    split_name = "train"
    num_records = microscopeimagequality.data_provider.get_num_records(tfrecord_file_pattern % split_name)
    assert TFRECORD_NUM_ENTRIES == num_records
    dataset = microscopeimagequality.data_provider.get_split(split_name, tfrecord_file_pattern, num_classes=num_classes, image_width=image_width, image_height=image_height)
    provider = tensorflow.contrib.slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=2 * batch_size, common_queue_min=batch_size, shuffle=False)
    [image, label, image_path] = provider.get([microscopeimagequality.data_provider.FEATURE_IMAGE, microscopeimagequality.data_provider.FEATURE_IMAGE_CLASS, microscopeimagequality.data_provider.FEATURE_IMAGE_PATH ])
    return image, label, image_path


def test_get_split():
    g = tensorflow.Graph()
    with g.as_default():
        image, label, image_path = get_data_from_tfrecord()

        sess = get_tf_session(g)

        # Check that the tensor shapes are as expected.
        np_image, np_label, np_image_path = sess.run([image, label, image_path])
        numpy.testing.assert_array_equal(list(np_image.shape), [microscopeimagequality.data_provider.IMAGE_WIDTH, microscopeimagequality.data_provider.IMAGE_WIDTH, 1])
        numpy.testing.assert_array_equal(list(np_label.shape), [num_classes])
        numpy.testing.assert_array_equal([1], list(np_image_path.shape))
        assert 22 == len(np_image_path[0])

        # Write the image for viewing.
        im = (numpy.squeeze(np_image) * 65535).astype(numpy.uint16)
        image_class = numpy.argmax(np_label)
        save16_bit_png("single_im_from_tfrecord_%g.png" % image_class, im)


def test_batching():
    g = tensorflow.Graph()
    with g.as_default():
        image, label, image_path = get_data_from_tfrecord()

        # Expand since get_batches() requires a larger dimension tensor.
        expanded_label = tensorflow.expand_dims(label, dim=0)
        expanded_image = tensorflow.expand_dims(image, dim=0)
        expanded_image_path = tensorflow.expand_dims(image_path, dim=0)

        images, labels, image_paths = microscopeimagequality.data_provider.get_batches(expanded_image, expanded_label, expanded_image_path, batch_size=batch_size, num_threads=1)

        sess = get_tf_session(g)

        [np_images, np_labels, np_image_paths] = sess.run([images, labels, image_paths])

        # Check the number of images and shape is as expected.
        numpy.testing.assert_array_equal(list(np_images.shape), [batch_size, microscopeimagequality.data_provider.IMAGE_WIDTH, microscopeimagequality.data_provider.IMAGE_WIDTH, 1 ])
        numpy.testing.assert_array_equal([batch_size, 1], list(np_image_paths.shape))
        assert 1 == len(np_image_paths[0])
        assert b"image_000" == os.path.basename(np_image_paths[0][0])

        # Check the ordering of labels in a single batch (which is preserved
        # since we used num_threads=1).
        image_classes = numpy.argmax(np_labels, axis=1).tolist()

        numpy.testing.assert_array_equal(image_classes, TFRECORD_LABEL_ORDERING)


def test_get_image_patch_tensor():
    patch_width = 280
    g = tensorflow.Graph()
    with g.as_default():
        image, label, image_path = get_data_from_tfrecord()
        patch, label, image_path = microscopeimagequality.data_provider.get_image_patch_tensor(image, label, image_path, patch_width=patch_width)

        sess = get_tf_session(g)

        [np_patch, np_label, np_image_path] = sess.run([patch, label, image_path])

        # Check that the tensor shapes are as expected.
        numpy.testing.assert_array_equal(list(np_patch.shape), [1, patch_width, patch_width, 1])
        numpy.testing.assert_array_equal(list(np_label.shape), [1, num_classes])
        numpy.testing.assert_array_equal([1, 1], list(np_image_path.shape))

        # Write the image for viewing.
        im = (numpy.squeeze(np_patch) * 65535).astype(numpy.uint16)
        save16_bit_png("single_random_patch_from_tfrecord.png", im)


def test_apply_random_brightness_adjust():
    g = tensorflow.Graph()
    with g.as_default():
        image, _, _ = get_data_from_tfrecord()
        factor = 2.0
        patch = microscopeimagequality.data_provider.apply_random_brightness_adjust(image, factor, factor)

        sess = get_tf_session(g)

        [np_patch, np_image] = sess.run([patch, image])

        numpy.testing.assert_array_equal(list(np_patch.shape), list(np_image.shape))
        numpy.testing.assert_array_equal(np_image * factor, np_patch)


def test_get_image_tiles_tensor():
    patch_width = 100
    g = tensorflow.Graph()
    with g.as_default():
        image, label, image_path = get_data_from_tfrecord()
        tiles, labels, image_paths = microscopeimagequality.data_provider.get_image_tiles_tensor(image, label, image_path, patch_width=patch_width)

        sess = get_tf_session(g)

        [np_tiles, np_labels, np_image_paths] = sess.run([tiles, labels, image_paths])

        # Check that the tensor shapes are as expected.
        num_tiles_expected = 25
        numpy.testing.assert_array_equal(list(np_tiles.shape), [num_tiles_expected, patch_width, patch_width, 1])
        numpy.testing.assert_array_equal(list(np_labels.shape), [num_tiles_expected, num_classes])
        numpy.testing.assert_array_equal([num_tiles_expected, 1], list(np_image_paths.shape))


def test_get_image_tiles_tensor_non_square():
    patch_width = 100
    g = tensorflow.Graph()
    with g.as_default():
        image = tensorflow.zeros([patch_width * 4, patch_width * 3, 1])
        label = tensorflow.constant([0, 0, 1])
        image_path = tensorflow.constant(["path"])
        tiles, labels, image_paths = microscopeimagequality.data_provider.get_image_tiles_tensor(image, label, image_path, patch_width=patch_width)

        sess = get_tf_session(g)

        [np_tiles, np_labels, np_image_paths] = sess.run([tiles, labels, image_paths])

        # Check that the tensor shapes are as expected.
        num_tiles_expected = 12
        numpy.testing.assert_array_equal(list(np_tiles.shape), [num_tiles_expected, patch_width, patch_width, 1])
        numpy.testing.assert_array_equal(list(np_labels.shape), [num_tiles_expected, num_classes])
        numpy.testing.assert_array_equal([num_tiles_expected, 1], list(np_image_paths.shape))


def test_provide_data_with_random_patches():
    images, one_hot_labels, image_paths, _ = microscopeimagequality.data_provider.provide_data(tfrecord_file_pattern, split_name="train", batch_size=batch_size, num_classes=num_classes, image_width=image_width, image_height=image_height, patch_width=28, randomize=True)

    assert images.get_shape().as_list(), [batch_size, 28, 28 == 1]
    assert one_hot_labels.get_shape().as_list(), [batch_size == num_classes]
    assert [batch_size, 1] == image_paths.get_shape().as_list()


def test_provide_data_image_path():
    g = tensorflow.Graph()
    with g.as_default():
        _, _, image_paths, _ = microscopeimagequality.data_provider.provide_data(tfrecord_file_pattern, split_name="train", batch_size=patches_per_image, num_classes=3, image_width=image_width, image_height=image_height, patch_width=28, randomize=False, num_threads=1)

        sess = get_tf_session(g)

        [np_image_paths] = sess.run([image_paths])

        filename_expected = b"image_000"
        assert 1 == len(np_image_paths[0])
        assert filename_expected == os.path.basename(np_image_paths[0][0])


def test_provide_data_uniform_tiles():
    g = tensorflow.Graph()
    with g.as_default():
        images, one_hot_labels, _, _ = microscopeimagequality.data_provider.provide_data(tfrecord_file_pattern, split_name="train", batch_size=patches_per_image, num_classes=num_classes, image_width=image_width, image_height=image_height, patch_width=28, randomize=False)

        num_tiles_expected = patches_per_image
        assert images.get_shape().as_list(), [num_tiles_expected, 28, 28 == 1]
        assert one_hot_labels.get_shape().as_list(), [num_tiles_expected == num_classes]

        sess = get_tf_session(g)

        [np_images, np_labels] = sess.run([images, one_hot_labels])
        assert np_labels.shape, (num_tiles_expected == num_classes)

        im = (numpy.squeeze(np_images[0, :, :, :]) * 65535).astype(numpy.uint16)
        save16_bit_png("first_tile_single_batch.png", im)


def test_provide_data_with_deterministic_ordering():
    # Use patches larger to speed up test, otherwise it will timeout.
    patch_size_factor = 3

    batch_size = patches_per_image / patch_size_factor ** 2

    g = tensorflow.Graph()

    with g.as_default():
        images, one_hot_labels, image_paths, _ = microscopeimagequality.data_provider.provide_data(tfrecord_file_pattern, split_name="train", batch_size=batch_size, num_classes=num_classes, image_width=image_width, image_height=image_height, patch_width=28 * patch_size_factor, randomize=False, num_threads=1 )

        sess = get_tf_session(g)

        # Here, we are looking at the first label across many batches, rather than
        # the ordering of labels in one batch, as in testBatching(). We check to
        # ensure the ordering is deterministic for num_threads = 1.
        image_classes = []
        num_batches_tested = min(20, TFRECORD_NUM_ENTRIES)
        for i in range(num_batches_tested):
            [np_images, np_labels, np_image_paths] = sess.run([images, one_hot_labels, image_paths])

            assert np_labels.shape, (batch_size == num_classes)

            # All class labels should be identical within this batch.
            image_class = numpy.argmax(np_labels, axis=1)
            assert numpy.all(image_class[0] == image_class)
            assert numpy.all(np_image_paths[0] == np_image_paths)
            image_classes.append(image_class[0])

            im = (numpy.squeeze(np_images[0, :, :, :]) * 65535).astype(numpy.uint16)
            save16_bit_png("first_tile_per_batch_%g.png" % i, im)

        numpy.testing.assert_array_equal(image_classes, TFRECORD_LABEL_ORDERING[0:num_batches_tested])
