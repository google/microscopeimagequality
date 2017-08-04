import logging
import os

import click
import numpy
import six
import tensorflow
import urllib

# Use this backend for producing PNGs without interactive display.
import matplotlib
matplotlib.use('Agg')

import quality.constants as constants
import quality.data_provider
import quality.dataset_creation
import quality.evaluation
import quality.prediction
import quality.miq
import quality.summarize
import quality.validation

_MAX_IMAGES_TO_VALIDATE = 1e6


@click.group()
def command():
    pass


@command.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path())
@click.option("--output", type=click.Path())
@click.option("--patch-width", default=84)
def evaluate(images, checkpoint, output, patch_width):
    """

    """
    num_classes = len(images)

    output_tfrecord_file_pattern = 'data_%s.sstable'

    image_size = quality.dataset_creation.image_size_from_glob(images[0], patch_width)

    quality.dataset_creation.dataset_to_examples_in_tfrecord(
        list_of_image_globs=images,
        output_directory=output,
        output_tfrecord_filename=output_tfrecord_file_pattern % 'test',
        num_classes=num_classes,
        image_width=image_size.width,
        image_height=image_size.height,
        image_background_value=0.0,
        normalize=False
    )

    tfexamples_tfrecord_file_pattern = os.path.join(output, output_tfrecord_file_pattern)

    graph = tensorflow.Graph()

    with graph.as_default():
        batch_size = int(image_size.height * image_size.width / patch_width ** 2)

        images, one_hot_labels, _, num_samples = quality.data_provider.provide_data(
            tfrecord_file_pattern=tfexamples_tfrecord_file_pattern,
            split_name='test',
            batch_size=batch_size,
            num_classes=num_classes,
            image_width=image_size.width,
            image_height=image_size.height,
            patch_width=patch_width,
            randomize=False
        )

        logits, labels, probabilities, predictions = quality.evaluation.get_model_and_metrics(
            images=images,
            num_classes=num_classes,
            one_hot_labels=one_hot_labels,
            is_training=False,
            model_id=0
        )

        # Define the loss
        quality.miq.add_loss(logits, one_hot_labels, use_rank_loss=True)

        loss = tensorflow.losses.get_total_loss()

        # Additional aggregate metrics
        aggregated_prediction, aggregated_label = quality.evaluation.get_aggregated_prediction(probabilities, labels, batch_size)

        metrics = {
            'Accuracy': tensorflow.contrib.metrics.streaming_accuracy(predictions, labels),
            'Mean_Loss': tensorflow.contrib.metrics.streaming_mean(loss),
            'Aggregated_Accuracy': tensorflow.contrib.metrics.streaming_accuracy(aggregated_prediction, aggregated_label),
        }

        names_to_values, names_to_updates = tensorflow.contrib.slim.metrics.aggregate_metric_map(metrics)

        for name, value in six.iteritems(names_to_values):
            tensorflow.summary.scalar(name, value)

        tensorflow.summary.histogram("eval_images", images)
        tensorflow.summary.histogram("eval_labels", labels)
        tensorflow.summary.histogram("eval_predictions", predictions)
        tensorflow.summary.histogram("eval_probabilities", probabilities)

        quality.evaluation.annotate_classification_errors(
            images,
            predictions,
            labels,
            probabilities,
            image_height=image_size[0],
            image_width=image_size[1]
        )

        # This ensures that we evaluate over exactly all samples.
        num_batches = num_samples

        tensorflow.contrib.slim.evaluation.evaluation_loop(
            master='',
            checkpoint_dir=checkpoint,
            logdir=output,
            num_evals=num_batches,
            eval_op=names_to_updates.values(),
            eval_interval_secs=60
        )


@command.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--output", nargs=1, type=click.Path())
def fit(images, output):
    if not os.path.exists(output):
        os.makedirs(output)

    num_classes = len(images)

    output_tfrecord_file_pattern = ('worker%g_' % 0) + 'data_%s.tfrecord'

    image_size = quality.dataset_creation.image_size_from_glob(images[0], 84)

    # Read images and convert to TFExamples in an TFRecord.
    quality.dataset_creation.dataset_to_examples_in_tfrecord(
        images,
        output,
        output_tfrecord_file_pattern % 'train',
        num_classes,
        image_width=image_size.width,
        image_height=image_size.height,
        image_background_value=0.0
    )

    tfexamples_tfrecord_file_pattern = os.path.join(output, output_tfrecord_file_pattern)

    graph = tensorflow.Graph()

    # builder = tensorflow.saved_model.builder.SavedModelBuilder("/tmp/quality-fit/")
    #
    # with tf.Session(graph=tf.Graph()) as sess:
    #     ...
    #     builder.add_meta_graph_and_variables(sess,
    #                                          ["foo-tag"],
    #                                          signature_def_map=foo_signatures,
    #                                          assets_collection=foo_assets)

    with graph.as_default():
        # If ps_tasks is zero, the local device is used. When using multiple
        # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
        # across the different devices.
        with tensorflow.device(tensorflow.train.replica_device_setter(0)):
            images, one_hot_labels, _, _ = quality.data_provider.provide_data(
                tfexamples_tfrecord_file_pattern,
                split_name='train',
                batch_size=64,
                num_classes=num_classes,
                image_width=image_size.width,
                image_height=image_size.height,
                patch_width=84
            )

            # Visualize the input
            tensorflow.summary.image('train_input', images)
            # slim.summaries.add_histogram_summaries([images, labels])

            # Define the model:
            logits = quality.miq.miq_model(
                images=images,
                num_classes=num_classes,
                is_training=True,
                model_id=0
            )

            # Specify the loss function:
            quality.miq.add_loss(logits, one_hot_labels, use_rank_loss=True)
            total_loss = tensorflow.losses.get_total_loss()
            tensorflow.summary.scalar('Total_Loss', total_loss)

            # Specify the optimization scheme:
            optimizer = tensorflow.train.AdamOptimizer(0.00003)

            # Set up training.
            train_op = tensorflow.contrib.slim.learning.create_train_op(total_loss, optimizer)

            # Monitor model variables for debugging.
            # slim.summaries.add_histogram_summaries(slim.get_model_variables())

            # Run training.
            tensorflow.contrib.slim.learning.train(
                train_op=train_op,
                logdir=output,
                is_chief=0 == 0,
                number_of_steps=10,
                save_summaries_secs=15,
                save_interval_secs=60
            )

@command.command()
@click.argument("output_path", nargs=1, type=click.Path())
def download(output_path):
    print "Downloading model from %s to %s." % (constants.REMOTE_MODEL_CHECKPOINT_PATH, output_path)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    file_extensions = [".index", ".meta", ".data-00000-of-00001"]
    for extension in file_extensions:
        remote_path = constants.REMOTE_MODEL_CHECKPOINT_PATH + extension
        local_path = os.path.join(output_path, os.path.basename(remote_path))
        urllib.urlretrieve(remote_path, local_path)
                                  
    print "Downloaded %d files to %s." % (len(file_extensions), output_path)

@command.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path())
@click.option("--height", type=int)
@click.option("--output", type=click.Path())
@click.option("--patch-width", default=84)
@click.option("--visualize", is_flag=True)
@click.option("--width", type=int)
def predict(images, checkpoint, output, width, height, patch_width, visualize):
    if output is None:
        logging.fatal('Eval directory required.')

    if checkpoint is None:
        logging.fatal('Model checkpoint file required.')

    if images is None:
        logging.fatal('Must provide image globs list.')

    if not os.path.isdir(output):
        os.makedirs(output)

    use_unlabeled_data = True

    # Input images will be cropped to image_height x image_width.
    image_size = quality.dataset_creation.image_size_from_glob(images[0], patch_width)

    if width is not None and height is not None:
        image_width = int(patch_width * numpy.floor(width / patch_width))

        image_height = int(patch_width * numpy.floor(height / patch_width))

        if image_width > image_size.width or image_height > image_size.height:
            raise ValueError('Specified (image_width, image_height) = (%d, %d) exceeds valid dimensions (%d, %d).' % (image_width, image_height, image_size.width, image_size.height))
    else:
        image_width = image_size.width

        image_height = image_size.height

    # All patches evaluated in a batch correspond to one single input image.
    batch_size = int(image_width * image_height / (patch_width ** 2))

    logging.info('Using batch_size=%d for image_width=%d, image_height=%d, model_patch_width=%d', batch_size, image_width, image_height, patch_width)

    tfexamples_tfrecord = quality.prediction.build_tfrecord_from_pngs(images, use_unlabeled_data, 11, output, 0.0, 1.0, 1, 1, image_width, image_height)

    num_samples = quality.data_provider.get_num_records(tfexamples_tfrecord % quality.prediction._SPLIT_NAME)

    logging.info('TFRecord has %g samples.', num_samples)

    graph = tensorflow.Graph()

    with graph.as_default():
        images, one_hot_labels, image_paths, _ = quality.data_provider.provide_data(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
            num_classes=11,
            num_threads=1,
            patch_width=patch_width,
            randomize=False,
            split_name=quality.prediction._SPLIT_NAME,
            tfrecord_file_pattern=tfexamples_tfrecord
        )

        model_metrics = quality.evaluation.get_model_and_metrics(
            images=images,
            is_training=False,
            model_id=0,
            num_classes=11,
            one_hot_labels=one_hot_labels
        )

        quality.prediction.run_model_inference(
            aggregation_method=quality.evaluation.METHOD_AVERAGE,
            image_height=image_height,
            image_paths=image_paths,
            image_width=image_width,
            images=images,
            labels=model_metrics.labels,
            model_ckpt_file=checkpoint,
            num_samples=num_samples,
            num_shards=1,
            output_directory=os.path.join(output, 'miq_result_images'),
            patch_width=patch_width,
            probabilities=model_metrics.probabilities,
            shard_num=1,
            show_plots=visualize
        )

    # Delete TFRecord to save disk space.
    tfrecord_path = tfexamples_tfrecord % quality.prediction._SPLIT_NAME

    os.remove(tfrecord_path)

    logging.info('Deleted %s', tfrecord_path)


@command.command()
@click.argument("experiments", type=click.Path(exists=True))
def summarize(experiments):
    if experiments is None:
        logging.fatal('Experiment directory required.')

    probabilities, labels, certainties, orig_names, predictions = quality.evaluation.load_inference_results(experiments)

    if not predictions:
        logging.fatal('No inference output found at %s.', experiments)

    quality.summarize.check_image_count_matches(experiments, len(predictions))

    output_path = os.path.join(experiments, 'summary')

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Less useful plots go here.
    output_path_all_plots = os.path.join(output_path, 'additional_plots')

    if not os.path.isdir(output_path_all_plots):
        os.makedirs(output_path_all_plots)

    quality.summarize.save_histograms_scatter_plots_and_csv(probabilities, labels, certainties, orig_names, predictions, output_path, output_path_all_plots)

    quality.summarize.save_summary_montages(probabilities, certainties, orig_names, predictions, experiments, output_path, output_path_all_plots)

    logging.info('Done summarizing results at %s', output_path)


# $ quality validate tests/data/images_for_glob_test/*.tif --width 100 --height 100
@command.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--width", type=int)
@click.option("--height", type=int)
@click.option("--patch-width", default=84)
def validate(images, width, height, patch_width):
    image_paths = []

    for image in images:
        image_paths += quality.dataset_creation.get_images_from_glob(image, _MAX_IMAGES_TO_VALIDATE)

    click.echo('Found {} paths'.format(len(image_paths)))

    if len(image_paths) == 0:
        raise ValueError('No images found.')

    quality.validation.check_duplicate_image_name(image_paths)

    if width is None or height is None:
        height, width = quality.dataset_creation.image_size_from_glob(images, patch_width)

    quality.validation.check_image_dimensions(image_paths, height, width)
