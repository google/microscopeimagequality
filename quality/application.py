import os

import click
import six
import tensorflow

import quality.data_provider
import quality.dataset_creation
import quality.evaluation
import quality.miq
import quality.validation

_MAX_IMAGES_TO_VALIDATE = 1e6


@click.group()
def command():
    pass


@command.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path(exists=True))
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
            'Mean Loss': tensorflow.contrib.metrics.streaming_mean(loss),
            'Aggregated Accuracy': tensorflow.contrib.metrics.streaming_accuracy(aggregated_prediction, aggregated_label),
        }

        names_to_values, names_to_updates = tensorflow.contrib.slim.metrics.aggregate_metric_map(metrics)

        for name, value in six.iteritems(names_to_values):
            tensorflow.summary.scalar(name, value)

        tensorflow.summary.histogram("eval" + ' images', images)
        tensorflow.summary.histogram("eval" + ' labels', labels)
        tensorflow.summary.histogram("eval" + ' predictions', predictions)
        tensorflow.summary.histogram("eval" + ' probabilities', probabilities)

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
def fit():
    pass


@command.command()
def predict():
    pass


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
