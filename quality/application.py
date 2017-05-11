import click

import quality.dataset_creation
import quality.validation

_MAX_IMAGES_TO_VALIDATE = 1e6


@click.group()
def command():
    pass


@command.command()
def fit():
    pass


@command.command()
def predict():
    pass


# $ quality validate tests/data/images_for_glob_test/*.tif --width 100 --height 100
@command.command()
@click.argument(
    "images",
    nargs=-1,
    type=click.Path(exists=True)
)
@click.option(
    "--width",
    help="width to crop to. Must be multiple of model_patch_width.",
    nargs=1,
    type=int
)
@click.option(
    "--height",
    help="height to crop to. Must be multiple of model_patch_width.",
    nargs=1,
    type=int
)
@click.option(
    "--patch-width",
    default=84,
    help="The image patch width, in pixels, for model input.",
    nargs=1,
    type=int
)
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
