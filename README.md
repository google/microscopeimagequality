# Microscope Image Focus Quality Classifier

This repo contains code for using a pre-trained TensorFlow model to classify the
quality (e.g. running inference) of image focus in microscope images.

Code for training a new model from a dataset of in-focus only images is included
as well.

This is not an official Google product.

## Running inference

### Requirements for running inference
* A pre-trained TensorFlow model, available in `model` directory.
* TensorFlow 1.0.0 or higher, numpy, scipy, pypng, PIL, skimage, matplotlib
* Input grayscale 16-bit images, `.png` of `.tif` format, all with the same
width and height.

### How to

Check that all images are of the same dimension:
```sh
  python quality/validate_data.py --image_globs_list "quality/testdata/images_for_glob_test/00_mcf-z-stacks-*,quality/testdata/BBBC006*10.png" --image_width=696 --image_height=520
```

Run inference on each image independently.
This now assumes the model checkpoint has been downloaded to
`model/model.ckpt-1000042`:

```sh
  python quality/run_inference.py --eval_directory=/tmp/miq/ --model_ckpt_file model/model.ckpt-1000042 --image_globs_list "quality/testdata/images_for_glob_test/00_mcf-z-stacks-*,quality/testdata/BBBC006*10.png"
```

Summarize the prediction results across the entire dataset.
```sh
python summarize_inference.py
     --experiment_directory /tmp/miq/miq_result_images/
```

## Training a new model

### Requirements
* TensorFlow 1.0.0 or higher, and several other python modules.
* A dataset of high quality, in-focus images (at least 400+), as grayscale 16-bit
images, `.png` of `.tif` format, all with the same width and height.

### How to

1. Generate additional labeled training examples of defocused images using `degrade.py`.
1. Launch `miq_train.py` to train a model.
1. Launch `miq_eval.py` with a held-out test dataset.
1. Use TensorBoard to view training and eval progress.
1. When satisfied with model accuracy, save the `model.ckpt` files for later use.

## Running tests

1. Add path to `quality` directory to PYTHONPATH.
1. Then run each test. For example,  `python quality/miq_test.py`.