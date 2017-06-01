# Microscope Image Focus Quality Classifier

This repo contains code for using a pre-trained TensorFlow model to classify the
quality (e.g. running inference) of image focus in microscope images.

Code for training a new model from a dataset of in-focus only images is included
as well.

This is not an official Google product.

## Getting started

Clone the `main` branch of this repository
```sh
git clone -b main <repository_url>
```

Install the package:

```sh
cd All-Projects
pip install --editable .
```

Add path to local repository (e.g. `/Users/user/my_repo/All-projects`)
to `PYTHONPATH` environment variable:
```sh
export PYTHONPATH="${PYTHONPATH}:/Users/user/my_repo/All-projects"
```

Run all tests to make sure everything works. Install any missing
packages (e.g. `sudo pip install pytest` or `sudo pip install nose`).
```sh
pytest --disable-pytest-warnings
```

You should now be able to run:
```sh
quality --help
```

or directly access the
module functions in a jupyter notebook or from your own python module:
```sh
from quality import degrade
degrade.degrade(...)
```

## Running inference

### Requirements for running inference
* A pre-trained TensorFlow model `.ckpt` file, available in
  `quality/data/model/` directory. Model `model.ckpt-1000042` has been
  trained for 1,000,042 steps and is the one for which results in the
  manuscript were computed with.  
* TensorFlow 1.0.0 or higher, numpy, scipy, pypng, PIL, skimage, matplotlib
* Input grayscale 16-bit images, `.png` of `.tif` format, all with the same
width and height.

### How to

Check that all images are of the same dimension:
```sh
 quality validate tests/data/images_for_glob_test/*.tif --width 100 --height 100
```

Run inference on each image independently.

```sh
  quality predict \
  --checkpoint tests/data/checkpoints/model.ckpt-10 \
  --output tests/output/ \
  tests/data/BBBC006*10.png
```
Note: this model checkpoint in the example has only been trained for
10 steps and will probably make random predictions.

Summarize the prediction results across the entire dataset.
TODO(samuely): This yields `Unknown file type` error, at least in
python 2.7.
```sh
quality summarize tests/output/miq_result_images/
```

## Training a new model

### Requirements
* TensorFlow 1.0.0 or higher, and several other python modules.
* A dataset of high quality, in-focus images (at least 400+), as grayscale 16-bit
images, `.png` of `.tif` format, all with the same width and height.

### How to

1. Generate additional labeled training examples of defocused images using `degrade.py`.
1. Launch `quality fit` to train a model.
1. Launch `quality evaluate` with a held-out test dataset.
1. Use TensorBoard to view training and eval progress (see `evaluation.py`).
1. When satisfied with model accuracy, save the `model.ckpt` files for later use.


Example fit:
```sh
quality fit \
	--output tests/train_output \
	tests/data/training/0/*.tif \
	tests/data/training/1/*.tif \
	tests/data/training/2/*.tif \
	tests/data/training/3/*.tif \
	tests/data/training/4/*.tif \
	tests/data/training/5/*.tif \
	tests/data/training/6/*.tif \
	tests/data/training/7/*.tif \
	tests/data/training/8/*.tif \
	tests/data/training/9/*.tif \
	tests/data/training/10/*.tif
```
Example evaluation:
```sh
quality evaluate \
	--checkpoint tests/data/checkpoints/model.ckpt-10 \
	--output tests/data/output \
	tests/data/training/0/*.tif \
	tests/data/training/1/*.tif \
	tests/data/training/2/*.tif \
	tests/data/training/3/*.tif \
	tests/data/training/4/*.tif \
	tests/data/training/5/*.tif \
	tests/data/training/6/*.tif \
	tests/data/training/7/*.tif \
	tests/data/training/8/*.tif \
	tests/data/training/9/*.tif \
	tests/data/training/10/*.tif
```


