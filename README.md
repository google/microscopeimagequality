Microscope Image Focus Quality Classifier
============================
This repo contains python code for using a pre-trained TensorFlow model to classify the
quality (e.g. running inference) of image focus in microscope images.

Code for training a new model from a dataset of in-focus only images is included
as well.

This is not an official Google product.

See our paper [PDF](http://rdcu.be/I5cE) for reference:

Yang, S. J., Berndl, M., Ando, D. M., Barch, M., Narayanaswamy, A. ,
Christiansen, E., Hoyer, S., Roat, C., Hung, J., Rueden, C. T.,
Shankar, A., Finkbeiner, S., & and Nelson, P. (2018). **Assessing
microscope image focus quality with deep learning**. *BMC Bioinformatics*,
19(1).

Also see the
[Fiji (ImageJ) Microscope Focus Qualtiy plugin](https://imagej.net/Microscope_Focus_Quality),
which allows use of the same pre-trained model on user-supplied images
in a user-friendly graphical user interface, without the need to write
any code. Fiji also has macro scripting capabilities for running
batches of images. This plugin is being actively maintained. I
recommend testing your images with the Fiji plugin before
investing further effort in this python library.

Finally, please note that this python library was developed on the
older and now (as of 2020) deprecated Python 2.7 and TensorFlow 1.x, and is not
being actively maintained and updated. The `setup.py`
currently restricts to these older versions. If you just want the
pre-trained model for integration with your own inference code, you
may need a `saved_model.pb` file that's currently only distributed
with the Fiji plugin and downloadable
[here](https://downloads.imagej.net/fiji/models/microscope-image-quality-model.zip). Updating
this library to work with Python 3.x may be fairly straight
forward. However, updating it to work on TensorFlow 2.x may require
quite a bit of refactoring (at the very least, it appears the data
provider implementation and interface may need to change).


Getting started
-------------

Clone the `main` branch of this repository

```
git clone -b main https://github.com/google/microscopeimagequality.git
```

Install the package:

```
cd microscopeimagequality
```

**Note**: This requires pip be installed.

**Note**: This library has not been migrated beyond TensorFlow 1.x

**Note**: As of now TensorFlow 1.x requires Python 3.7 or earlier.

**Note**: This library has been tested with Python 3.7.9 (using `pyenv`).

```
python --version
python -m pip install --editable .
```

If using `pyenv`, run `pyenv rehash`.

Download the model:
This downloads the `model.ckpt-1000042` checkpoint (a model trained
for 1000042 steps) specified in `constants.py`.
```
microscopeimagequality download 
```
or alternatively:
```python
import microscopeimagequality.miq
microscopeimagequality.miq.download_model()
```

Add path to local repository (e.g. `/Users/user/my_repo/microscopeimagequality`)
to `PYTHONPATH` environment variable:
```
export PYTHONPATH="${PYTHONPATH}:/Users/user/my_repo/microscopeimagequality"
```

Run all tests to make sure everything works. Install any missing
packages (e.g. `python -m pip install pytest`, then if using `pyenv`,
run `pyenv rehash`).

```
pytest --disable-pytest-warnings
```

You should now be able to run:
```
microscopeimagequality --help
```

or directly access the
module functions in a jupyter notebook or from your own python module:
```
python
from microscopeimagequality import degrade
degrade.degrade(...)
```

Running inference
-------------
### Requirements for running inference
* A pre-trained TensorFlow model `.ckpt` files, downloadable using
  download instructions above.
* TensorFlow 1.0.0 or higher, numpy, scipy, pypng, PIL, skimage, matplotlib
* Input grayscale 16-bit images, `.png` of `.tif` format, all with the same
width and height.

### How to

(Optional) Confirm that all images are of the same dimension:
```sh
 microscopeimagequality validate tests/data/images_for_glob_test/*.tif --width 100 --height 100
```

Run inference on each image independently.

```
  microscopeimagequality predict \
  --output tests/output/ \
  tests/data/BBBC006*10.png
```

Summarize the prediction results across the entire dataset. Output will be in
"summary" sub directory.
```
microscopeimagequality summarize tests/output/miq_result_images/
```

Training a new model
----------------

### Requirements
* TensorFlow 1.0.0 or higher, and several other python modules.
* A dataset of high quality, in-focus images (at least 400+), as grayscale 16-bit
images, `.png` of `.tif` format, all with the same width and height.

### How to

1. Generate additional labeled training examples of defocused images using `degrade.py`.
1. Launch `microscopeimagequality fit` to train a model.
1. Launch `microscopeimagequality evaluate` with a held-out test dataset.
1. Use TensorBoard to view training and eval progress (see `evaluation.py`).
1. When satisfied with model accuracy, save the `model.ckpt` files for later use.


Example fit:
```
microscopeimagequality fit \
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
```
microscopeimagequality evaluate \
	--checkpoint <path_to_model_checkpoint>/model.ckpt-XXXXXXX \
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


