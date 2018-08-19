Microscope Image Focus Quality Classifier
============================
This repo contains code for using a pre-trained TensorFlow model to classify the
quality (e.g. running inference) of image focus in microscope images.

Code for training a new model from a dataset of in-focus only images is included
as well.

This is not an official Google product.

See the paper [PDF](http://rdcu.be/I5cE) for reference:

Yang, S. J., Berndl, M., Ando, D. M., Barch, M., Narayanaswamy, A. , Christiansen, E., Hoyer, S., Roat, C., Hung, J., Rueden, C. T., Shankar, A., Finkbeiner, S., & and Nelson, P. (2018). Assessing microscope image focus quality with deep learning. BMC Bioinformatics, 19(1).

Getting started
-------------

Clone the `main` branch of this repository

```
git clone -b main https://github.com/google/microscopeimagequality.git
```

Install the package:

```
cd microscopeimagequality
# Note: To install pip, run "sudo easy_install pip".
# Note: This may need to be run with "sudo pip".
pip install --editable .
```

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
packages (e.g. `sudo pip install pytest` or `sudo pip install nose`).
```
pytest --disable-pytest-warnings
```

You should now be able to run:
```
microscopeimagequality --help
```

or directly access the
module functions in a jupyter notebook or from your own python module:
```python
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


