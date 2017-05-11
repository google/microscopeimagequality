import os

import pytest

import quality.validation

directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

pathname = os.path.join(directory, "BBBC006_z_aligned__a01__s1__w1_10.png")


def test_check_duplicate_image_name_runs():
    quality.validation.check_duplicate_image_name(["/a/b.c", "/d/e.f"])


def test_check_duplicate_image_name_same_name():
    with pytest.raises(ValueError):
        quality.validation.check_duplicate_image_name(["/a/b.c", "/a/b.c"])


def test_check_duplicate_image_name_different_path_and_extension():
    with pytest.raises(ValueError):
        quality.validation.check_duplicate_image_name(["/a/b.c", "/d/b.f"])


def test_check_image_dimensions_runs():
    quality.validation.check_image_dimensions([pathname], 10, 10)


def test_check_image_dimensions_image_too_small():
    with pytest.raises(ValueError):
        quality.validation.check_image_dimensions([pathname], 1e4, 1e4)
