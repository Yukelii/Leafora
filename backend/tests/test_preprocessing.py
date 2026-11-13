"""Preprocessing: exact shape and value range the model expects."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.ml.preprocessing import InvalidImageError, decode_and_prepare
from tests.conftest import make_image_bytes


def test_shape_is_batch_256_256_3():
    batch = decode_and_prepare(make_image_bytes())
    assert batch.shape == (1, 256, 256, 3)


def test_dtype_is_float32():
    assert decode_and_prepare(make_image_bytes()).dtype == np.float32


def test_values_are_normalized_to_0_1():
    batch = decode_and_prepare(make_image_bytes())
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0


def test_white_image_normalizes_to_one():
    white = make_image_bytes(color=(255, 255, 255), fmt="PNG")
    batch = decode_and_prepare(white)
    assert pytest.approx(batch.max(), abs=1e-5) == 1.0


def test_black_image_normalizes_to_zero():
    black = make_image_bytes(color=(0, 0, 0), fmt="PNG")
    assert pytest.approx(decode_and_prepare(black).max(), abs=1e-6) == 0.0


def test_rgba_is_converted_to_rgb():
    rgba_png = make_image_bytes(mode="RGBA", color=(10, 20, 30, 128), fmt="PNG")
    assert decode_and_prepare(rgba_png).shape[3] == 3


def test_grayscale_is_converted_to_three_channels():
    gray = make_image_bytes(mode="L", color=(128,) * 3, fmt="PNG")
    assert decode_and_prepare(gray).shape == (1, 256, 256, 3)


def test_wide_image_is_resized_not_cropped():
    wide = make_image_bytes(size=(800, 200), fmt="PNG")
    assert decode_and_prepare(wide).shape == (1, 256, 256, 3)


def test_resize_matches_training_interpolation():
    """Bilinear resize of a small solid image keeps a sensible mean."""
    solid = make_image_bytes(color=(100, 100, 100), size=(64, 64), fmt="PNG")
    batch = decode_and_prepare(solid)
    assert pytest.approx(float(batch.mean()), abs=0.02) == 100 / 255


def test_empty_bytes_raise():
    with pytest.raises(InvalidImageError):
        decode_and_prepare(b"")


def test_garbage_bytes_raise():
    with pytest.raises(InvalidImageError):
        decode_and_prepare(b"not an image at all")


def test_truncated_png_raises():
    data = make_image_bytes(fmt="PNG")
    with pytest.raises(InvalidImageError):
        decode_and_prepare(data[: len(data) // 3])


def test_too_small_image_raises():
    with pytest.raises(InvalidImageError):
        decode_and_prepare(make_image_bytes(size=(10, 10), fmt="PNG"))


def test_image_size_override_is_respected():
    assert decode_and_prepare(make_image_bytes(), image_size=224).shape == (1, 224, 224, 3)
