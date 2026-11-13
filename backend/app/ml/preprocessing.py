"""Image preprocessing for inference.

This must mirror training exactly:

    tf.keras.utils.image_dataset_from_directory(image_size=(256, 256), label_mode='categorical')
    train_data = train_data.map(lambda x, y: (x / 255.0, y))

So: decoded image -> RGB -> resize to 256x256 -> float32 -> divide by 255.0
-> add batch dimension -> shape (1, 256, 256, 3) with values in [0.0, 1.0].
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

#: Minimal size we accept before resizing. Anything smaller than this is very
#: likely not a usable leaf photo.
MIN_SIDE_PIXELS = 32


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a usable image."""


def decode_and_prepare(image_bytes: bytes, image_size: int = 256) -> np.ndarray:
    """Turn raw upload bytes into a normalized batch of one image.

    Returns a ``float32`` array of shape ``(1, image_size, image_size, 3)``
    with every pixel value in ``[0.0, 1.0]``.

    Raises :class:`InvalidImageError` for empty, unreadable, undersized or
    corrupt input.
    """
    if not image_bytes:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The uploaded file is not a readable image.") from exc

    # `verify()` leaves the file object unusable, so reopen it.
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            if min(img.size) < MIN_SIDE_PIXELS:
                raise InvalidImageError(
                    f"Image is too small ({img.size[0]}x{img.size[1]}). "
                    f"Minimum side is {MIN_SIDE_PIXELS}px."
                )
            img = img.resize((image_size, image_size), Image.BILINEAR)
            array = np.asarray(img, dtype="float32")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The uploaded image could not be processed.") from exc

    if array.shape != (image_size, image_size, 3):
        raise InvalidImageError("The uploaded image could not be processed.")

    array = array / 255.0
    return np.expand_dims(array, axis=0)
