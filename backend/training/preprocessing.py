"""Dataset loading and class discovery, matching the notebook exactly.

Notebook:
    tf.keras.utils.image_dataset_from_directory(
        path, labels='inferred', label_mode='categorical',
        image_size=(256, 256), batch_size=32)
    train_data = train_data.map(lambda x, y: (x / 255.0, y))

`image_dataset_from_directory` sorts directory names alphabetically, so the
class order is deterministic. We record that order to class_names.json and the
inference service reuses it - a mismatch would silently mislabel predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import tensorflow as tf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def discover_class_names(train_dir: Path) -> list[str]:
    """Return the class names in the order Keras will use (alphabetical)."""
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    names = sorted(
        entry.name
        for entry in train_dir.iterdir()
        if entry.is_dir() and any(f.suffix.lower() in IMAGE_EXTENSIONS for f in entry.iterdir() if f.is_file())
    )
    if not names:
        raise ValueError(f"No class sub-folders with images found in {train_dir}")
    return names


def count_images(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_class_names(class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_names, indent=2) + "\n", encoding="utf-8")


def load_datasets(image_size: int, batch_size: int):
    """Build the normalized train/val datasets from `training/config.py`."""
    import tensorflow as tf

    from training import config

    train_data = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=config.RANDOM_SEED,
    )
    val_data = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    normalize = lambda x, y: (x / 255.0, y)  # noqa: E731 - matches the notebook
    return train_data.map(normalize), val_data.map(normalize)


def load_field_datasets(image_size: int, batch_size: int):
    """Build normalized train/val datasets from the real-photo (field) folder.

    Same preprocessing as the lab data: 256x256, labels inferred, categorical,
    shuffled train, ordered val, pixels divided by 255.0.
    """
    import tensorflow as tf

    from training import config

    train_data = tf.keras.utils.image_dataset_from_directory(
        config.FIELD_TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=config.RANDOM_SEED,
    )
    val_data = tf.keras.utils.image_dataset_from_directory(
        config.FIELD_VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    normalize = lambda x, y: (x / 255.0, y)  # noqa: E731 - matches the notebook
    return train_data.map(normalize), val_data.map(normalize)


def field_class_names() -> list[str]:
    """Class order of the field folders, in the order Keras will use it."""
    from training import config

    return discover_class_names(config.FIELD_TRAIN_DIR)


def take_fraction(dataset, fraction: float, minimum: int = 1):
    """Keep a fraction of a batched dataset, counted in batches.

    `fraction=0.25` keeps a quarter of the batches. Used to hold back most of the
    field photos so the fine-tune sees little data and the test split stays unseen.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    import math

    total = int(dataset.cardinality().numpy())
    keep = max(minimum, math.ceil(total * fraction))
    return dataset.take(keep), keep, total


def verify_class_order(train_dir: Path, class_names: list[str]) -> list[str]:
    """Warn loudly if Keras' ordering and class_names.json disagree."""
    actual = discover_class_names(train_dir)
    if actual != class_names:
        raise ValueError(
            "Class order mismatch between class_names.json and the dataset.\n"
            f"  dataset: {actual}\n  json:    {class_names}"
        )
    return actual
