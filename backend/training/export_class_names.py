"""Bootstrap `app/ml/class_names.json` from the dataset, without training.

The class order is a fact of the dataset, not a guess, so it can be written
before a model exists. Training overwrites this file with the same order.

Usage (from `backend/`):

    python -m training.export_class_names
"""

from __future__ import annotations

import sys

from training import config
from training.preprocessing import count_images, discover_class_names, save_class_names


def main() -> int:
    names = discover_class_names(config.TRAIN_DIR)
    save_class_names(names, config.CLASS_NAMES_PATH)
    print(f"Wrote {len(names)} classes to {config.CLASS_NAMES_PATH}")
    for index, name in enumerate(names):
        print(f"  [{index}] {name}")
    print(f"Train images: {count_images(config.TRAIN_DIR)}")
    print(f"Val images:   {count_images(config.VAL_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
