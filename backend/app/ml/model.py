"""DenseNet121 model construction, identical to the reference notebook.

Notebook configuration reproduced here:
    conv_base = DenseNet121(weights='imagenet', include_top=False,
                            input_shape=(256, 256, 3), pooling='avg')
    conv_base.trainable = False

    model = Sequential()
    model.add(conv_base)
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.35))
    model.add(BatchNormalization())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

Only difference: ``Dense(10, ...)`` in the notebook is ``Dense(num_classes, ...)``
here, because ``num_classes`` is read from the dataset instead of hard-coded.
The notebook's hard-coded 10 happens to match this project's dataset, but the
code no longer relies on that coincidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tensorflow import keras

IMAGE_SIZE = 256
LEARNING_RATE = 1e-4
DENSE_1_UNITS = 256
DENSE_2_UNITS = 120
DROPOUT = 0.35


def build_model(num_classes: int, image_size: int = IMAGE_SIZE) -> "keras.Model":
    """Build and compile the frozen-DenseNet121 classifier.

    ``num_classes`` must come from the training dataset, never be guessed.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")

    from tensorflow import keras
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    conv_base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    conv_base.trainable = False

    model = Sequential(name="leafora_densenet121")
    model.add(conv_base)
    model.add(BatchNormalization())
    model.add(Dense(DENSE_1_UNITS, activation="relu"))
    model.add(Dropout(DROPOUT))
    model.add(BatchNormalization())
    model.add(Dense(DENSE_2_UNITS, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
