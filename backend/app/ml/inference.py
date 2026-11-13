"""Inference wrapper around the exported Keras model.

Responsibilities:
    * load the model and class mapping once
    * preprocess a single uploaded image
    * run inference
    * map the winning index back to the trained class name
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.ml.preprocessing import decode_and_prepare

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """The model or the class mapping could not be loaded."""


class InferenceError(RuntimeError):
    """The model was loaded but a prediction failed."""


def load_class_names(path: Path) -> list[str]:
    """Read and validate the trained class ordering."""
    if not path.is_file():
        raise ModelNotAvailableError(f"Class names file not found: {path}")
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelNotAvailableError(f"Class names file is unreadable: {path}") from exc

    if isinstance(raw, dict):  # tolerate {"class_names": [...]} or {"0": "..."}
        raw = raw.get("class_names", list(raw.values()))
    if not isinstance(raw, list) or not raw:
        raise ModelNotAvailableError("Class names file must hold a non-empty list.")
    if not all(isinstance(item, str) and item for item in raw):
        raise ModelNotAvailableError("Class names must be non-empty strings.")
    if len(set(raw)) != len(raw):
        raise ModelNotAvailableError("Class names contain duplicates.")
    return list(raw)


def load_model(path: Path) -> "Any":
    """Load a ``.keras`` / SavedModel export. Never call this per request."""
    if not path.exists():
        raise ModelNotAvailableError(f"Model file not found: {path}")
    try:
        import tensorflow as tf

        # `compile=False`: inference does not need the optimizer state, and
        # skipping it keeps startup fast and memory small.
        return tf.keras.models.load_model(path, compile=False)
    except ModelNotAvailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as a 503 upstream
        raise ModelNotAvailableError(f"Model could not be loaded: {exc}") from exc


class ModelService:
    """Holds the loaded model in memory for the life of the process."""

    def __init__(
        self,
        model_path: Path,
        class_names_path: Path,
        model_version: str,
        image_size: int = 256,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.model_version = model_version
        self.image_size = image_size
        self._model: Any | None = None
        self._class_names: list[str] = []
        self._load_error: str | None = None

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        """Load model + class mapping. Safe to call once at startup."""
        try:
            self._class_names = load_class_names(self.class_names_path)
            self._model = load_model(self.model_path)
            output_shape = self._model.output_shape
            n_outputs = output_shape[-1] if isinstance(output_shape, tuple) else None
            if n_outputs is not None and n_outputs != len(self._class_names):
                raise ModelNotAvailableError(
                    f"Model outputs {n_outputs} classes but class_names.json lists "
                    f"{len(self._class_names)}."
                )
            self._load_error = None
            logger.info(
                "Model loaded: version=%s classes=%d path=%s",
                self.model_version,
                len(self._class_names),
                self.model_path,
            )
        except ModelNotAvailableError as exc:
            self._model = None
            self._class_names = []
            self._load_error = str(exc)
            logger.warning("Model unavailable: %s", exc)

    def unload(self) -> None:
        self._model = None
        self._class_names = []
        self._load_error = None

    # -- state ---------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None and bool(self._class_names)

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    # -- inference -----------------------------------------------------------
    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Run one prediction. Raises on invalid input or inference failure."""
        if not self.is_loaded:
            raise ModelNotAvailableError(
                self._load_error or "The inference model is not available."
            )

        batch = decode_and_prepare(image_bytes, image_size=self.image_size)
        try:
            import numpy as np

            probabilities = self._model.predict(batch, verbose=0)[0]
            probabilities = np.asarray(probabilities, dtype="float64").ravel()
        except Exception as exc:  # noqa: BLE001 - surfaced as a 500/503 upstream
            raise InferenceError(f"Inference failed: {exc}") from exc

        if probabilities.size != len(self._class_names):
            raise InferenceError(
                "Model returned an unexpected number of class probabilities."
            )

        index = int(probabilities.argmax())
        return {
            "predicted_class": self._class_names[index],
            "confidence": float(probabilities[index]),
            "model_version": self.model_version,
            "class_index": index,
            "all_probabilities": {
                name: float(prob) for name, prob in zip(self._class_names, probabilities)
            },
        }
