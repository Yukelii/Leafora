"""Prediction service: glue between the HTTP layer and the ML model.

The API layer must not build or hold a model. It asks this service, which in
turn asks the single :class:`~app.ml.inference.ModelService` loaded at startup.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.ml.inference import InferenceError, ModelNotAvailableError, ModelService

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, model_service: ModelService) -> None:
        self._model_service = model_service

    @property
    def model_service(self) -> ModelService:
        return self._model_service

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Run one prediction and return a response-ready dict.

        Raises:
            ModelNotAvailableError: model not loaded -> 503
            InferenceError: prediction failed -> 500
            InvalidImageError: bad image -> 415/422 (raised by preprocessing)
        """
        result = self._model_service.predict(image_bytes)
        return {
            "predicted_class": result["predicted_class"],
            "confidence": round(result["confidence"], 6),
            "model_version": result["model_version"],
            "timestamp": datetime.now(timezone.utc),
            "probabilities": result["all_probabilities"],
        }


__all__ = [
    "PredictionService",
    "ModelNotAvailableError",
    "InferenceError",
]
