"""Shared test fixtures.

The tests never need a trained model or a real ESP32. A stub model service
returns fixed probabilities so the HTTP layer can be tested for real: real
routing, real validation, real image decoding, fake prediction.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import create_app
from app.ml.inference import ModelService
from app.services.prediction_service import PredictionService
from app.services.sensor_service import SensorService

CLASS_NAMES = json.loads(
    (Path(__file__).resolve().parents[1] / "app" / "ml" / "class_names.json").read_text(
        encoding="utf-8"
    )
)


class StubModelService(ModelService):
    """Behaves like a loaded model, without TensorFlow or a real export."""

    def __init__(self, winning_index: int = 1, fail: Exception | None = None) -> None:
        super().__init__(
            model_path=Path("stub-model.keras"),
            class_names_path=Path("stub-class_names.json"),
            model_version="leafora-densenet121-test",
        )
        self._class_names = list(CLASS_NAMES)
        self._model = object()
        self._winning_index = winning_index
        self._fail = fail
        self.predict_calls = 0

    def load(self) -> None:
        """No-op: the stub is already 'loaded' and must survive app startup."""
        self._load_error = None

    def predict(self, image_bytes: bytes):
        self.predict_calls += 1
        if not self.is_loaded:
            from app.ml.inference import ModelNotAvailableError

            raise ModelNotAvailableError(self._load_error or "model not loaded")
        from app.ml.preprocessing import decode_and_prepare

        decode_and_prepare(image_bytes)  # real preprocessing, real errors
        if self._fail is not None:
            raise self._fail
        probabilities = {name: 0.01 for name in self._class_names}
        winner = self._class_names[self._winning_index]
        probabilities[winner] = 1.0 - 0.01 * (len(self._class_names) - 1)
        return {
            "predicted_class": winner,
            "confidence": probabilities[winner],
            "model_version": self.model_version,
            "class_index": self._winning_index,
            "all_probabilities": probabilities,
        }


def make_image_bytes(
    size: tuple[int, int] = (320, 320),
    mode: str = "RGB",
    color: tuple[int, int, int] = (90, 140, 60),
    fmt: str = "JPEG",
) -> bytes:
    buffer = io.BytesIO()
    if mode == "L":
        # A palette-free greyscale image takes a single int, not a tuple.
        Image.new(mode, size, color[0]).save(buffer, format=fmt)
    else:
        Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def app():
    """App with a stub model, so no TensorFlow load happens during the tests."""
    application = create_app()
    application.state.model_service = StubModelService()
    application.state.prediction_service = PredictionService(application.state.model_service)
    application.state.sensor_service = SensorService(history_limit=10)
    return application


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def settings_snapshot():
    """Restore mutable settings values after a test changes them."""
    saved = (settings.max_upload_bytes, settings.allowed_image_types)
    yield
    settings.__dict__["max_upload_bytes"] = saved[0]
    settings.__dict__["allowed_image_types"] = saved[1]
