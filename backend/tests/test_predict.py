"""POST /api/predict"""

from __future__ import annotations

import pytest

from app.config import settings
from tests.conftest import CLASS_NAMES, StubModelService, make_image_bytes


def test_predict_valid_image_returns_prediction(client):
    response = client.post(
        "/api/predict",
        files={"image": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"]
    assert "timestamp" in body


def test_predict_response_matches_schema_keys(client):
    body = client.post(
        "/api/predict", files={"image": ("leaf.png", make_image_bytes(fmt="PNG"), "image/png")}
    ).json()
    assert set(body) == {"predicted_class", "confidence", "model_version", "timestamp"}


def test_predict_missing_image_is_400(client):
    response = client.post("/api/predict")
    assert response.status_code == 400
    assert response.json()["error"]["code"] in {"BAD_REQUEST", "INVALID_IMAGE"}


def test_predict_empty_file_is_400(client):
    response = client.post(
        "/api/predict", files={"image": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_predict_unsupported_type_is_415(client):
    response = client.post(
        "/api/predict",
        files={"image": ("notes.txt", b"just some text", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_IMAGE_TYPE"


def test_predict_corrupted_image_is_422(client):
    response = client.post(
        "/api/predict",
        files={"image": ("broken.jpg", b"\xff\xd8\xff\xe0not-a-real-jpeg", "image/jpeg")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_predict_tiny_image_is_422(client):
    response = client.post(
        "/api/predict",
        files={"image": ("tiny.png", make_image_bytes(size=(8, 8), fmt="PNG"), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_predict_model_unavailable_is_503(app, client):
    broken = StubModelService()
    broken.unload()
    broken._load_error = "Model file not found"
    app.state.model_service = broken
    from app.services.prediction_service import PredictionService

    app.state.prediction_service = PredictionService(broken)
    response = client.post(
        "/api/predict", files={"image": ("leaf.jpg", make_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_predict_inference_failure_is_500(app, client):
    from app.ml.inference import InferenceError
    from app.services.prediction_service import PredictionService

    failing = StubModelService(fail=InferenceError("boom"))
    app.state.model_service = failing
    app.state.prediction_service = PredictionService(failing)
    response = client.post(
        "/api/predict", files={"image": ("leaf.jpg", make_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INFERENCE_FAILED"


def test_predict_oversized_upload_is_413(client, settings_snapshot):
    settings.__dict__["max_upload_bytes"] = 512
    response = client.post(
        "/api/predict",
        files={"image": ("big.jpg", make_image_bytes(size=(1200, 1200)), "image/jpeg")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


@pytest.mark.parametrize("fmt,content_type", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_predict_accepts_allowed_formats(client, fmt, content_type):
    response = client.post(
        "/api/predict",
        files={"image": (f"leaf.{fmt.lower()}", make_image_bytes(fmt=fmt), content_type)},
    )
    assert response.status_code == 200
