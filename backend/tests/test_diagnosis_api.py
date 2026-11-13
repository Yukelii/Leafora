"""HTTP tests for POST /api/diagnose and GET /api/diagnose/symptoms.

The model is the shared stub from conftest, so these test the routing, the
answer parsing and the decision gate - not TensorFlow.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.sensor import SensorReading
from app.services.prediction_service import PredictionService
from app.services.sensor_service import SensorService
from tests.conftest import StubModelService, make_image_bytes


def build_client(winning_index: int = 1) -> TestClient:
    application = create_app()
    application.state.model_service = StubModelService(winning_index=winning_index)
    application.state.prediction_service = PredictionService(application.state.model_service)
    application.state.sensor_service = SensorService(history_limit=5)
    return TestClient(application)


def post_diagnose(client: TestClient, answers: dict | None = None, payload: bytes | None = None):
    files = {"image": ("leaf.jpg", payload or make_image_bytes(), "image/jpeg")}
    data = {}
    if answers is not None:
        data["answers"] = json.dumps(answers)
    return client.post("/api/diagnose", files=files, data=data)


def test_checklist_lists_every_question_and_option():
    with build_client() as client:
        response = client.get("/api/diagnose/symptoms")
    assert response.status_code == 200
    body = response.json()
    assert len(body["questions"]) >= 6
    for question in body["questions"]:
        assert question["prompt"]
        assert len(question["options"]) >= 2
    assert body["evidence"], "the honesty note must be shipped with the questions"


def test_photo_only_diagnose_returns_a_status_and_ranked_list():
    with build_client() as client:
        response = post_diagnose(client)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"confident", "uncertain", "disagree"}
    assert len(body["top_predictions"]) == 3
    assert body["photo_prediction"] is not None
    assert body["symptom_prediction"] is None
    assert body["weights"]["photo"] == 1.0
    assert body["weights"]["symptoms"] == 0.0
    assert body["weights"]["climate"] is None  # no sensor history was sent
    assert body["model_version"] == "leafora-densenet121-test"


def test_high_confidence_photo_is_labelled_confident():
    with build_client() as client:
        response = post_diagnose(client)
    body = response.json()
    assert body["status"] == "confident"
    assert body["predicted_class"]


def test_answers_are_accepted_and_change_the_outcome():
    # The stub reports class index 1 (Early blight) at ~0.91; a Leaf Mold
    # checklist must disagree with it.
    with build_client() as client:
        plain = post_diagnose(client).json()
        answered = post_diagnose(
            client, {"damage_type": "fuzzy", "underside": "grey_violet"}
        ).json()
    assert plain["status"] == "confident"
    assert answered["status"] == "disagree"
    assert answered["predicted_class"] is None
    assert answered["symptom_prediction"]["predicted_class"] == "Tomato___Leaf_Mold"


def test_every_question_is_its_own_dropdown_in_the_api_docs():
    # Swagger must show the questions as pickable form fields, not as one
    # hidden JSON string.
    from app.decision.symptoms import QUESTIONS

    application = create_app()
    body = application.openapi()["components"]["schemas"][
        "Body_diagnose_api_diagnose_post"
    ]["properties"]
    for question in QUESTIONS:
        assert question.key in body, f"{question.key} is missing from the form"
        assert question.prompt in body[question.key]["description"]
        for option in question.options:
            assert option.key in body[question.key]["description"]


def test_form_dropdowns_are_scored_like_json_answers():
    # Sending the answers as form fields must give the same verdict as JSON.
    with build_client() as client:
        as_json = post_diagnose(
            client, {"damage_type": "fuzzy", "underside": "grey_violet"}
        ).json()
        as_form = client.post(
            "/api/diagnose",
            files={"image": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
            data={"damage_type": "fuzzy", "underside": "grey_violet"},
        ).json()
    assert as_form["status"] == as_json["status"] == "disagree"
    assert as_form["symptom_prediction"] == as_json["symptom_prediction"]


def test_an_invented_dropdown_option_is_rejected():
    with build_client() as client:
        response = client.post(
            "/api/diagnose",
            files={"image": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
            data={"damage_type": "purple_spots"},
        )
    assert response.status_code == 422


def test_missing_image_is_400():
    with build_client() as client:
        response = client.post("/api/diagnose", data={"answers": "{}"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_answers_that_are_not_json_are_400():
    with build_client() as client:
        response = client.post(
            "/api/diagnose",
            files={"image": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
            data={"answers": "damage=fuzzy"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ANSWERS"


def test_answers_that_are_a_list_are_400():
    with build_client() as client:
        response = client.post(
            "/api/diagnose",
            files={"image": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
            data={"answers": json.dumps(["fuzzy"])},
        )
    assert response.status_code == 400


def test_unknown_question_is_422():
    with build_client() as client:
        response = post_diagnose(client, {"what_colour": "green"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNKNOWN_SYMPTOM_ANSWER"


def test_unknown_option_is_422():
    with build_client() as client:
        response = post_diagnose(client, {"damage_type": "purple_spots"})
    assert response.status_code == 422


def test_unsupported_image_type_is_415():
    with build_client() as client:
        response = client.post(
            "/api/diagnose",
            files={"image": ("leaf.gif", b"GIF89a", "image/gif")},
        )
    assert response.status_code == 415


def test_corrupted_image_is_422():
    with build_client() as client:
        response = post_diagnose(client, payload=b"this is not an image")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_empty_image_is_400():
    with build_client() as client:
        response = client.post(
            "/api/diagnose",
            files={"image": ("leaf.jpg", b"", "image/jpeg")},
        )
    assert response.status_code == 400


def test_model_unavailable_is_503():
    application = create_app()
    stub = StubModelService()
    stub.unload()  # same state as a failed model load at startup
    application.state.model_service = stub
    application.state.prediction_service = PredictionService(stub)
    application.state.sensor_service = SensorService(history_limit=5)
    with TestClient(application) as client:
        response = post_diagnose(client)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def record_history(client: TestClient, device_id: str) -> None:
    """Five readings over 16 hours: humid, cool, soil moisture steady."""
    now = datetime.now(timezone.utc)
    for hours_ago in (16, 12, 8, 4, 0):
        client.app.state.sensor_service.record(
            SensorReading(
                device_id=device_id,
                soil_moisture=80,
                temperature=16,
                humidity=95,
                timestamp=now - timedelta(hours=hours_ago),
            )
        )


def test_sensor_history_of_the_device_is_a_prior_that_never_outvotes_the_photo():
    # Cool, wet history favours late blight, but the photo says early blight.
    # The history must show up and carry its weight, and the photo still wins.
    with build_client() as client:
        record_history(client, "esp32-01")
        response = client.post(
            "/api/diagnose",
            files={"image": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
            data={"device_id": "esp32-01"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["climate"]["favoured_class"] == "Tomato___Late_blight"
    assert body["weights"]["climate"] == 0.2
    assert body["status"] == "confident"
    assert body["predicted_class"] == "Tomato___Early_blight"


def test_sensor_history_from_another_device_is_ignored():
    with build_client() as client:
        record_history(client, "esp32-02")
        response = client.post(
            "/api/diagnose",
            files={"image": ("leaf.jpg", make_image_bytes(), "image/jpeg")},
            data={"device_id": "esp32-01"},
        )
    body = response.json()
    assert body["climate"] is None
    assert body["weights"]["climate"] is None
