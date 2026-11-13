"""GET /api/health"""

from __future__ import annotations

from tests.conftest import StubModelService


def test_health_ok_when_model_loaded(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]
    assert body["num_classes"] == 10


def test_health_reports_degraded_when_model_missing(app, client):
    broken = StubModelService()
    broken.unload()
    broken._load_error = "Model file not found: nowhere.keras"
    app.state.model_service = broken
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"
    assert "not available" in (body["detail"] or "").lower() or body["detail"]


def test_health_does_not_claim_loaded_for_a_real_missing_model(tmp_path):
    """The honest path: point a fresh ModelService at nothing."""
    from app.ml.inference import ModelService

    service = ModelService(
        model_path=tmp_path / "missing.keras",
        class_names_path=tmp_path / "missing.json",
        model_version="test",
    )
    service.load()
    assert service.is_loaded is False
    assert service.load_error
