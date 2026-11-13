"""Sensor endpoints: POST /api/sensors and GET /api/sensors/latest."""

from __future__ import annotations

import pytest

VALID_PAYLOAD = {
    "device_id": "leafora-esp32-01",
    "soil_moisture": 42.5,
    "temperature": 29.4,
    "humidity": 68.2,
    "timestamp": "2025-01-15T08:30:00Z",
}


def test_submit_valid_reading_returns_ack(client):
    response = client.post("/api/sensors", json=VALID_PAYLOAD)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "success"
    assert body["device_id"] == "leafora-esp32-01"
    assert body["reading"]["soil_moisture"] == 42.5


def test_submit_without_timestamp_uses_server_time(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "timestamp"}
    response = client.post("/api/sensors", json=payload)
    assert response.status_code == 201
    assert response.json()["timestamp"]


def test_submit_optional_pressure(client):
    payload = {**VALID_PAYLOAD, "pressure": 1013.2}
    response = client.post("/api/sensors", json=payload)
    assert response.status_code == 201
    assert response.json()["reading"]["pressure"] == 1013.2


@pytest.mark.parametrize(
    "payload",
    [
        {k: v for k, v in VALID_PAYLOAD.items() if k != "device_id"},
        {k: v for k, v in VALID_PAYLOAD.items() if k != "soil_moisture"},
        {k: v for k, v in VALID_PAYLOAD.items() if k != "temperature"},
        {k: v for k, v in VALID_PAYLOAD.items() if k != "humidity"},
    ],
)
def test_missing_required_field_is_422(client, payload):
    response = client.post("/api/sensors", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    "field,value",
    [
        ("soil_moisture", 150.0),
        ("soil_moisture", -1.0),
        ("humidity", 101.0),
        ("temperature", "hot"),
        ("temperature", 200.0),
    ],
)
def test_out_of_range_or_non_numeric_is_422(client, field, value):
    response = client.post("/api/sensors", json={**VALID_PAYLOAD, field: value})
    assert response.status_code == 422


def test_extra_field_is_rejected(client):
    response = client.post("/api/sensors", json={**VALID_PAYLOAD, "battery": 99})
    assert response.status_code == 422


def test_naive_timestamp_is_rejected(client):
    response = client.post(
        "/api/sensors", json={**VALID_PAYLOAD, "timestamp": "2025-01-15T08:30:00"}
    )
    assert response.status_code == 422


def test_malformed_timestamp_is_rejected(client):
    response = client.post(
        "/api/sensors", json={**VALID_PAYLOAD, "timestamp": "not-a-date"}
    )
    assert response.status_code == 422


def test_blank_device_id_is_rejected(client):
    response = client.post("/api/sensors", json={**VALID_PAYLOAD, "device_id": "   "})
    assert response.status_code == 422


def test_non_json_body_is_422(client):
    response = client.post(
        "/api/sensors", content=b"this is not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_latest_returns_most_recent(client):
    client.post("/api/sensors", json=VALID_PAYLOAD)
    second = {**VALID_PAYLOAD, "soil_moisture": 55.0, "timestamp": "2025-01-15T09:00:00Z"}
    client.post("/api/sensors", json=second)

    response = client.get("/api/sensors/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["reading"]["soil_moisture"] == 55.0
    assert body["device_id"] == "leafora-esp32-01"


def test_latest_without_data_is_404(client):
    response = client.get("/api/sensors/latest")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NO_SENSOR_DATA"


def test_esp32_integration_round_trip(client):
    """Simulated ESP32 JSON -> validation -> processing -> response."""
    esp32_payload = {
        "device_id": "esp32-field-02",
        "soil_moisture": 31.0,
        "temperature": 30.1,
        "humidity": 74.0,
        "timestamp": "2025-02-01T06:15:00+08:00",
    }
    posted = client.post("/api/sensors", json=esp32_payload)
    assert posted.status_code == 201

    latest = client.get("/api/sensors/latest").json()
    assert latest["device_id"] == "esp32-field-02"
    assert latest["reading"]["temperature"] == 30.1
