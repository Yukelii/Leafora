"""Sensor endpoints: POST /api/sensors and GET /api/sensors/latest.

The backend validates and acknowledges; it does not own the user's data. The
mobile app stores each accepted reading in its own local SQLite database.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.api.deps import get_sensor_service
from app.api.errors import error_body
from app.schemas.sensor import (
    LatestSensorResponse,
    SensorAckResponse,
    SensorReading,
    SensorReadingIn,
)
from app.services.sensor_service import SensorService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sensors"])


@router.post(
    "/sensors",
    response_model=SensorAckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit one ESP32 sensor reading",
    responses={
        422: {"description": "Payload failed validation"},
    },
)
def submit_reading(
    payload: SensorReadingIn,
    service: SensorService = Depends(get_sensor_service),
) -> SensorAckResponse:
    """Validate an ESP32 reading and echo it back for the mobile app to persist.

    Validation happens in the Pydantic schema: required fields, numeric ranges,
    extra fields rejected, and the timestamp must carry a timezone.
    """
    reading = SensorReading.from_input(payload)
    service.record(reading)
    logger.info("Sensor reading stored in development buffer: %s", reading.device_id)
    return SensorAckResponse(
        status="success",
        device_id=reading.device_id,
        timestamp=reading.timestamp,
        reading=reading,
    )


@router.get(
    "/sensors/latest",
    response_model=LatestSensorResponse,
    summary="Latest accepted sensor reading",
    responses={404: {"description": "No reading has been received yet"}},
)
def latest_reading(
    service: SensorService = Depends(get_sensor_service),
) -> JSONResponse | LatestSensorResponse:
    """Return the newest reading, or 404 when nothing has arrived yet.

    404 (not an empty 200) so the mobile app can tell "no data" apart from
    "a reading that happens to be zero".
    """
    reading = service.latest()
    if reading is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=error_body(
                "NO_SENSOR_DATA",
                "No sensor reading has been received yet.",
            ),
        )
    return LatestSensorResponse(
        device_id=reading.device_id,
        timestamp=reading.timestamp,
        reading=reading,
    )
