"""Pydantic schemas for ESP32 sensor readings.

UNIT POLICY: the hardware group has not confirmed the units the ESP32 will send
yet, so the schema validates *numbers* only and documents units as pending.
The assumed readings, to be confirmed by the hardware group:

    soil_moisture   -- raw sensor value, 0-100 (%) is the working assumption
    temperature     -- degrees Celsius, working assumption
    humidity        -- relative humidity percent, working assumption

Each field below repeats that "unit pending" note so nobody mistakes it for a
final decision. Once the hardware is confirmed, tighten the ranges here and
update `backend/README.md`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SensorReadingIn(BaseModel):
    """Payload the ESP32 POSTs to `/api/sensors`."""

    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(..., min_length=1, max_length=64, examples=["leafora-esp32-01"])
    #: UNIT: TBD - hardware specification required. Working assumption: percent.
    soil_moisture: float = Field(..., ge=0.0, le=100.0, examples=[42.5])
    #: UNIT: TBD - hardware specification required. Working assumption: Celsius.
    temperature: float = Field(..., ge=-40.0, le=85.0, examples=[29.4])
    #: UNIT: TBD - hardware specification required. Working assumption: % RH.
    humidity: float = Field(..., ge=0.0, le=100.0, examples=[68.2])
    #: Optional: only send this if the BME280 pressure read is implemented.
    pressure: Optional[float] = Field(
        default=None, ge=300.0, le=1100.0, examples=[1013.2]
    )
    timestamp: Optional[datetime] = Field(
        default=None, examples=["2025-01-15T08:30:00Z"]
    )

    @field_validator("device_id")
    @classmethod
    def _strip_device_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("device_id must not be blank")
        return cleaned

    @field_validator("soil_moisture", "temperature", "humidity", "pressure")
    @classmethod
    def _reject_non_finite(cls, value: float | None) -> float | None:
        if value is not None and value != value:  # NaN
            raise ValueError("must be a finite number")
        return value

    @field_validator("timestamp")
    @classmethod
    def _require_timezone(cls, value: datetime | None) -> datetime | None:
        """Reject naive timestamps: an ESP32 without RTC sends UTC or nothing."""
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamp must include a timezone, e.g. 2025-01-15T08:30:00Z")
        return value


class SensorReading(BaseModel):
    """A validated reading, ready for the mobile app to store in SQLite."""

    device_id: str
    soil_moisture: float
    temperature: float
    humidity: float
    pressure: Optional[float] = None
    timestamp: datetime

    @classmethod
    def from_input(cls, payload: SensorReadingIn) -> "SensorReading":
        return cls(
            device_id=payload.device_id,
            soil_moisture=payload.soil_moisture,
            temperature=payload.temperature,
            humidity=payload.humidity,
            pressure=payload.pressure,
            timestamp=payload.timestamp or datetime.now(timezone.utc),
        )


class SensorAckResponse(BaseModel):
    """:class:`SensorReading` plus the acknowledgment the ESP32 expects."""

    status: str = Field(default="success", examples=["success"])
    device_id: str
    timestamp: datetime
    reading: SensorReading


class LatestSensorResponse(BaseModel):
    device_id: Optional[str] = Field(default=None, examples=["leafora-esp32-01"])
    timestamp: Optional[datetime] = None
    reading: Optional[SensorReading] = None
