"""Sensor processing.

The backend is intentionally stateless in production. A real deployment has no
place to keep the user's data, and the user's data belongs in the mobile app's
local SQLite database.

For development and for the live "latest reading" endpoint, readings are kept in
a small in-process ring buffer. This is marked DEVELOPMENT-ONLY: it lives in RAM,
it is lost on restart, and it is NOT the production database.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from threading import Lock

from app.schemas.sensor import SensorReading

logger = logging.getLogger(__name__)


class SensorService:
    """Keeps the most recent readings in memory (development-only)."""

    def __init__(self, history_limit: int = 200) -> None:
        self._readings: deque[SensorReading] = deque(maxlen=max(1, history_limit))
        self._lock = Lock()

    def record(self, reading: SensorReading) -> None:
        with self._lock:
            self._readings.append(reading)
        logger.info(
            "Sensor reading accepted: device=%s timestamp=%s",
            reading.device_id,
            reading.timestamp.isoformat(),
        )

    def latest(self) -> SensorReading | None:
        with self._lock:
            return self._readings[-1] if self._readings else None

    def recent(self, limit: int = 20) -> list[SensorReading]:
        with self._lock:
            return list(self._readings)[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._readings.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._readings)


def is_stale(reading: SensorReading, max_age_seconds: int = 3600) -> bool:
    """True when a reading is older than ``max_age_seconds``."""
    age = (datetime.now(reading.timestamp.tzinfo) - reading.timestamp).total_seconds()
    return age > max_age_seconds
