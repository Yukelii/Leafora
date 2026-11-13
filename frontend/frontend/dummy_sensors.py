"""Dummy sensor readings for the Vitals card, until the ESP32 is connected.

USE_DUMMY_SENSORS = True  -> the app makes up realistic readings (this file).
USE_DUMMY_SENSORS = False -> the app does not touch the vitals; wire the real
                             readings from the backend's GET /api/sensors/latest
                             into the same three app values later.

Readings follow a slow random walk, so the numbers drift like a real plant
bed instead of jumping around. Units match the backend's working assumptions:
soil moisture in %, temperature in Celsius, humidity in % RH.
"""
import random

USE_DUMMY_SENSORS = True
DUMMY_INTERVAL_SECONDS = 10 * 60   # how often the vitals are refreshed

# (low, high) ranges the simulated values stay inside
RANGES = {
    'soil_moisture': (40.0, 85.0),
    'temperature': (20.0, 32.0),
    'humidity': (50.0, 90.0),
}


class DummySensorFeed:
    """Keeps a small state and returns the next made-up reading on each call."""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self._state = {
            'soil_moisture': 68.0,
            'temperature': 26.0,
            'humidity': 66.0,
        }

    def next_reading(self):
        for key, (low, high) in RANGES.items():
            step = self._rng.uniform(-2.0, 2.0)
            value = self._state[key] + step
            self._state[key] = min(high, max(low, value))
        return {key: round(value, 1) for key, value in self._state.items()}
