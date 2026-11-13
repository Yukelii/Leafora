"""Bridge between the Kivy front end and the Flask back end.

Your groupmate only needs to match this contract:

    POST  {API_BASE_URL}/api/scan
    form-data: image=<jpg file>
    returns JSON, e.g.:
        {"diagnosis": "Early Blight", "confidence": 0.88}

Change API_BASE_URL to the IP of the PC/server running Flask
(phone and PC must be on the same Wi-Fi, e.g. http://192.168.1.10:5000).
"""
import os
import threading

from kivy.clock import Clock

API_BASE_URL = os.environ.get("LEAFORA_API_URL", "http://192.168.1.10:5000")
SCAN_ENDPOINT = "/api/predict"
TIMEOUT = 30


def _to_scan_result(raw):
    """Turn the backend's /api/predict reply into the {diagnosis, confidence} shape the screens use."""
    # e.g. "Tomato___Early_blight" -> "Early blight"
    name = str(raw.get("predicted_class", "Unknown")).split("___", 1)[-1].replace("_", " ")
    return {"diagnosis": name, "confidence": raw.get("confidence")}


def upload_scan(path, on_ok, on_err):
    """Send the captured photo in a background thread.

    on_ok(dict) / on_err(str) are always called back on the Kivy main thread,
    so they can safely touch widgets.
    """
    def work():
        try:
            import requests
            with open(path, "rb") as f:
                resp = requests.post(
                    API_BASE_URL + SCAN_ENDPOINT,
                    files={"image": (os.path.basename(path), f, "image/jpeg")},
                    timeout=TIMEOUT,
                )
            resp.raise_for_status()
            data = _to_scan_result(resp.json())
            Clock.schedule_once(lambda dt: on_ok(data))
        except Exception as exc:  # network down, server error, bad JSON...
            msg = str(exc)
            Clock.schedule_once(lambda dt: on_err(msg))

    threading.Thread(target=work, daemon=True).start()


# ------------------------------------------------------------ live weather
# Open-Meteo is free and needs no API key.
# The location is approximate: it comes from the internet address (city level),
# so no phone permission is needed.
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
LOCATION_URL = "https://ipapi.co/json/"


def fetch_weather(on_ok, on_err):
    """on_ok({'temp_c': float, 'code': int}) or on_err(str) on the Kivy main thread."""
    def work():
        try:
            import requests
            loc = requests.get(LOCATION_URL, timeout=10)
            loc.raise_for_status()
            where = loc.json()
            resp = requests.get(
                WEATHER_URL,
                params={"latitude": where["latitude"], "longitude": where["longitude"],
                        "current": "temperature_2m,weather_code", "timezone": "auto"},
                timeout=15,
            )
            resp.raise_for_status()
            cur = resp.json()["current"]
            data = {"temp_c": float(cur["temperature_2m"]), "code": int(cur["weather_code"])}
            Clock.schedule_once(lambda dt: on_ok(data))
        except Exception as exc:
            msg = str(exc)
            Clock.schedule_once(lambda dt: on_err(msg))

    threading.Thread(target=work, daemon=True).start()