"""Application configuration.

All values can be overridden with environment variables (see `.env.example`).
Nothing here holds a secret; the project has no secrets to hold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


#: Stage-2 export: the lab model fine-tuned on real field photos.
FIELD_MODEL_PATH = BACKEND_ROOT / "models" / "exported" / "leafora_densenet121_field.keras"
#: Stage-1 export: the notebook-reproduction lab model.
LAB_MODEL_PATH = BACKEND_ROOT / "models" / "exported" / "leafora_densenet121.keras"


def _default_model_path() -> Path:
    """Prefer the field fine-tuned model; fall back to the lab model."""
    return FIELD_MODEL_PATH if FIELD_MODEL_PATH.is_file() else LAB_MODEL_PATH


def _default_model_version() -> str:
    """Keep the version string honest about which export is loaded."""
    return "leafora-densenet121-field-v1" if FIELD_MODEL_PATH.is_file() else "leafora-densenet121-v1"


@dataclass(frozen=True)
class Settings:
    # --- ML model -----------------------------------------------------------
    #: Exported Keras model produced by `training/train.py`.
    model_path: Path = field(
        default_factory=lambda: _env_path("MODEL_PATH", _default_model_path())
    )
    #: JSON list with the exact class ordering used during training.
    class_names_path: Path = field(
        default_factory=lambda: _env_path(
            "CLASS_NAMES_PATH", APP_ROOT / "ml" / "class_names.json"
        )
    )
    #: Version string reported by `/api/health` and every prediction.
    model_version: str = field(
        default_factory=lambda: os.getenv("MODEL_VERSION", _default_model_version())
    )

    # --- image input --------------------------------------------------------
    #: Must match training: 256x256 RGB.
    image_size: int = field(default_factory=lambda: int(os.getenv("IMAGE_SIZE", "256")))
    #: Reject uploads above this size (bytes). 10 MiB by default.
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    )
    allowed_image_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")

    # --- sensors ------------------------------------------------------------
    #: How many recent readings the development store keeps in memory.
    sensor_history_limit: int = field(
        default_factory=lambda: int(os.getenv("SENSOR_HISTORY_LIMIT", "200"))
    )

    # --- rule-based diagnosis layer ----------------------------------------
    #: Weight of the photo probabilities when symptom answers are supplied.
    diagnosis_image_weight: float = field(
        default_factory=lambda: float(os.getenv("DIAGNOSIS_IMAGE_WEIGHT", "0.6"))
    )
    #: Weight of the symptom checklist when answers are supplied.
    diagnosis_symptom_weight: float = field(
        default_factory=lambda: float(os.getenv("DIAGNOSIS_SYMPTOM_WEIGHT", "0.4"))
    )
    #: Below this photo confidence the answer is reported as uncertain.
    diagnosis_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("DIAGNOSIS_MIN_CONFIDENCE", "0.75"))
    )
    #: Top-1 and top-2 closer than this are not separated.
    diagnosis_min_margin: float = field(
        default_factory=lambda: float(os.getenv("DIAGNOSIS_MIN_MARGIN", "0.15"))
    )

    # --- climate prior (sensor history) -------------------------------------
    #: Share of the answer the climate prior may take when readings exist.
    diagnosis_climate_weight: float = field(
        default_factory=lambda: float(os.getenv("DIAGNOSIS_CLIMATE_WEIGHT", "0.2"))
    )
    #: Largest confidence lift the prior may add when it backs the photo.
    diagnosis_climate_bonus: float = field(
        default_factory=lambda: float(os.getenv("DIAGNOSIS_CLIMATE_BONUS", "0.08"))
    )
    #: Readings needed before a climate prior is produced at all.
    climate_min_readings: int = field(
        default_factory=lambda: int(os.getenv("CLIMATE_MIN_READINGS", "3"))
    )
    #: Hours of history kept for the prior; older readings age out.
    climate_window_hours: float = field(
        default_factory=lambda: float(os.getenv("CLIMATE_WINDOW_HOURS", "24"))
    )
    #: How many stored readings the prior may look at.
    climate_max_readings: int = field(
        default_factory=lambda: int(os.getenv("CLIMATE_MAX_READINGS", "200"))
    )
    #: Relative humidity (% RH) counted as leaf wetness.
    climate_wet_humidity: float = field(
        default_factory=lambda: float(os.getenv("CLIMATE_WET_HUMIDITY", "85"))
    )
    #: How strongly the prior pulls a class (0 = no effect).
    climate_strength: float = field(
        default_factory=lambda: float(os.getenv("CLIMATE_STRENGTH", "0.35"))
    )

    # --- server -------------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
        )
    )


settings = Settings()
