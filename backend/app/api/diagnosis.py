"""POST /api/diagnose - photo prediction plus the symptom checklist.

The image endpoint (`/api/predict`) is left exactly as it was: it answers "what
does the model see". This endpoint answers "what should we tell the user", and
it refuses to sound certain when the evidence is thin.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Optional, Type

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.deps import get_diagnosis_engine, get_prediction_service, get_sensor_service
from app.api.errors import error_body
from app.config import settings
from app.decision.diagnosis import DiagnosisEngine
from app.decision.environment import EnvironmentSettings, build_prior
from app.decision.symptoms import (
    EVIDENCE_NOTE,
    QUESTIONS,
    Question,
    UnknownAnswerError,
)
from app.ml.inference import InferenceError, ModelNotAvailableError
from app.ml.preprocessing import InvalidImageError
from app.schemas.diagnosis import DiagnosisResponse, SymptomChecklistResponse
from app.services.prediction_service import PredictionService
from app.services.sensor_service import SensorService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["diagnosis"])

#: The climate prior is built from the same settings the diagnosis engine uses.
ENVIRONMENT_SETTINGS = EnvironmentSettings(
    min_readings=settings.climate_min_readings,
    window_hours=settings.climate_window_hours,
    wet_humidity=settings.climate_wet_humidity,
    strength=settings.climate_strength,
)

MAX_ANSWERS = len(QUESTIONS)


def _option_enum(question: Question) -> Type[Enum]:
    """One dropdown per question, so clients cannot send an invented answer."""
    values = {option.key: option.key for option in question.options}
    return Enum(f"{question.key}_choice", values, type=str)  # type: ignore[return-value]


OPTION_ENUMS: dict[str, Type[Enum]] = {
    question.key: _option_enum(question) for question in QUESTIONS
}


def _describe(question_key: str) -> str:
    """The description shown under one form dropdown: prompt plus every option."""
    question = next(item for item in QUESTIONS if item.key == question_key)
    options = "; ".join(f"{option.key} = {option.label}" for option in question.options)
    return f"{question.prompt} Options: {options}"


def collect_answers(**picked: Any) -> dict[str, str] | None:
    """Turn the form dropdown values into ``{question_key: option_key}``."""
    answers: dict[str, str] = {}
    for key, value in picked.items():
        if value is None:
            continue
        answers[key] = value.value if isinstance(value, Enum) else str(value)
    return answers or None


def merge_answers(
    json_answers: dict[str, str] | None,
    form_answers: dict[str, str] | None,
) -> dict[str, str] | None:
    """Combine the JSON `answers` field with the form dropdowns (dropdowns win)."""
    if not json_answers and not form_answers:
        return None
    merged: dict[str, str] = dict(json_answers or {})
    merged.update(form_answers or {})
    return merged or None

def answers_from_form(form: Any) -> dict[str, str] | None:
    """Read the per-question form fields into ``{question_key: option_key}``."""
    if form is None:
        return None
    picked: dict[str, str] = {}
    raw = form.model_dump(exclude_none=True) if hasattr(form, "model_dump") else dict(form)
    for key, value in raw.items():
        if value is None:
            continue
        picked[str(key)] = value.value if isinstance(value, Enum) else str(value)
    return picked or None


def merge_answers(
    json_answers: dict[str, str] | None,
    form_answers: dict[str, str] | None,
) -> dict[str, str] | None:
    """Combine the JSON `answers` field with the form fields (form wins)."""
    if not json_answers and not form_answers:
        return None
    merged: dict[str, str] = dict(json_answers or {})
    merged.update(form_answers or {})
    return merged or None


def parse_answers(raw: str | None) -> dict[str, str] | None:
    """Read the `answers` form field: a small JSON object of question -> option.

    Raises:
        ValueError: not JSON, not an object, too many keys, or a non-string value.
    """
    if raw is None or not raw.strip():
        return None
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("The 'answers' field must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("The 'answers' field must be a JSON object.")
    if len(parsed) > MAX_ANSWERS:
        raise ValueError(f"The 'answers' field must hold at most {MAX_ANSWERS} entries.")
    cleaned: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Every answer must be a string key and a string option.")
        cleaned[key] = value
    return cleaned or None


@router.get(
    "/diagnose/symptoms",
    response_model=SymptomChecklistResponse,
    summary="The symptom checklist the mobile app renders",
)
def symptom_checklist() -> SymptomChecklistResponse:
    """Return every question, its options and the honesty note beside them."""
    return SymptomChecklistResponse(
        questions=[
            {
                "key": question.key,
                "prompt": question.prompt,
                "options": [
                    {"key": option.key, "label": option.label}
                    for option in question.options
                ],
            }
            for question in QUESTIONS
        ],
        evidence=[EVIDENCE_NOTE],
    )


@router.post(
    "/diagnose",
    response_model=DiagnosisResponse,
    summary="Diagnose from a photo and optional symptom answers",
    description=(
        "Two ways to send the symptom answers:\n\n"
        "1. **Pick them in this form** - every question is its own dropdown, and "
        "only the listed options are accepted.\n"
        "2. **Send the `answers` JSON field** - e.g. "
        '`{"damage_type": "fuzzy", "underside": "grey_violet"}`, which is what '
        "the mobile app uses.\n\n"
        "Get the questions with `GET /api/diagnose/symptoms`. The reply is "
        "`confident`, `uncertain` or `disagree` - never a guess dressed up as a "
        "diagnosis."
    ),
    responses={
        400: {"description": "Missing image or unreadable answers field"},
        413: {"description": "Image too large"},
        415: {"description": "Unsupported image type"},
        422: {"description": "Image could not be decoded, or an unknown answer"},
        500: {"description": "Inference failure"},
        503: {"description": "Model not loaded"},
    },
)
async def diagnose(
    image: UploadFile | None = File(default=None),
    damage_type: Optional[OPTION_ENUMS["damage_type"]] = Form(
        default=None, description=_describe("damage_type")
    ),
    spot_centre: Optional[OPTION_ENUMS["spot_centre"]] = Form(
        default=None, description=_describe("spot_centre")
    ),
    yellow_halo: Optional[OPTION_ENUMS["yellow_halo"]] = Form(
        default=None, description=_describe("yellow_halo")
    ),
    leaf_curling: Optional[OPTION_ENUMS["leaf_curling"]] = Form(
        default=None, description=_describe("leaf_curling")
    ),
    mottle: Optional[OPTION_ENUMS["mottle"]] = Form(
        default=None, description=_describe("mottle")
    ),
    underside: Optional[OPTION_ENUMS["underside"]] = Form(
        default=None, description=_describe("underside")
    ),
    webbing: Optional[OPTION_ENUMS["webbing"]] = Form(
        default=None, description=_describe("webbing")
    ),
    first_leaves: Optional[OPTION_ENUMS["first_leaves"]] = Form(
        default=None, description=_describe("first_leaves")
    ),
    answers: str | None = Form(
        default=None,
        description=(
            "Optional JSON object of question -> option keys, e.g. "
            '{"damage_type": "fuzzy"}. The dropdowns above win if both are sent.'
        ),
    ),
    device_id: str | None = Form(
        default=None,
        description=(
            "Optional ESP32 device id. When its recent readings are already in the "
            "store, they are used as a climate prior for this diagnosis."
        ),
    ),
    service: PredictionService = Depends(get_prediction_service),
    engine: DiagnosisEngine = Depends(get_diagnosis_engine),
    sensors: SensorService = Depends(get_sensor_service),
) -> JSONResponse:
    """Return one of three honest outcomes: confident, uncertain or disagree."""
    if image is None or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file was provided. Send the file as field 'image'.",
        )

    content_type = (image.content_type or "").lower()
    if content_type and content_type not in settings.allowed_image_types:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content=error_body(
                "UNSUPPORTED_IMAGE_TYPE",
                f"Unsupported image type '{content_type}'. "
                f"Allowed: {', '.join(settings.allowed_image_types)}.",
            ),
        )

    try:
        parsed_answers = merge_answers(
            parse_answers(answers),
            collect_answers(
                damage_type=damage_type,
                spot_centre=spot_centre,
                yellow_halo=yellow_halo,
                leaf_curling=leaf_curling,
                mottle=mottle,
                underside=underside,
                webbing=webbing,
                first_leaves=first_leaves,
            ),
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_body("INVALID_ANSWERS", str(exc)),
        )

    payload = await image.read()
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is empty.",
        )
    if len(payload) > settings.max_upload_bytes:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=error_body(
                "IMAGE_TOO_LARGE",
                f"Image exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
            ),
        )

    try:
        result = service.predict(payload)
    except InvalidImageError as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_body("INVALID_IMAGE", str(exc)),
        )
    except ModelNotAvailableError:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_body(
                "MODEL_UNAVAILABLE",
                "The inference model is not available. Try again later.",
            ),
        )
    except InferenceError:
        logger.exception("Inference failed for diagnose upload '%s'", image.filename)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body("INFERENCE_FAILED", "The prediction could not be completed."),
        )

    try:
        climate = None
        if device_id:
            climate = build_prior(
                [
                    reading
                    for reading in sensors.recent(settings.climate_max_readings)
                    if reading.device_id == device_id.strip()
                ],
                ENVIRONMENT_SETTINGS,
            )
        diagnosis = engine.evaluate(result["probabilities"], parsed_answers, climate)
    except UnknownAnswerError as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_body("UNKNOWN_SYMPTOM_ANSWER", str(exc)),
        )

    body = diagnosis.to_dict()
    body["model_version"] = result["model_version"]
    logger.info(
        "Diagnosis: status=%s photo=%s %.4f -> class=%s",
        body["status"],
        result["predicted_class"],
        result["confidence"],
        body["predicted_class"],
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=DiagnosisResponse.model_validate(body).model_dump(mode="json"),
    )
