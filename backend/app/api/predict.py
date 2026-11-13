"""POST /api/predict - tomato leaf image classification."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.deps import get_prediction_service
from app.api.errors import error_body
from app.config import settings
from app.ml.inference import InferenceError, ModelNotAvailableError
from app.ml.preprocessing import InvalidImageError
from app.schemas.prediction import PredictionResponse
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify one tomato leaf image",
    responses={
        400: {"description": "Missing or empty image"},
        413: {"description": "Image too large"},
        415: {"description": "Unsupported image format"},
        422: {"description": "Image could not be decoded"},
        500: {"description": "Inference failure"},
        503: {"description": "Model not loaded"},
    },
)
async def predict(
    image: UploadFile | None = File(default=None),
    service: PredictionService = Depends(get_prediction_service),
) -> JSONResponse:
    """Accept `multipart/form-data` with one `image` file and return the top class.

    The class name always comes from the trained model's `class_names.json`, so
    it can never be a guessed or invented label.
    """
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
        logger.exception("Inference failed for upload '%s'", image.filename)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body("INFERENCE_FAILED", "The prediction could not be completed."),
        )

    logger.info(
        "Prediction: class=%s confidence=%.4f",
        result["predicted_class"],
        result["confidence"],
    )
    body = PredictionResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        timestamp=result["timestamp"],
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump(mode="json"))
