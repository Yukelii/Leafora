"""Shared error envelope and exception handlers.

Every error response looks the same, so the mobile app has one shape to parse:

    {"error": {"code": "INVALID_IMAGE", "message": "..."}}
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {
            400: "BAD_REQUEST",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "PAYLOAD_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            422: "VALIDATION_ERROR",
            500: "INTERNAL_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }.get(exc.status_code, "ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code, str(exc.detail)),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Keep it concise: name the offending fields, do not dump the schema.
        problems = []
        for err in exc.errors():
            location = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
            problems.append(f"{location}: {err.get('msg', 'invalid value')}")
        message = "; ".join(problems) or "The request payload is invalid."
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_body("VALIDATION_ERROR", message),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Log the detail, never send a stack trace to the client.
        logger.exception("Unhandled server error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body("INTERNAL_ERROR", "An unexpected error occurred."),
        )
