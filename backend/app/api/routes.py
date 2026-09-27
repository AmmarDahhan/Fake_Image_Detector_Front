"""API routes.

Infrastructure only: validate the upload, hand off to the inference seam,
normalise the result. No model knowledge lives here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.validation import UploadRejected, read_validated
from app.schemas.analysis import AnalysisResponse, ErrorResponse, HealthResponse
from app.services.model_service import (
    ImagePayload,
    ModelInferenceError,
    ModelNotAvailableError,
    get_model_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_UPLOAD_ERRORS = {
    "application/json": {"schema": ErrorResponse.model_json_schema()},
}


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness probe.

    Reports that the API process is up and serving. It intentionally does
    NOT report model readiness: loading the ~544 MiB checkpoint to answer a
    probe would make the probe the slow part, and a probe that reported model
    state would stop being a liveness check. Model readiness is observable
    through ``POST /analyze`` returning 503.
    """
    return HealthResponse(status="ok")


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["analysis"],
    responses={
        400: _UPLOAD_ERRORS,
        413: _UPLOAD_ERRORS,
        422: _UPLOAD_ERRORS,
        503: _UPLOAD_ERRORS,
    },
    summary="Analyse an uploaded image",
)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """Analyse an image and return its verdict and confidence.

    Contract::

        {"verdict": "real" | "fake", "confidence": 0.0-1.0}

    ``verdict`` is binary. The model itself classifies into
    ``full_synthetic`` / ``real`` / ``tampered``; that collapse is provisional
    and defined once, in ``app/services/model_class_map.py``.

    Status codes:
        200  prediction produced
        400  the upload is not a decodable image
        413  the upload exceeds the size limit
        422  no file part in the request
        500  inference failed for this image
        503  the model could not be loaded
    """
    try:
        data = await read_validated(file, file.filename, file.content_type)
    except UploadRejected as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.message
        ) from exc

    payload = ImagePayload(
        data=data,
        filename=file.filename or "upload",
        content_type=file.content_type,
    )

    try:
        prediction = get_model_service().analyze(payload)
    except ModelNotAvailableError as exc:
        # The model is missing or unloadable. Reported as a service problem,
        # never as a verdict.
        logger.warning("Model unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The analysis model is not available.",
        ) from exc
    except ModelInferenceError as exc:
        # Traceback goes to the log, not to the client.
        logger.exception("Model inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed.",
        ) from exc

    try:
        return AnalysisResponse.from_prediction(prediction)
    except (ValueError, TypeError) as exc:
        logger.exception("Model prediction did not match the API contract")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis returned an unexpected result.",
        ) from exc
