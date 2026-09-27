"""Public API contract.

PROVISIONAL. These schemas define the frontend/backend integration contract
agreed before the model exists. They are intentionally model-agnostic: only
``verdict`` and ``confidence`` are exposed, so the shape can be adjusted once
the AI team confirms the real model contract.

Do not add model-specific fields (class names, per-class scores, artifact
metadata) here without renegotiating the contract with the frontend.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Verdict(str, Enum):
    """Binary authenticity outcome.

    The two members mirror the frontend's ``core.models.Verdict`` so the
    Streamlit side can consume the response without translation.
    """

    REAL = "real"
    FAKE = "fake"


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str = Field(examples=["ok"])


class AnalysisResponse(BaseModel):
    """Response for ``POST /analyze``.

    Example::

        {"verdict": "fake", "confidence": 0.94}
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"verdict": "fake", "confidence": 0.94},
        }
    )

    verdict: Verdict = Field(description="Classification outcome.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict, normalised to 0.0-1.0.",
    )

    @classmethod
    def from_prediction(cls, prediction: object) -> "AnalysisResponse":
        """Normalise a model-layer prediction onto the public contract.

        This is the single mapping point between the inference layer and the
        API. The model service is expected to return an object exposing
        ``verdict`` and ``confidence``; anything else is a contract violation
        and is surfaced as a 500 rather than being silently coerced.
        """
        verdict = getattr(prediction, "verdict", None)
        confidence = getattr(prediction, "confidence", None)

        if verdict is None or confidence is None:
            raise ValueError(
                "Model prediction must expose 'verdict' and 'confidence' "
                f"attributes; got {type(prediction).__name__!r}."
            )

        return cls(verdict=verdict, confidence=confidence)


class ErrorResponse(BaseModel):
    """Error body shape returned for rejected uploads and unavailable models."""

    detail: str = Field(examples=["Unsupported file type 'txt'."])
