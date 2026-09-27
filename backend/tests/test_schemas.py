"""Tests for the provisional response contract in app.schemas.analysis.

These pin the shape the frontend depends on, independent of the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from app.schemas.analysis import AnalysisResponse, Verdict
from app.services.model_service import ModelPrediction


class TestVerdict:
    def test_values_match_frontend_contract(self) -> None:
        assert Verdict.REAL.value == "real"
        assert Verdict.FAKE.value == "fake"

    def test_serialises_to_plain_strings(self) -> None:
        assert AnalysisResponse(verdict=Verdict.REAL, confidence=0.9).model_dump() == {
            "verdict": "real",
            "confidence": 0.9,
        }


class TestAnalysisResponse:
    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_accepts_confidence_in_range(self, confidence: float) -> None:
        assert AnalysisResponse(verdict=Verdict.FAKE, confidence=confidence)

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 42.0])
    def test_rejects_confidence_out_of_range(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            AnalysisResponse(verdict=Verdict.FAKE, confidence=confidence)

    def test_rejects_unknown_verdict(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisResponse(verdict="synthetic", confidence=0.5)

    def test_json_shape_is_exactly_the_contract(self) -> None:
        dumped = AnalysisResponse(verdict=Verdict.FAKE, confidence=0.94).model_dump_json()
        assert dumped == '{"verdict":"fake","confidence":0.94}'


class TestPredictionNormalisation:
    """AnalysisResponse.from_prediction is the model->API mapping point."""

    def test_maps_model_prediction(self) -> None:
        prediction = ModelPrediction(verdict=Verdict.REAL, confidence=0.88)
        assert AnalysisResponse.from_prediction(prediction) == AnalysisResponse(
            verdict=Verdict.REAL, confidence=0.88
        )

    def test_accepts_duck_typed_prediction(self) -> None:
        @dataclass
        class Raw:
            verdict: Verdict
            confidence: float

        response = AnalysisResponse.from_prediction(Raw(Verdict.FAKE, 0.72))
        assert response.verdict is Verdict.FAKE
        assert response.confidence == 0.72

    def test_rejects_prediction_missing_fields(self) -> None:
        @dataclass
        class Partial:
            verdict: Verdict

        with pytest.raises(ValueError, match="confidence"):
            AnalysisResponse.from_prediction(Partial(Verdict.REAL))

    def test_rejects_out_of_range_confidence_from_model(self) -> None:
        prediction = ModelPrediction(verdict=Verdict.REAL, confidence=1.7)
        with pytest.raises(ValidationError):
            AnalysisResponse.from_prediction(prediction)
