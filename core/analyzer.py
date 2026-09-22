"""Analyzer contract.

Every analysis source (demo mock, future real backend, anything else) must
implement this interface. The UI talks to an ``Analyzer`` and never cares
which implementation produced the result - so swapping the backend later
means changing one factory/toggle, not the UI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.config import ANALYZER_BACKEND
from core.models import AnalysisResult, ImageInfo


class AnalyzerUnavailableError(RuntimeError):
    """Raised when the configured analyzer cannot serve a request."""


class Analyzer(ABC):
    """Common contract for all analysis providers."""

    @property
    def is_available(self) -> bool:
        return True

    @abstractmethod
    def analyze(self, image_bytes: bytes, image_info: ImageInfo | None = None) -> AnalysisResult:
        """Run authenticity analysis on raw image bytes and return a result."""
        raise NotImplementedError


def create_analyzer(backend: str | None = None) -> Analyzer:
    """Instantiate the analyzer selected in configuration.

    The UI only ever calls ``analyze()`` on the returned object. To connect
    the real backend later, set ``config.ANALYZER_BACKEND = "api"`` (and
    implement ``services.api_analyzer.ApiAnalyzer``) - no UI changes needed.
    """
    backend = (backend or ANALYZER_BACKEND).strip().lower()
    if backend == "mock":
        from services.mock_analyzer import MockAnalyzer

        return MockAnalyzer()
    if backend == "api":
        from services.api_analyzer import ApiAnalyzer

        return ApiAnalyzer()
    raise ValueError(f"Unknown analyzer backend: {backend!r}")