"""Future backend integration point.

Database: intentionally NOT implemented and does NOT guess an API contract.

When the real backend exists, implement ``analyze()`` to:
  1. POST the image bytes to the real endpoint,
  2. parse the agreed response schema,
  3. map it onto :class:`~core.models.AnalysisResult`
     (at minimum ``verdict`` + ``confidence``).

The UI requires no changes - it only depends on the ``Analyzer`` contract.
"""

from __future__ import annotations

from core.analyzer import Analyzer, AnalyzerUnavailableError
from core.models import AnalysisResult, ImageInfo


class ApiAnalyzer(Analyzer):
    """Placeholder for the real backend. Raises until implemented."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = base_url
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        return False

    def analyze(self, image_bytes: bytes, image_info: ImageInfo | None = None) -> AnalysisResult:
        raise AnalyzerUnavailableError(
            "The backend analysis API is not available yet. Implement "
            "`services.api_analyzer.ApiAnalyzer.analyze()` once the API "
            "contract is agreed; the UI will keep working unchanged."
        )