"""DEMO / TEMPORARY analyzer.

This is NOT a real AI model. It produces a plausible-looking outcome purely
so the full UI flow can be demonstrated without a backend. It is internally
labelled as demo infrastructure ("source": "demo") and the UI never exposes
these details to the end user.

Replace by implementing :class:`~services.api_analyzer.ApiAnalyzer` and
flipping ``config.ANALYZER_BACKEND`` to ``"api"``.
"""

from __future__ import annotations

import hashlib
import random
import time

from core.analyzer import Analyzer
from core.config import MOCK_ANALYSIS_DELAY_SECONDS
from core.models import AnalysisResult, ImageInfo, Verdict


class MockAnalyzer(Analyzer):
    """Deterministic demo analyzer.

    The same input bytes always yield the same verdict/confidence so the demo
    behaves consistently across reruns for a given image. The verdict is
    pseudo-random between images - it carries no analytical meaning.
    """

    def __init__(self, delay_range: tuple[float, float] | None = None) -> None:
        self._delay_range = delay_range or MOCK_ANALYSIS_DELAY_SECONDS

    @property
    def is_available(self) -> bool:
        return True

    def analyze(self, image_bytes: bytes, image_info: ImageInfo | None = None) -> AnalysisResult:
        seed = hashlib.sha256(image_bytes).digest()
        rng = random.Random(seed)

        if self._delay_range:
            time.sleep(rng.uniform(*self._delay_range))

        p_real = rng.random()
        verdict = Verdict.REAL if p_real >= 0.5 else Verdict.FAKE
        # Demo confidence is always decisive-looking (55%..98%); a real model
        # would return its actual softmax / probability here.
        confidence = rng.uniform(0.55, 0.98)

        return AnalysisResult(
            verdict=verdict,
            confidence=confidence,
            metadata={"source": "demo", "provider": "MockAnalyzer"},
        )