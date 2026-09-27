"""Core domain models shared by the UI, analyzers, and the future API layer.

The UI must depend only on these abstractions - never on analyzer-specific
fields. A real backend response will be mapped onto these same types later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Binary authenticity outcome returned by an analyzer.

    Reflects the specification's required outcome (Real / Fake). The enum
    shape makes it trivial to extend with additional classes later without
    touching the UI.
    """

    REAL = "real"
    FAKE = "fake"

    @property
    def label(self) -> str:
        return "Real" if self is Verdict.REAL else "Fake"

    @property
    def css_class(self) -> str:
        return "real" if self is Verdict.REAL else "fake"

    @classmethod
    def parse(cls, raw: str) -> "Verdict":
        try:
            return cls(raw.strip().lower())
        except (ValueError, AttributeError):
            raise ValueError(f"Unknown verdict: {raw!r}") from None


@dataclass(frozen=True)
class ImageInfo:
    """Validated facts about the uploaded image, derived at upload time."""

    filename: str
    mime_type: str
    size_bytes: int
    width: int
    height: int
    format_name: str

    @property
    def dimensions_label(self) -> str:
        return f"{self.width} × {self.height} px"

    @property
    def size_label(self) -> str:
        return format_bytes(self.size_bytes)

    @property
    def type_label(self) -> str:
        return (self.format_name or self.mime_type.split("/")[-1]).upper()


@dataclass
class AnalysisResult:
    """Result of an authenticity analysis.

    `verdict` and `confidence` are the only guaranteed fields - they match the
    specification's output. `message` and `metadata` are optional and unused
    by the UI; a future backend may populate them without UI changes.
    """

    verdict: Verdict
    confidence: float
    message: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, Verdict):
            raise TypeError("verdict must be a Verdict")
        if not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be a number")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def confidence_percent(self) -> float:
        return self.confidence * 100.0


@dataclass(frozen=True)
class AnalysisError:
    """User-facing error surfaced in the UI (never a Python traceback)."""

    code: str
    title: str
    detail: str


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    value = float(size)
    for unit in ("KB", "MB", "GB"):
        value /= 1024.0
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TB"


def format_confidence(confidence: float, decimals: int = 1) -> str:
    """Format a 0..1 confidence as a human-readable percentage string."""
    if not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be a number")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return f"{confidence * 100.0:.{decimals}f}%"