"""Application configuration and presentation-level constants.

Everything the UI needs to render sits here so it never leaks into the
analysis services. Swap values here to rebrand or retune the product.
"""

from __future__ import annotations

from pathlib import Path

from core.models import Verdict, format_confidence

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
STYLES_DIR = BASE_DIR / "styles"
LOGO_PATH = ASSETS_DIR / "logo.png"

# --- Product identity -----------------------------------------------------
APP_NAME = "Fake Image Detector"
APP_TAGLINE = "AI-Powered Image Authenticity Analysis"
APP_DESCRIPTION = (
    "Upload an image and the analysis engine will assess whether it is an "
    "authentic capture or was generated / manipulated by artificial "
    "intelligence."
)

# --- Upload constraints ---------------------------------------------------
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}

# --- Analysis backend -----------------------------------------------------
# "mock" = built-in demo analyzer (clearly NOT a real model).
# "api"  = future real backend; switch to it once the API contract exists.
ANALYZER_BACKEND = "mock"

# Demo-only simulated latency range (seconds). Ignored by the real backend.
MOCK_ANALYSIS_DELAY_SECONDS = (1.1, 2.0)

# --- Verdict palette ------------------------------------------------------
COLOR_REAL = "#34D399"
COLOR_REAL_TEXT = "#A7F3D0"
COLOR_FAKE = "#F87171"
COLOR_FAKE_TEXT = "#FECACA"
COLOR_ACCENT = "#2ED8C6"
COLOR_ACCENT_AMBER = "#F5B942"

# --- Copy: result interpretation ----------------------------------------
def interpret_result(verdict: Verdict, confidence: float) -> str:
    pct = format_confidence(confidence)
    if verdict is Verdict.REAL:
        return (
            f"No significant indicators of AI generation or manipulation were "
            f"found. The image is assessed as authentic (Real) at {pct} "
            f"confidence."
        )
    return (
        f"Strong indicators of AI generation or artificial manipulation were "
        f"found. The image is assessed as Fake at {pct} confidence."
    )


# --- Copy: validation / error states --------------------------------------
VALIDATION_ERROR_TITLES = {
    "empty": "No image selected",
    "unsupported_file": "File type not supported",
    "oversized": "Image too large",
    "invalid_image": "Unreadable image",
}

# --- Copy: empty / welcome state ------------------------------------------
EMPTY_FEATURES = (
    "Authenticity verdict",
    "Confidence scoring",
    "Image-level forensics",
    "Instant results",
)
EMPTY_PRIVACY_NOTE = (
    "Images are used only for this analysis session. Nothing is stored or "
    "shared by this interface."
)
UPLOAD_HINT = "JPG, PNG, WEBP, GIF, BMP — up to 15 MB"