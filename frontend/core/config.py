"""Application configuration and presentation-level constants.

Everything the UI needs to render sits here so it never leaks into the
analysis services. Swap values here to rebrand or retune the product.

Runtime infrastructure settings (the analysis backend URL and its timeout) also
live here, so there is exactly one place to repoint the app at a different
deployment. Each can be overridden by an environment variable for the same
reason the backend reads ``.env``; the defaults match a local ``uvicorn`` on
port 8000.
"""

from __future__ import annotations

import os
from pathlib import Path

from core.models import Verdict, format_confidence

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
STYLES_DIR = BASE_DIR / "styles"
LOGO_PATH = ASSETS_DIR / "logo.png"

# --- Product identity -----------------------------------------------------
APP_NAME = "Fake Image Detector"
APP_TAGLINE = "AI AUTHENTICITY LAB"
APP_DESCRIPTION = (
    "Upload an image and the system will classify it as Real or Fake with a "
    "confidence score."
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
# "api"  = the real FastAPI backend (default now that it exists).
ANALYZER_BACKEND = os.getenv("FID_ANALYZER_BACKEND", "api").strip().lower()

# Base URL of the backend API. No trailing slash is required.
# Repoint at another deployment with FID_API_BASE_URL; no code change needed.
API_BASE_URL = os.getenv("FID_API_BASE_URL", "http://localhost:8000").strip().rstrip("/")

# Path of the analysis endpoint, appended to API_BASE_URL by the analyzer.
API_ANALYZE_PATH = "/analyze"

# Seconds to wait for the backend before giving up. Generous because the first
# request may also pay the model's one-time load cost.
API_TIMEOUT_SECONDS = float(os.getenv("FID_API_TIMEOUT_SECONDS", "60"))

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
    """Neutral result wording.

    The current analyzer only returns a verdict + confidence, so the copy
    states exactly that - it must never imply forensic signals or specific
    ML analyses that did not happen.
    """
    pct = format_confidence(confidence)
    return (
        f"The image was classified as {verdict.label} with {pct} confidence. "
        "Confidence indicates how strongly the analysis supports this "
        "classification."
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
    "Real / Fake verdict",
    "Confidence score",
    "Upload & analyze",
    "Instant results",
)
EMPTY_PRIVACY_NOTE = (
    "Images are used only for this analysis session. Nothing is stored or "
    "shared by this interface."
)
UPLOAD_HINT = "JPG · PNG · WEBP · GIF · BMP — up to 15 MB"

HOW_IT_WORKS_STEPS = (
    ("Upload", "Add an image for analysis."),
    ("Analyze", "The system evaluates the uploaded image."),
    ("Verify", "Receive a Real/Fake classification and confidence score."),
)