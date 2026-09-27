"""Centralized configuration.

Every runtime knob the backend reads lives here, sourced from environment
variables (optionally seeded from a local ``backend/.env`` file - copy
``.env.example`` to ``.env`` to use it).

Model location is expressed as a path *relative to the repository* so the
project stays portable and no machine-specific absolute path is baked into the
code. ``MODEL_PATH`` overrides it for deployments that keep the checkpoint
somewhere else; the default points at the reconstructed checkpoint produced by
``backend/tools/prepare_checkpoint.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")

# Where the loadable checkpoint lives by default.
#
# The AI team shipped ai_model/best_accuracy_model/, which is an *extracted*
# PyTorch archive and not loadable by torch.load (which requires a single
# seekable file, with its members under a top-level directory). Running
#     python tools/prepare_checkpoint.py
# re-zips it byte-for-byte into the file below, leaving ai_model/ untouched.
DEFAULT_MODEL_DIR = BACKEND_DIR / "artifacts"
DEFAULT_MODEL_FILENAME = "best_accuracy_model.pth"

# Mirrors the frontend's accepted formats (frontend/core/config.py).
# The backend revalidates independently: a client is never trusted.
DEFAULT_ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "gif", "bmp")

DEFAULT_ALLOWED_MIME_TYPES = (
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
)

#: Device names the model layer will honour from ``MODEL_DEVICE``. Left as
#: ``None`` the service auto-detects (CUDA when present, else CPU).
VALID_MODEL_DEVICES = ("cpu", "cuda")



def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_optional(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


class Settings(BaseModel):
    """Backend settings.

    A plain mutable model so tests can override individual limits with
    ``monkeypatch.setattr(settings, "max_upload_bytes", ...)``.
    """

    # --- Server ---
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)

    # --- Upload validation ---
    # 15 MiB, matching the frontend's MAX_UPLOAD_BYTES and Streamlit's
    # .streamlit/config.toml maxUploadSize = 15.
    max_upload_bytes: int = Field(default=15 * 1024 * 1024)
    allowed_extensions: tuple[str, ...] = Field(
        default=DEFAULT_ALLOWED_EXTENSIONS
    )
    allowed_mime_types: tuple[str, ...] = Field(default=DEFAULT_ALLOWED_MIME_TYPES)

    # --- Model ---
    # MODEL_PATH: absolute or relative path to the checkpoint archive. When
    # unset the project-relative default above is used.
    model_path: str | None = None
    # MODEL_DEVICE: "cpu" or "cuda". When unset, auto-detected.
    model_device: str | None = None

    @property
    def resolved_model_path(self) -> Path:
        """Absolute path of the checkpoint archive to load.

        An explicit ``MODEL_PATH`` wins; otherwise the project-relative
        default. Resolved so a relative override is interpreted against the
        process working directory rather than silently matching nothing.
        """
        if self.model_path:
            return Path(self.model_path).expanduser().resolve()
        return DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME

    @field_validator("model_device", mode="before")
    @classmethod
    def _validate_device(cls, value: object) -> object:
        if value is None:
            return None
        device = str(value).strip().lower()
        return device or None

    @field_validator("allowed_extensions", "allowed_mime_types", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept either a JSON-ish list or a comma-separated env string."""
        if isinstance(value, str):
            return tuple(
                part.strip().lower()
                for part in value.split(",")
                if part.strip()
            )
        return value

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables.

        Upload size is expressed in megabytes here (``MAX_UPLOAD_MB``) because
        that is how the limit is specified on both sides of the project.
        """
        max_upload_mb = _env_int("MAX_UPLOAD_MB", 15)

        kwargs: dict[str, object] = {
            "api_host": os.getenv("API_HOST", "127.0.0.1"),
            "api_port": _env_int("API_PORT", 8000),
            "max_upload_bytes": max_upload_mb * 1024 * 1024,
            "model_path": _env_optional("MODEL_PATH"),
            "model_device": _env_optional("MODEL_DEVICE"),
        }

        for env_name, field_name in (
            ("ALLOWED_EXTENSIONS", "allowed_extensions"),
            ("ALLOWED_MIME_TYPES", "allowed_mime_types"),
        ):
            raw = _env_optional(env_name)
            if raw is not None:
                kwargs[field_name] = raw

        return cls(**kwargs)


settings = Settings.from_env()
