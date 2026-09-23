"""Upload validation.

Kept free of Streamlit imports so it can be unit-tested in isolation.
Produces an :class:`~core.models.ImageInfo` or raises an
:class:`ImageValidationError` carrying a stable error ``code`` that the UI
maps to a friendly error state.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from core.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_UPLOAD_BYTES,
)
from core.models import ImageInfo


@dataclass(frozen=True)
class ImageValidationError(Exception):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


def validate_image(
    filename: str,
    data: bytes,
    mime_type: str | None = None,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    allowed_extensions: set[str] = ALLOWED_IMAGE_EXTENSIONS,
    allowed_mime_types: set[str] = ALLOWED_MIME_TYPES,
) -> ImageInfo:
    from PIL import Image

    if not filename or not data:
        raise ImageValidationError("empty", "No image was selected.")

    if len(data) > max_bytes:
        from core.models import format_bytes

        raise ImageValidationError(
            "oversized",
            f"This image is {format_bytes(len(data))}, which is larger than the "
            f"{format_bytes(max_bytes)} upload limit. Please choose a smaller "
            "or compressed image.",
        )

    extension = Path(filename).suffix.lstrip(".").lower()
    if extension not in allowed_extensions and (
        not mime_type or mime_type.lower() not in allowed_mime_types
    ):
        raise ImageValidationError(
            "unsupported_file",
            f"File type '{extension or 'unknown'}' is not supported. Use "
            f"{', '.join(sorted(allowed_extensions)).upper()}.",
        )

    try:
        with Image.open(BytesIO(data)) as img:
            format_name = (img.format or extension).upper()
            width, height = img.size
        if width <= 0 or height <= 0:
            raise ValueError("empty dimensions")
    except Exception as exc:
        raise ImageValidationError(
            "invalid_image",
            "The file could not be read as an image. It may be corrupted or "
            "in an unsupported encoding.",
        ) from exc

    return ImageInfo(
        filename=filename,
        mime_type=(mime_type or "").lower(),
        size_bytes=len(data),
        width=width,
        height=height,
        format_name=format_name,
    )