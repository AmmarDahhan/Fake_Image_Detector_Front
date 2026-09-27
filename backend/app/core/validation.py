"""Upload validation.

Deliberately model-free: these checks must hold whether or not an inference
implementation exists, and they must be enforceable on a server that does
not trust its clients.

Checks performed, in order:

1. declared format (extension or MIME type) - client-declarable
2. size limit                              - client-declarable
3. decodability                            - verified from the actual bytes

(1) and (2) are trivially spoofed: a text file renamed to ``.png`` and sent
with ``Content-Type: image/png`` satisfies both. (3) is therefore the only
check that establishes the payload is really an image.

Pillow is used solely to decode and measure the image so it can be proven
decodable. No preprocessing, resizing, colour conversion, or inference
happens here - that belongs to the model layer.

Limits mirror the frontend (``frontend/core/config.py``) so a file that the
UI accepts is also accepted by the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath

from PIL import Image, UnidentifiedImageError

from app.core.config import settings

# Read in bounded chunks so an oversized upload is rejected without ever
# buffering the whole body in memory.
_CHUNK_SIZE = 1024 * 1024


class UploadRejected(Exception):
    """An upload failed validation.

    ``status_code`` is the HTTP status the API should answer with.
    """

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _extension_of(filename: str | None) -> str:
    if not filename:
        return ""
    return PurePath(filename).suffix.lstrip(".").lower()


def check_filename(filename: str | None, content_type: str | None) -> None:
    """Reject uploads whose declared type is not an accepted image format.

    Mirrors the frontend's rule: a file passes when either its extension or
    its declared MIME type is in the accepted set.
    """
    extension = _extension_of(filename)
    mime = (content_type or "").split(";")[0].strip().lower()

    extension_ok = extension in settings.allowed_extensions
    mime_ok = mime in settings.allowed_mime_types

    if extension_ok or mime_ok:
        return

    accepted = ", ".join(sorted(settings.allowed_extensions)).upper()
    described = extension or mime or "unknown"
    raise UploadRejected(
        f"Unsupported file type '{described}'. Accepted types: {accepted}."
    )


async def read_validated(file, filename: str | None, content_type: str | None) -> bytes:
    """Validate an uploaded file and return its bytes.

    Args:
        file: a Starlette ``UploadFile``.
        filename: the client-declared filename.
        content_type: the client-declared MIME type.

    Returns:
        The uploaded bytes.

    Raises:
        UploadRejected: empty file, unsupported type, over the size limit, or
            bytes that are not a decodable image.
    """
    check_filename(filename, content_type)

    data = await _read_capped(file)
    if not data:
        raise UploadRejected("No image was uploaded.")

    # Last check: the size limit is enforced first so an oversized upload is
    # never handed to a decoder.
    verify_decodable(data)

    return data


def verify_decodable(data: bytes) -> None:
    """Reject bytes that are not a real, decodable image.

    ``Image.open`` alone only parses a header, so a truncated or garbage
    payload can pass it. ``load()`` forces the full pixel decode, which is
    what makes the image provably readable. Pillow also enforces its own
    decompression-bomb ceiling during that decode.

    Raises:
        UploadRejected: the bytes are not a decodable image.
    """
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        # Same wording the frontend uses for its "invalid_image" state, so the
        # user-facing copy stays consistent across both services.
        raise UploadRejected(
            "The file could not be read as an image. It may be corrupted or "
            "in an unsupported encoding."
        ) from exc

    if width <= 0 or height <= 0:
        raise UploadRejected(
            "The file could not be read as an image. It may be corrupted or "
            "in an unsupported encoding."
        )


async def _read_capped(file) -> bytes:
    """Read at most ``settings.max_upload_bytes`` + 1 bytes, then stop.

    Stopping one byte past the limit is enough to detect an oversized upload
    without reading the remainder.
    """
    limit = settings.max_upload_bytes
    buffer = bytearray()
    while len(buffer) <= limit:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        buffer.extend(chunk)

    if len(buffer) > limit:
        raise UploadRejected(
            f"Image is larger than the {_format_mb(limit)} upload limit.",
            status_code=413,
        )

    return bytes(buffer)


def _format_mb(limit_bytes: int) -> str:
    return f"{limit_bytes / (1024 * 1024):.0f} MB"
