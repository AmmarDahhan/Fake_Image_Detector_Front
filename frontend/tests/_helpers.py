"""Shared helpers for the test suite."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def make_png_bytes(w: int = 16, h: int = 16) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (w, h), (90, 90, 90)).save(buf, format="PNG")
    return buf.getvalue()