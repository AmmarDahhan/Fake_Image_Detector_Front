"""Widget callbacks.

Streamlit requires callbacks to be module-level functions. They only mutate
session state - never widget keys directly, and never business logic. The
heavy lifting (validation / analysis) happens elsewhere.
"""

from __future__ import annotations

import streamlit as st

from core import config, session
from core.validation import validate_image, ImageValidationError

FILE_UPLOADER_KEY = "file_uploader"


def on_upload_change() -> None:
    """Handle a new/removed file from the uploader widget."""

    file = st.session_state.get(FILE_UPLOADER_KEY)
    if file is None:
        session.reset_image()
        return

    data = file.getvalue()
    mime = getattr(file, "type", "") or None
    try:
        info = validate_image(
            file.name,
            data,
            mime_type=mime,
            max_bytes=config.MAX_UPLOAD_BYTES,
            allowed_extensions=config.ALLOWED_IMAGE_EXTENSIONS,
            allowed_mime_types=config.ALLOWED_MIME_TYPES,
        )
    except ImageValidationError as err:
        session.reset_image()
        session.set_error(
            code=err.code,
            title=config.VALIDATION_ERROR_TITLES.get(err.code, "Invalid file"),
            detail=err.message,
        )
        return

    st.session_state["file_bytes"] = data
    st.session_state["image_info"] = info
    session.clear_analysis()
    st.session_state["error"] = None


def on_analyze_click() -> None:
    """Begin an analysis for the currently selected image."""

    if st.session_state.get("is_analyzing"):
        return
    session.clear_analysis()
    st.session_state["is_analyzing"] = True
    st.session_state["analysis_started"] = False


def on_remove_image() -> None:
    """Start over: clear image + analysis and reset the uploader widget."""

    st.session_state.pop(FILE_UPLOADER_KEY, None)
    session.reset_image()
    st.rerun()


def on_dismiss_error() -> None:
    """Dismiss the current error and stay on the current screen."""

    st.session_state["error"] = None