"""Session-state helpers.

The UI keeps its small amount of cross-rerun state here using well-defined
keys and setters, so components never write ad-hoc values. We deliberately
do NOT assign values to widget keys (file_uploader / buttons) - widget
driven state stays managed by Streamlit via callbacks.
"""

from __future__ import annotations

import streamlit as st

from core.models import AnalysisError

_ANALYSIS_KEYS = ("analysis", "error", "is_analyzing", "analysis_started")


def init() -> None:
    st.session_state.setdefault("file_bytes", None)
    st.session_state.setdefault("image_info", None)
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("is_analyzing", False)
    st.session_state.setdefault("analysis_started", False)


def clear_analysis() -> None:
    """Reset the analysis outcome while keeping the uploaded image."""

    for key in _ANALYSIS_KEYS:
        st.session_state[key] = None
    st.session_state["is_analyzing"] = False
    st.session_state["analysis_started"] = False


def reset_image() -> None:
    """Drop the uploaded image and its analysis state entirely."""

    st.session_state["file_bytes"] = None
    st.session_state["image_info"] = None
    clear_analysis()


def get_state(key: str):
    return st.session_state.get(key)


def set_error(code: str, title: str, detail: str) -> None:
    st.session_state["error"] = AnalysisError(code=code, title=title, detail=detail)