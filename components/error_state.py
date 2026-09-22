"""User-facing error card.

Shows a friendly title/detail mapped from an ``AnalysisError`` code. Never
renders Python tracebacks.
"""

from __future__ import annotations

import html

import streamlit as st

from components import events
from core.models import AnalysisError


def render(error: AnalysisError) -> None:
    st.markdown(
        f"""
        <div class="error-card" role="alert">
          <div class="error-kicker">Analysis failed</div>
          <div class="error-title">{html.escape(error.title)}</div>
          <div class="error-detail">{html.escape(error.detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button(
        "Dismiss",
        key="dismiss_error_btn",
        type="secondary",
        on_click=events.on_dismiss_error,
    )