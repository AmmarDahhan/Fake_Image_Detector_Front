"""In-progress analysis state with custom scan animation."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown(
        """
        <div class="card">
          <div class="scan-wrap">
            <div class="scan-ring"></div>
            <div class="scan-line"></div>
            <div class="scan-title">Running background analysis</div>
            <div class="scan-sub">Examining the image for authenticity signals…</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )