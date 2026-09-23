"""In-progress analysis state with custom scan animation.

Copy is deliberately neutral - it must not claim specific model operations
(such as face detection or artifact extraction) that do not actually occur.
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown(
        """
        <div class="card">
          <div class="scan-wrap">
            <div class="scan-ring"></div>
            <div class="scan-line"></div>
            <div class="scan-title">Analysis in progress</div>
            <div class="scan-sub">Analyzing image…</div>
            <div class="scan-hint">This typically takes a few seconds.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )