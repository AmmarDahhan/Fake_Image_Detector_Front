"""Dominant result card: verdict, confidence meter, interpretation, image.

Depends only on the ``AnalysisResult`` abstraction - never on which analyzer
produced it - so a future backend result renders identically.
"""

from __future__ import annotations

import html

import streamlit as st

from components import events
from core import config, session
from core.models import AnalysisResult, Verdict, format_confidence


def render(result: AnalysisResult) -> None:
    verdict_class = result.verdict.css_class
    percentage = format_confidence(result.confidence)

    st.markdown(
        f"""
        <div class="card">
          <div class="result-kicker">Analysis result</div>
          <div>
            <div class="verdict-badge {verdict_class}">
              <span class="dot"></span>
              {html.escape(result.verdict.label)}
            </div>
          </div>

          <div class="confidence-block">
            <div class="confidence-value">{percentage}</div>
            <div class="confidence-scale">
              <div class="confidence-fill {verdict_class}"
                   style="width:{percentage}"></div>
            </div>
            <div class="confidence-caption">Confidence</div>
          </div>

          <div class="interpretation">
            {html.escape(config.interpret_result(result.verdict, result.confidence))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    file_bytes = session.get_state("file_bytes")
    if file_bytes is not None:
        st.markdown(
            """
            <div class="card" >
              <span class="card-label">Analyzed sample</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(file_bytes, width="stretch")

    st.button(
        "Analyze another image",
        key="reset_btn",
        type="secondary",
        on_click=events.on_remove_image,
    )