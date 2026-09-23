"""Dominant result card: verdict, confidence, interpretation, file details.

Depends only on the ``AnalysisResult`` abstraction - never on which analyzer
produced it - so a future backend result renders identically. The uploaded
image is shown once (in the left panel); the card carries no second image.
"""

from __future__ import annotations

import html

import streamlit as st

from components import events
from core import config, session
from core.models import AnalysisResult, format_confidence


def render(result: AnalysisResult) -> None:
    verdict_class = result.verdict.css_class
    percentage = format_confidence(result.confidence)

    st.markdown(
        f"""
        <div class="card result-hero">
          <div class="result-kicker">Analysis result</div>
          <div class="verdict-badge {verdict_class}">
            <span class="dot"></span>
            {html.escape(result.verdict.label)}
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

    st.markdown(_details_html(result), unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 14px"></div>', unsafe_allow_html=True)

    st.button(
        "Analyze another image",
        key="reset_btn",
        type="secondary",
        on_click=events.on_remove_image,
    )


def _details_html(result: AnalysisResult) -> str:
    info = session.get_state("image_info")
    rows = [
        ("Verdict", result.verdict.label),
        ("Confidence", format_confidence(result.confidence)),
        ("File type", info.type_label if info else "—"),
        ("Dimensions", info.dimensions_label if info else "—"),
        ("File size", info.size_label if info else "—"),
        ("Status", "Completed"),
    ]
    body = "".join(
        f'<div class="details-row">'
        f'<span class="d-k">{html.escape(k)}</span>'
        f'<span class="d-v">{html.escape(v)}</span>'
        f"</div>"
        for k, v in rows
    )
    return (
        '<div class="card details-card">'
        '<div class="result-kicker">Analysis details</div>'
        f'<div class="details-list">{body}</div></div>'
    )