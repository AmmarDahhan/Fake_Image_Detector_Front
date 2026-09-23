"""Welcome / empty state shown before any image is selected."""

from __future__ import annotations

import html

import streamlit as st

from components import error_state, uploader
from components._assets import asset_to_data_uri
from core import config, session


def render() -> None:
    logo_uri = asset_to_data_uri(config.LOGO_PATH)
    features = "".join(
        f'<span class="feature-pill"><span class="fp"></span>{html.escape(f)}</span>'
        for f in config.EMPTY_FEATURES
    )
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-mark"><img src="{logo_uri}" alt=""/></div>
          <h1 class="hero-title">Detect <span class="hl">AI-generated</span>
          {"&"} manipulated images</h1>
          <p class="hero-sub">{html.escape(config.APP_DESCRIPTION)}</p>
          <div class="feature-row">{features}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="drop-margin">', unsafe_allow_html=True)
        uploader.render_upload_area()
        st.markdown(
            f'<div class="upload-hint">{html.escape(config.UPLOAD_HINT)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if session.get_state("error") is not None:
        st.markdown('<div style="margin-top: 12px"></div>', unsafe_allow_html=True)
        error_state.render(session.get_state("error"))

    st.markdown(_steps_html(), unsafe_allow_html=True)


def _steps_html() -> str:
    steps = "".join(
        f'<div class="step">'
        f'<div class="step-num">{html.escape(num)}</div>'
        f'<div class="step-body">'
        f'<div class="step-title">{html.escape(title)}</div>'
        f'<div class="step-text">{html.escape(text)}</div>'
        f"</div>"
        f"</div>"
        for num, (title, text) in zip(("01", "02", "03"), config.HOW_IT_WORKS_STEPS)
    )
    return (
        '<div class="how-wrap"><div class="how-kicker">How it works</div>'
        f'<div class="steps">{steps}</div></div>'
    )