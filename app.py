"""Fake Image Detector - Streamlit frontend entry point.

Run with:  streamlit run app.py

Renders the full UX state machine powered by the generic ``Analyzer``
contract. To connect a real backend later, set
``core.config.ANALYZER_BACKEND = "api"`` and implement
``services.api_analyzer.ApiAnalyzer`` - this file does not change.
"""

from __future__ import annotations

import streamlit as st

from components import _assets, analysis_panel, empty_state, header, uploader
from core import config, session
from core.analyzer import create_analyzer

st.set_page_config(
    page_title=f"{config.APP_NAME} · {config.APP_TAGLINE}",
    page_icon=str(config.LOGO_PATH),
    layout="wide",
    initial_sidebar_state="collapsed",
)

session.init()
_assets.render_css(config.STYLES_DIR / "main.css")

header.render()

_analyzer = create_analyzer(config.ANALYZER_BACKEND)

has_image = session.get_state("image_info") is not None

if not has_image:
    empty_state.render()
else:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        uploader.render_image_panel()
    with right:
        analysis_panel.render(_analyzer)

st.markdown(
    f'<div class="footer-note">{config.APP_NAME} · {config.APP_TAGLINE}</div>',
    unsafe_allow_html=True,
)