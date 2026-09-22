"""Small asset/IO helpers shared by UI components."""

from __future__ import annotations

import base64
from pathlib import Path


def asset_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def render_css(style_path: Path) -> None:
    import streamlit as st

    css = style_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)