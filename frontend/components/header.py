"""Product header / brand bar."""

from __future__ import annotations

import streamlit as st

from components._assets import asset_to_data_uri
from core import config

from html import escape


def render() -> None:
    logo_uri = asset_to_data_uri(config.LOGO_PATH)
    html = f"""
    <div class="brand-row">
      <div class="brand-mark"><img src="{logo_uri}" alt=""/></div>
      <div>
        <div class="brand-name">{escape(config.APP_NAME)}</div>
        <div class="brand-tagline">{escape(config.APP_TAGLINE)}</div>
      </div>
    </div>
    <hr class="brand-rule"/>
    """
    st.markdown(html, unsafe_allow_html=True)