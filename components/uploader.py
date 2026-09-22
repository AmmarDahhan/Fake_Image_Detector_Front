"""Image uploader widget + the image inspection panel (left column).

Renders the dropzone, the preview with metadata, and the prominent
"Analyze" action. All state transitions are driven by callbacks in
``components.events``.
"""

from __future__ import annotations

import html

import streamlit as st

from components import events
from core import config, session
from core.models import ImageInfo


def render_upload_area() -> None:
    st.file_uploader(
        "Upload an image",
        key=events.FILE_UPLOADER_KEY,
        type=sorted(config.ALLOWED_IMAGE_EXTENSIONS),
        accept_multiple_files=False,
        on_change=events.on_upload_change,
        help="JPG, PNG, WEBP, GIF or BMP — up to 15 MB",
    )


def render_image_panel() -> None:
    file_bytes = session.get_state("file_bytes")
    info: ImageInfo = session.get_state("image_info")
    is_analyzing = bool(session.get_state("is_analyzing"))

    st.markdown(
        f"""
        <div class="card">
          <div class="panel-head">
            <span class="card-label">Uploaded image</span>
            <span class="type-chip">TYPE · {html.escape(info.type_label)}</span>
          </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(file_bytes, width="stretch")

    st.markdown(_meta_grid(info), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="action-row">
          <div>
        """,
        unsafe_allow_html=True,
    )
    st.button(
        "Analyze Image",
        key="analyze_btn",
        type="primary",
        disabled=is_analyzing,
        on_click=events.on_analyze_click,
    )
    st.markdown(
        """
          </div>
          <div class="grow-small">
        """,
        unsafe_allow_html=True,
    )
    st.button(
        "Remove",
        key="remove_btn",
        type="tertiary",
        on_click=events.on_remove_image,
    )
    st.markdown(
        """
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def _meta_grid(info: ImageInfo) -> str:
    cells = [
        ("Filename", info.filename),
        ("Type", info.type_label),
        ("Dimensions", info.dimensions_label),
        ("Size", info.size_label),
    ]
    body = "".join(
        f'<div class="meta-cell"><div class="meta-k">{html.escape(k)}</div>'
        f'<div class="meta-v">{html.escape(v)}</div></div>'
        for k, v in cells
    )
    return f'<div class="meta-grid">{body}</div>'