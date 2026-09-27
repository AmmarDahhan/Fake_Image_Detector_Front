"""Right-hand analysis panel - the UI state machine.

One of:
  - informational "ready" panel while an image is selected
  - animated analyzing state
  - dominant result card
  - user-facing error card

Analysis itself is invoked here via the generic ``Analyzer`` contract:
the UI never knows whether the result came from the demo mock or a future
real backend.
"""

from __future__ import annotations

import logging

import streamlit as st

from components import analyzing_state, error_state, result_card
from core import session
from core.analyzer import Analyzer, AnalyzerUnavailableError
from core.models import AnalysisError

logger = logging.getLogger(__name__)


def render(analyzer: Analyzer) -> None:
    if session.get_state("error") is not None:
        error_state.render(session.get_state("error"))
        return

    if session.get_state("analysis") is not None:
        result_card.render(session.get_state("analysis"))
        return

    if session.get_state("is_analyzing"):
        if not session.get_state("analysis_started"):
            _run_analysis(analyzer)
        else:
            analyzing_state.render()
        return

    _render_ready()


def _run_analysis(analyzer: Analyzer) -> None:
    # Mark first, then render the animated state so it streams to the
    # browser while the (demo) delay runs.
    st.session_state["analysis_started"] = True
    analyzing_state.render()

    file_bytes = session.get_state("file_bytes")
    image_info = session.get_state("image_info")

    try:
        result = analyzer.analyze(file_bytes, image_info)
    except AnalyzerUnavailableError as exc:
        logger.warning("Analyzer unavailable: %s", exc)
        st.session_state["is_analyzing"] = False
        session.set_error(
            code="backend_unavailable",
            title="Analysis service unavailable",
            detail=(
                "The analysis engine is not reachable right now. "
                "Please try again shortly."
            ),
        )
        st.rerun()
        return
    except Exception as exc:  # noqa: BLE001 - keep tracebacks off the UI
        logger.exception("Analysis failed: %s", exc)
        st.session_state["is_analyzing"] = False
        session.set_error(
            code="analysis_failed",
            title="Analysis could not be completed",
            detail=(
                "The analysis engine did not finish processing this image. "
                "Please try again."
            ),
        )
        st.rerun()
        return

    st.session_state["analysis"] = result
    st.session_state["is_analyzing"] = False
    st.rerun()


def _render_ready() -> None:
    st.markdown(
        f"""
        <div class="card">
          <div class="status-chip"><span class="dot"></span>Image ready</div>
          <div style="margin-top:16px">
            <div class="result-kicker">Awaiting analysis</div>
            <div class="scan-sub" style="margin-top:8px">
              Your image is staged and validated. Press
              <b>Analyze Image</b> to run the authenticity assessment.
            </div>
          </div>
          <ul class="ready-list">
            <li><span class="r-dot"></span>Authenticity verdict</li>
            <li><span class="r-dot"></span>Confidence percentage</li>
            <li><span class="r-dot"></span>Concise interpretation</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )