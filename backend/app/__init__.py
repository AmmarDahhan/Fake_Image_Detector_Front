"""Fake Image Detector backend.

A FastAPI service that exposes the analysis API consumed by the Streamlit
frontend in ``../frontend``.

Scope of this initial scaffold: API infrastructure only. The inference layer
is deliberately not implemented - see ``app.services.model_service``.
"""

__version__ = "0.1.0"
