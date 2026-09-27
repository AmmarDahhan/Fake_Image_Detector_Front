"""FastAPI application entry point.

Run from the ``backend/`` directory::

    uvicorn app.main:app --reload --port 8000

``API_HOST`` / ``API_PORT`` in ``.env`` are honoured by the ``__main__``
block below; the ``uvicorn`` CLI takes its bind address from its own flags.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

API_DESCRIPTION = (
    "Backend API for Fake Image Detector.\n\n"
    "`POST /analyze` validates an uploaded image and classifies it with the "
    "AI team's trained ConvNeXt model, returning a binary verdict and a "
    "confidence in 0.0-1.0. `GET /health` confirms the API itself is running.\n\n"
    "The model classifies into `full_synthetic` / `real` / `tampered`. "
    "Collapsing those three onto the binary `real`/`fake` contract is "
    "**provisional** and pending confirmation from the AI team; see "
    "`app/services/model_class_map.py`. The endpoint answers 503 if the model "
    "cannot be loaded."
)

app = FastAPI(
    title="Fake Image Detector API",
    description=API_DESCRIPTION,
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
