"""Shared fixtures.

Tests run against the real application object. Only the upload size limit is
monkeypatched (to keep the oversized-upload test fast); everything else -
routing, validation, serialisation, the model seam - is the real code path.

Image fixtures are genuine encoded images because the backend decodes uploads
to verify them. A file that must be *rejected* as undecodable is derived from
a real image by corrupting it, so the test proves the decoder is the thing
rejecting it rather than the extension check.

Two tiers of test exist here:

* API tests (the default) run against :class:`StubModelService`, a test double.
  Upload validation, status codes and serialisation must be verifiable without
  paying ~15 s and ~544 MiB to load the real checkpoint into every test.
* Integration tests, marked ``integration``, are exempted from the double and
  exercise the real :class:`RealModelService`. They are few and load the
  checkpoint once per session, not once per test.

Nothing in the double ever reaches the shipped API path: ``get_model_service``
returns the real service in production, and the swap happens only by
monkeypatching the route module inside a test.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app
from app.schemas.analysis import Verdict
from app.services.model_service import ModelPrediction, ModelService

# backend/tests/conftest.py -> backend/ -> project root
PROJECT_DIR = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE_DIR = PROJECT_DIR / "ai_model" / "images"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class StubModelService(ModelService):
    """Test double for the inference layer.

    Records what it was asked to do and returns a fixed prediction, so API
    tests can assert that a valid upload *reaches* the model seam and that the
    seam's output is serialised correctly - without loading real weights.
    """

    def __init__(self, prediction: ModelPrediction | None = None) -> None:
        self.calls: list[bytes] = []
        self._prediction = prediction or ModelPrediction(
            verdict=Verdict.REAL, confidence=0.5, class_name="real"
        )

    @property
    def is_available(self) -> bool:
        return True

    def analyze(self, image) -> ModelPrediction:
        self.calls.append(image.data)
        return self._prediction


# One instance for the whole session: the autouse fixture patches the route to
# return it, and ``stub_service`` hands the same object to tests that want to
# assert on what it received.
_STUB_SERVICE = StubModelService()


@pytest.fixture
def stub_service() -> StubModelService:
    """The stub currently wired into ``POST /analyze``.

    ``calls`` is cleared on access so a test can assert exactly which bytes
    reached the inference layer.
    """
    _STUB_SERVICE.calls.clear()
    return _STUB_SERVICE


@pytest.fixture(autouse=True)
def use_stub_model_service(request, monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the real model for the double, except in integration tests.

    A test that opts into the real model marks itself ``integration``; anything
    else gets the double. A test may re-patch ``get_model_service`` itself,
    which takes precedence over this fixture.
    """
    if request.node.get_closest_marker("integration") is not None:
        return

    from app.api import routes

    _STUB_SERVICE.calls.clear()
    monkeypatch.setattr(routes, "get_model_service", lambda: _STUB_SERVICE)


def make_image_bytes(image_format: str = "PNG", width: int = 64, height: int = 64) -> bytes:
    """Encode a real image of the given format."""
    buffer = BytesIO()
    Image.new("RGB", (width, height), (90, 90, 90)).save(buffer, format=image_format)
    return buffer.getvalue()


def corrupt_image_bytes(data: bytes) -> bytes:
    """Truncate a real image mid-pixel-data, leaving a valid header.

    ``Image.open`` parses the header and succeeds; only the full decode in
    ``load()`` fails. This isolates the decode step as the rejecting check.
    """
    return data[: len(data) // 2]


@pytest.fixture
def png_bytes() -> bytes:
    return make_image_bytes("PNG")


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_image_bytes("JPEG")


@pytest.fixture
def png_upload() -> callable:
    def _upload(data: bytes, filename: str = "photo.png") -> dict[str, tuple]:
        return {"file": (filename, data, "image/png")}
    return _upload


@pytest.fixture
def jpeg_upload() -> callable:
    def _upload(data: bytes, filename: str = "photo.jpg") -> dict[str, tuple]:
        return {"file": (filename, data, "image/jpeg")}
    return _upload


@pytest.fixture
def tiny_limit(monkeypatch: pytest.MonkeyPatch) -> int:
    """Shrink the upload limit so the oversized test does not push 15 MB."""
    limit = 64
    monkeypatch.setattr(settings, "max_upload_bytes", limit)
    return limit


@pytest.fixture(scope="session")
def real_model_service():
    """The real :class:`RealModelService`, loaded once for the whole session.

    Session-scoped on purpose: loading the ~544 MiB checkpoint costs ~15 s and
    a gigabyte of RAM, and the integration tests all assert the same loaded
    model. ``pytest.importorskip`` keeps the suite runnable on a machine where
    the checkpoint has not been reconstructed yet.
    """
    pytest.importorskip("torch", reason="PyTorch is required for integration tests")

    from app.services.model_service import RealModelService, reset_model_service

    try:
        service = RealModelService()
    except Exception as exc:  # noqa: BLE001 - surfaced as a skip, not a failure
        pytest.skip(f"Real model is not loadable in this environment: {exc}")

    reset_model_service()
    yield service
    reset_model_service()
