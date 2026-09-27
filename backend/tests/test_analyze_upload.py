"""Tests for POST /analyze request handling.

Covers upload validation (API infrastructure) and the request/response
contract around the model seam.

The model itself is not loaded here: ``conftest`` wires a stub service in, so
these tests assert what the API does with a model's answer. The real
inference path is covered separately in ``test_real_model.py`` under the
``integration`` marker.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.schemas.analysis import Verdict
from app.services.model_service import (
    ModelInferenceError,
    ModelPrediction,
    UnavailableModelService,
)
from tests.conftest import StubModelService, corrupt_image_bytes, make_image_bytes

# A non-image payload: plain text, no image container anywhere in it.
TEXT_PAYLOAD = b"this is a text file, definitely not an image" * 4


class TestMissingUpload:
    def test_no_file_field_is_rejected(self, client: TestClient) -> None:
        assert client.post("/analyze", files={}).status_code == 422

    def test_no_body_at_all_is_rejected(self, client: TestClient) -> None:
        assert client.post("/analyze").status_code == 422


class TestEmptyUpload:
    def test_zero_byte_file_is_rejected(self, client: TestClient, png_upload) -> None:
        response = client.post("/analyze", files=png_upload(b""))
        assert response.status_code == 400
        assert "no image" in response.json()["detail"].lower()


class TestUnsupportedFormat:
    @pytest.mark.parametrize("filename", ["notes.txt", "archive.zip", "data.json"])
    def test_non_image_extension_is_rejected(
        self, client: TestClient, filename: str
    ) -> None:
        response = client.post(
            "/analyze", files={"file": (filename, b"payload", "text/plain")}
        )
        assert response.status_code == 400
        assert "unsupported file type" in response.json()["detail"].lower()

    def test_no_extension_and_no_content_type_is_rejected(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/analyze", files={"file": ("mystery", b"payload", "application/octet-stream")}
        )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        ("filename", "content_type"),
        [
            ("photo.png", "image/png"),
            ("photo.jpg", "image/jpeg"),
            ("photo.jpeg", "image/jpeg"),
            ("photo.webp", "image/webp"),
            ("photo.gif", "image/gif"),
            ("photo.bmp", "image/bmp"),
        ],
    )
    def test_formats_aligned_with_frontend_pass_validation(
        self, client: TestClient, filename: str, content_type: str
    ) -> None:
        """Accepted formats reach the model seam, i.e. validation let them through.

        A real PNG is used for every case: the decode check reads the actual
        bytes, so a genuine image is what proves the format was accepted.
        200 (not 400) is the signal that validation passed.
        """
        response = client.post(
            "/analyze",
            files={"file": (filename, make_image_bytes("PNG"), content_type)},
        )
        assert response.status_code == 200, response.json()

    def test_extension_accepted_even_with_wrong_mime(
        self, client: TestClient, png_bytes: bytes
    ) -> None:
        """Mirrors the frontend rule: extension OR mime type may match."""
        response = client.post(
            "/analyze", files={"file": ("photo.png", png_bytes, "application/octet-stream")}
        )
        assert response.status_code == 200, response.json()


class TestDecodability:
    """Server-side decode verification.

    The extension and MIME checks are client-declarable, so a non-image can
    claim to be a .png. These tests cover the decode step that closes that gap.
    """

    def test_text_file_renamed_to_png_is_rejected(
        self, client: TestClient, png_upload
    ) -> None:
        """Text bytes with a .png name and image/png MIME must be refused."""
        response = client.post("/analyze", files=png_upload(TEXT_PAYLOAD))
        assert response.status_code == 400
        assert "could not be read as an image" in response.json()["detail"]

    def test_text_file_renamed_to_jpg_is_rejected(
        self, client: TestClient, jpeg_upload
    ) -> None:
        response = client.post("/analyze", files=jpeg_upload(TEXT_PAYLOAD))
        assert response.status_code == 400
        assert "could not be read as an image" in response.json()["detail"]

    def test_text_file_with_image_mime_type_is_still_rejected(
        self, client: TestClient
    ) -> None:
        """The declared MIME type is not evidence of image content."""
        response = client.post(
            "/analyze",
            files={"file": ("photo.png", TEXT_PAYLOAD, "image/png")},
        )
        assert response.status_code == 400

    def test_corrupted_image_bytes_are_rejected(
        self, client: TestClient, png_bytes: bytes, png_upload
    ) -> None:
        """A real PNG truncated mid-pixel-data has a valid header but will not decode."""
        response = client.post("/analyze", files=png_upload(corrupt_image_bytes(png_bytes)))
        assert response.status_code == 400
        assert "could not be read as an image" in response.json()["detail"]

    def test_garbage_after_valid_header_is_rejected(
        self, client: TestClient, png_upload
    ) -> None:
        """Correct PNG signature, garbage body - must not pass on the header alone."""
        response = client.post(
            "/analyze", files=png_upload(b"\x89PNG\r\n\x1a\n" + b"garbage" * 32)
        )
        assert response.status_code == 400

    def test_rejection_uses_the_same_structure_as_other_validation_errors(
        self, client: TestClient, png_upload
    ) -> None:
        """Same 400 + {"detail": ...} envelope as the other validation failures."""
        undecodable = client.post("/analyze", files=png_upload(TEXT_PAYLOAD))
        unsupported = client.post(
            "/analyze", files={"file": ("notes.txt", b"payload", "text/plain")}
        )
        assert undecodable.status_code == unsupported.status_code == 400
        assert set(undecodable.json()) == set(unsupported.json()) == {"detail"}

    def test_undecodable_upload_never_reaches_the_model_seam(
        self, client: TestClient, png_upload
    ) -> None:
        """Must be 400, not the 503 that the missing model would produce."""
        response = client.post("/analyze", files=png_upload(TEXT_PAYLOAD))
        assert response.status_code == 400
        assert "verdict" not in response.json()
        assert "confidence" not in response.json()


class TestValidImagesStillPass:
    """Regression guard: the decode check must not reject good uploads."""

    @pytest.mark.parametrize(
        ("image_format", "filename", "content_type"),
        [
            ("PNG", "photo.png", "image/png"),
            ("JPEG", "photo.jpg", "image/jpeg"),
            ("JPEG", "photo.jpeg", "image/jpeg"),
            ("WEBP", "photo.webp", "image/webp"),
            ("GIF", "photo.gif", "image/gif"),
            ("BMP", "photo.bmp", "image/bmp"),
        ],
    )
    def test_real_images_reach_the_model_seam(
        self, client: TestClient, image_format: str, filename: str, content_type: str
    ) -> None:
        """200 (not 400) proves validation passed and the seam answered."""
        response = client.post(
            "/analyze",
            files={
                "file": (filename, make_image_bytes(image_format), content_type)
            },
        )
        assert response.status_code == 200, response.json()

    def test_large_but_legal_image_passes(
        self, client: TestClient
    ) -> None:
        """A 2000x2000 PNG is well under 15 MB and must decode fine."""
        data = make_image_bytes("PNG", 2000, 2000)
        assert len(data) < settings.max_upload_bytes
        response = client.post(
            "/analyze", files={"file": ("big.png", data, "image/png")}
        )
        assert response.status_code == 200, response.json()


class TestOversizedUpload:
    def test_rejects_file_over_the_limit(
        self, client: TestClient, png_upload, tiny_limit: int
    ) -> None:
        response = client.post(
            "/analyze", files=png_upload(b"x" * (tiny_limit + 1))
        )
        assert response.status_code == 413
        assert "upload limit" in response.json()["detail"].lower()

    def test_rejects_file_far_over_the_limit(
        self, client: TestClient, png_upload, tiny_limit: int
    ) -> None:
        response = client.post("/analyze", files=png_upload(b"x" * 4096))
        assert response.status_code == 413

    def test_size_check_runs_before_the_decode_check(
        self, client: TestClient, png_upload, tiny_limit: int
    ) -> None:
        """Oversized garbage is 413, not 400 - the size cap is applied first."""
        response = client.post("/analyze", files=png_upload(b"x" * (tiny_limit + 1)))
        assert response.status_code == 413

    def test_default_limit_is_15mb(self) -> None:
        assert settings.max_upload_bytes == 15 * 1024 * 1024


class TestModelSeam:
    """What the API does with the model layer's answer."""

    def test_valid_upload_returns_the_models_prediction(
        self, client: TestClient, png_bytes: bytes, png_upload
    ) -> None:
        response = client.post("/analyze", files=png_upload(png_bytes))
        assert response.status_code == 200, response.json()
        assert response.json() == {"verdict": "real", "confidence": 0.5}

    def test_validated_upload_reaches_the_model_service(
        self, client: TestClient, png_bytes: bytes, png_upload, stub_service
    ) -> None:
        """The exact validated bytes are what the service receives."""
        client.post("/analyze", files=png_upload(png_bytes))
        assert stub_service.calls == [png_bytes]

    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [(Verdict.REAL, "real"), (Verdict.FAKE, "fake")],
    )
    def test_either_verdict_is_passed_through_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient,
        png_bytes, png_upload, verdict, expected
    ) -> None:
        """The route does not second-guess or re-map the model's verdict."""
        from app.api import routes

        service = StubModelService(
            ModelPrediction(verdict=verdict, confidence=0.42, class_name="x")
        )
        monkeypatch.setattr(routes, "get_model_service", lambda: service)

        response = client.post("/analyze", files=png_upload(png_bytes))
        assert response.status_code == 200, response.json()
        assert response.json()["verdict"] == expected

    def test_confidence_is_returned_as_a_fraction(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient,
        png_bytes, png_upload
    ) -> None:
        """The reference prints a percentage; the API must return 0.0-1.0."""
        from app.api import routes

        service = StubModelService(
            ModelPrediction(verdict=Verdict.FAKE, confidence=0.999, class_name="tampered")
        )
        monkeypatch.setattr(routes, "get_model_service", lambda: service)

        response = client.post("/analyze", files=png_upload(png_bytes))
        assert response.status_code == 200, response.json()
        assert response.json()["confidence"] == 0.999

    def test_no_fallback_prediction_is_invented(
        self, client: TestClient, png_bytes: bytes, png_upload
    ) -> None:
        """A successful response must always carry both contract fields."""
        response = client.post("/analyze", files=png_upload(png_bytes))
        assert set(response.json()) == {"verdict", "confidence"}


class TestModelFailurePaths:
    """Model failures are distinct from upload failures, and stay distinct."""

    def test_unavailable_model_returns_503(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient,
        png_bytes, png_upload
    ) -> None:
        from app.api import routes

        monkeypatch.setattr(
            routes,
            "get_model_service",
            lambda: UnavailableModelService("The analysis model is not available."),
        )

        response = client.post("/analyze", files=png_upload(png_bytes))
        assert response.status_code == 503
        assert "model" in response.json()["detail"].lower()
        assert "verdict" not in response.json()
        assert "confidence" not in response.json()

    def test_inference_failure_returns_500(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient,
        png_bytes, png_upload
    ) -> None:
        from app.api import routes

        class FailingService(StubModelService):
            def analyze(self, image):
                raise ModelInferenceError("secret internal detail")

        monkeypatch.setattr(routes, "get_model_service", lambda: FailingService())

        response = client.post("/analyze", files=png_upload(png_bytes))
        assert response.status_code == 500
        assert "secret internal detail" not in response.text
        assert "Traceback" not in response.text

    def test_malformed_prediction_returns_500_not_a_guess(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient,
        png_bytes, png_upload
    ) -> None:
        """A contract violation must not be silently coerced into a verdict."""
        from app.api import routes

        class GarbageService(StubModelService):
            def analyze(self, image):
                return object()

        monkeypatch.setattr(routes, "get_model_service", lambda: GarbageService())

        response = client.post("/analyze", files=png_upload(png_bytes))
        assert response.status_code == 500
        assert "verdict" not in response.json()
