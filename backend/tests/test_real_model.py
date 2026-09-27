"""Integration tests against the real trained model.

Marked ``integration`` and excluded by default::

    pytest -m integration

These are the only tests that touch the ~544 MiB checkpoint. The
``real_model_service`` fixture is session-scoped, so the archive is read and
the network built exactly once no matter how many tests are collected; per-test
work is a single forward pass (~1 s on CPU).

The expected predictions below were produced by running the AI team's own
reference script, ``ai_model/predict.py``, unmodified against the same
checkpoint, and are pinned here so a change in the service's preprocessing,
class ordering or confidence maths fails loudly instead of drifting.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.schemas.analysis import Verdict
from app.services.model_service import (
    MODEL_INPUT_SIZE,
    MODEL_NORMALIZE_MEAN,
    MODEL_NORMALIZE_STD,
    MODEL_RESAMPLE,
    ImagePayload,
    ModelNotAvailableError,
    RealModelService,
)
from tests.conftest import SAMPLE_IMAGE_DIR, make_image_bytes

pytestmark = pytest.mark.integration

# Recorded from an unmodified run of ai_model/predict.py on CPU, over every
# image currently in ai_model/images/:
#   1554-00-01-..._book-olds.ru.jpg -> real          (99.90%)
#   1_Atlantis.png                 -> full_synthetic (99.45%)
#   jonny-gios-...-unsplash.jpg    -> real          (90.66%)
#   men with fake things.jpg       -> real          (79.84%)
#   robot.jpg                      -> full_synthetic (100.00%)
#   women fake.jpg                 -> tampered      (73.44%)
# Confidence is stored as the 0.0-1.0 fraction the API returns.
#
# Two of these are worth a note, because both are less obvious than the
# filename suggests and a future reader would otherwise "correct" them:
#
#   * `men with fake things.jpg` is classified real. The filename describes the
#     subject matter, not the ground truth; the model does not agree that this
#     image is synthetic.
#   * `robot.jpg` is not literally 1.0. The reference script prints two
#     decimal places of a percentage, so 0.99997 renders as "100.00%". The
#     extra digit is kept here so the pinned value is the true one and does
#     not imply the model is claiming certainty.
#
# `women fake.jpg` is the only sample the model reports as `tampered`, which is
# what makes the provisional `tampered -> fake` collapse observable rather than
# merely documented.
EXPECTED_SAMPLE_PREDICTIONS = {
    "1_Atlantis.png": ("full_synthetic", Verdict.FAKE, 0.9945),
    "jonny-gios-pfzJIn6wVas-unsplash.jpg": ("real", Verdict.REAL, 0.9066),
    "1554-00-01-цвет_Сиверская._Железнодорожный_мост_-_book-olds.ru.jpg": (
        "real",
        Verdict.REAL,
        0.9990,
    ),
    "men with fake things.jpg": ("real", Verdict.REAL, 0.7984),
    "robot.jpg": ("full_synthetic", Verdict.FAKE, 0.99997),
    "women fake.jpg": ("tampered", Verdict.FAKE, 0.7344),
}

#: Sample filenames the reference run classified as an authentic photograph.
#: Derived from the table above rather than listed separately, so adding a
#: sample cannot leave this set silently stale.
EXPECTED_REAL_SAMPLES = frozenset(
    name
    for name, (_, verdict, _) in EXPECTED_SAMPLE_PREDICTIONS.items()
    if verdict is Verdict.REAL
)

#: Sample filenames the reference run classified as not authentic.
EXPECTED_FAKE_SAMPLES = frozenset(
    name
    for name, (_, verdict, _) in EXPECTED_SAMPLE_PREDICTIONS.items()
    if verdict is Verdict.FAKE
)


def sample_images() -> list:
    return sorted(
        path
        for path in SAMPLE_IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )


class TestCheckpointWiring:
    """The model loaded is the model the reference script loads."""

    def test_model_is_loaded(self, real_model_service: RealModelService) -> None:
        assert real_model_service.is_available

    def test_class_order_comes_from_the_checkpoint(
        self, real_model_service: RealModelService
    ) -> None:
        assert real_model_service.class_names == (
            "full_synthetic",
            "real",
            "tampered",
        )

    def test_classifier_outputs_three_classes(
        self, real_model_service: RealModelService
    ) -> None:
        assert real_model_service._model.classifier[2].out_features == 3

    def test_default_path_is_project_relative_not_absolute_windows_drive(
        self, real_model_service: RealModelService
    ) -> None:
        """No drive-letter path is baked into the code."""
        configured = settings.model_path or ""
        assert not configured.startswith(("C:", "D:", "F:"))

    def test_device_is_resolved(self, real_model_service: RealModelService) -> None:
        assert real_model_service.device in {"cpu", "cuda"}


class TestPreprocessingParity:
    """Preprocessing must match ai_model/predict.py exactly."""

    def test_input_size_is_350(self) -> None:
        assert MODEL_INPUT_SIZE == (350, 350)

    def test_resampling_is_bilinear(self) -> None:
        assert MODEL_RESAMPLE is Image.Resampling.BILINEAR

    def test_normalisation_is_imagenet(self) -> None:
        assert MODEL_NORMALIZE_MEAN == [0.485, 0.456, 0.406]
        assert MODEL_NORMALIZE_STD == [0.229, 0.224, 0.225]

    def test_preprocessed_tensor_has_the_expected_shape(
        self, real_model_service: RealModelService, png_bytes: bytes
    ) -> None:
        tensor = real_model_service._preprocess(
            ImagePayload(data=png_bytes, filename="photo.png")
        )
        assert tuple(tensor.shape) == (1, 3, 350, 350)
        assert tensor.dtype.is_floating_point

    def test_grayscale_input_is_converted_to_rgb(
        self, real_model_service: RealModelService
    ) -> None:
        """A greyscale PNG must still yield three channels, not one."""
        from io import BytesIO

        buffer = BytesIO()
        Image.new("L", (200, 120), 128).save(buffer, format="PNG")

        tensor = real_model_service._preprocess(
            ImagePayload(data=buffer.getvalue(), filename="grey.png")
        )
        assert tuple(tensor.shape) == (1, 3, 350, 350)

    def test_non_square_input_is_resized_not_cropped(
        self, real_model_service: RealModelService
    ) -> None:
        from io import BytesIO

        buffer = BytesIO()
        Image.new("RGB", (1200, 400), (10, 20, 30)).save(buffer, format="PNG")

        tensor = real_model_service._preprocess(
            ImagePayload(data=buffer.getvalue(), filename="wide.png")
        )
        assert tuple(tensor.shape) == (1, 3, 350, 350)


class TestSampleImageParity:
    """The recorded reference predictions must be reproduced exactly."""

    def test_all_sample_images_are_present(self) -> None:
        """The pinned table must cover the sample directory exactly.

        A one-sided comparison is what let this drift: new images appeared in
        ai_model/images/ with no corresponding entry here. The message names
        both directions of the difference so the fix is obvious.
        """
        on_disk = {path.name for path in sample_images()}
        pinned = set(EXPECTED_SAMPLE_PREDICTIONS)

        assert on_disk == pinned, (
            f"sample set drifted - on disk but not pinned: "
            f"{sorted(on_disk - pinned)}; pinned but not on disk: "
            f"{sorted(pinned - on_disk)}. Re-run ai_model/predict.py over "
            f"ai_model/images/ and refresh EXPECTED_SAMPLE_PREDICTIONS."
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_SAMPLE_PREDICTIONS))
    def test_prediction_matches_the_reference_script(
        self, real_model_service: RealModelService, name: str
    ) -> None:
        expected_class, expected_verdict, expected_confidence = (
            EXPECTED_SAMPLE_PREDICTIONS[name]
        )

        data = (SAMPLE_IMAGE_DIR / name).read_bytes()
        prediction = real_model_service.analyze(
            ImagePayload(data=data, filename=name)
        )

        assert prediction.class_name == expected_class
        assert prediction.verdict is expected_verdict
        # 2 decimal places is the reference script's own print precision.
        assert prediction.confidence == pytest.approx(
            expected_confidence, abs=0.0001
        )

    def test_synthetic_sample_is_reported_as_fake(
        self, real_model_service: RealModelService
    ) -> None:
        """The headline case: an AI-generated image must come back 'fake'."""
        data = (SAMPLE_IMAGE_DIR / "1_Atlantis.png").read_bytes()
        prediction = real_model_service.analyze(
            ImagePayload(data=data, filename="1_Atlantis.png")
        )
        assert prediction.class_name == "full_synthetic"
        assert prediction.verdict is Verdict.FAKE

    def test_photograph_samples_are_reported_as_real(
        self, real_model_service: RealModelService
    ) -> None:
        """Every sample the reference run called real must come back real.

        Scoped to ``EXPECTED_REAL_SAMPLES`` rather than to "everything except
        1_Atlantis.png". The sample directory holds genuine photographs *and*
        synthetic and tampered images, so the old blanket assumption - that
        anything which is not the headline fake is a real photograph - was
        false and failed as soon as the directory grew past its first three
        files.
        """
        assert EXPECTED_REAL_SAMPLES, "fixture is empty; the test proves nothing"

        for name in sorted(EXPECTED_REAL_SAMPLES):
            prediction = real_model_service.analyze(
                ImagePayload(
                    data=(SAMPLE_IMAGE_DIR / name).read_bytes(), filename=name
                )
            )
            assert prediction.verdict is Verdict.REAL, name

    def test_synthetic_and_tampered_samples_are_reported_as_fake(
        self, real_model_service: RealModelService
    ) -> None:
        """The mirror of the test above, and the reason it was worth splitting.

        This sample set deliberately contains a `full_synthetic` image and a
        `tampered` one, so both halves of the provisional binary collapse in
        app/services/model_class_map.py are exercised against real data.
        """
        assert EXPECTED_FAKE_SAMPLES, "fixture is empty; the test proves nothing"

        for name in sorted(EXPECTED_FAKE_SAMPLES):
            prediction = real_model_service.analyze(
                ImagePayload(
                    data=(SAMPLE_IMAGE_DIR / name).read_bytes(), filename=name
                )
            )
            assert prediction.verdict is Verdict.FAKE, name

    def test_the_tampered_sample_is_really_tampered(
        self, real_model_service: RealModelService
    ) -> None:
        """Pin which sample carries the `tampered` class.

        The class is what makes the provisional collapse meaningful, so if a
        future checkpoint stops producing it, that is a contract change worth
        failing on rather than silently absorbing into "fake".
        """
        data = (SAMPLE_IMAGE_DIR / "women fake.jpg").read_bytes()
        prediction = real_model_service.analyze(
            ImagePayload(data=data, filename="women fake.jpg")
        )
        assert prediction.class_name == "tampered"
        assert prediction.verdict is Verdict.FAKE


class TestConfidenceContract:
    def test_confidence_is_a_fraction_not_a_percentage(
        self, real_model_service: RealModelService, png_bytes: bytes
    ) -> None:
        prediction = real_model_service.analyze(
            ImagePayload(data=png_bytes, filename="photo.png")
        )
        assert 0.0 <= prediction.confidence <= 1.0

    def test_confidence_is_never_reported_above_one(
        self, real_model_service: RealModelService
    ) -> None:
        """Catches the reference script's *100 being carried over by mistake."""
        for path in sample_images():
            prediction = real_model_service.analyze(
                ImagePayload(data=path.read_bytes(), filename=path.name)
            )
            assert prediction.confidence <= 1.0, path.name


class TestRealServiceThroughTheApi:
    """End to end through POST /analyze, with the real model in the seam."""

    def test_analyze_returns_a_valid_prediction(
        self, real_model_service: RealModelService, client: TestClient
    ) -> None:
        from app.api import routes

        original = routes.get_model_service
        try:
            routes.get_model_service = lambda: real_model_service

            data = (SAMPLE_IMAGE_DIR / "1_Atlantis.png").read_bytes()
            response = client.post(
                "/analyze", files={"file": ("1_Atlantis.png", data, "image/png")}
            )
        finally:
            routes.get_model_service = original

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["verdict"] == "fake"
        assert 0.0 <= body["confidence"] <= 1.0
        assert set(body) == {"verdict", "confidence"}

    def test_health_still_passes_with_the_real_model_loaded(
        self, real_model_service: RealModelService, client: TestClient
    ) -> None:
        """Model readiness must not leak into the liveness probe."""
        assert real_model_service.is_available
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_invalid_upload_still_returns_400_with_the_real_model(
        self, real_model_service: RealModelService, client: TestClient
    ) -> None:
        from app.api import routes

        original = routes.get_model_service
        try:
            routes.get_model_service = lambda: real_model_service
            response = client.post(
                "/analyze",
                files={
                    "file": (
                        "photo.png",
                        b"not an image at all, just text",
                        "image/png",
                    )
                },
            )
        finally:
            routes.get_model_service = original

        assert response.status_code == 400


class TestModelReuse:
    """The checkpoint is loaded once, not per request."""

    def test_repeated_inference_reuses_the_same_model(
        self, real_model_service: RealModelService
    ) -> None:
        first = real_model_service.analyze(
            ImagePayload(data=make_image_bytes("PNG"), filename="a.png")
        )
        model_after_first = real_model_service._model

        second = real_model_service.analyze(
            ImagePayload(data=make_image_bytes("PNG"), filename="b.png")
        )

        assert real_model_service._model is model_after_first
        assert first.verdict is second.verdict

    def test_repeated_inference_is_deterministic(
        self, real_model_service: RealModelService
    ) -> None:
        data = (SAMPLE_IMAGE_DIR / "1_Atlantis.png").read_bytes()
        payload = ImagePayload(data=data, filename="1_Atlantis.png")

        first = real_model_service.analyze(payload)
        second = real_model_service.analyze(payload)

        assert first.class_name == second.class_name
        assert first.confidence == pytest.approx(second.confidence, abs=1e-6)


class TestLoadFailure:
    """A checkpoint that cannot be loaded is a 503, not a verdict."""

    def test_missing_checkpoint_raises_model_not_available(self) -> None:
        with pytest.raises(ModelNotAvailableError):
            RealModelService(checkpoint_path="does_not_exist.pth")

    def test_load_failure_message_leaks_no_traceback(self) -> None:
        with pytest.raises(ModelNotAvailableError) as excinfo:
            RealModelService(checkpoint_path="does_not_exist.pth")
        assert "Traceback" not in str(excinfo.value)

    def test_missing_checkpoint_does_not_crash_the_api(
        self, client: TestClient
    ) -> None:
        """The 503 path still holds when the model genuinely cannot load."""
        from app.api import routes

        original = routes.get_model_service

        def exploding_factory():
            raise ModelNotAvailableError("The analysis model could not be loaded.")

        try:
            routes.get_model_service = exploding_factory
            response = client.post(
                "/analyze",
                files={"file": ("photo.png", make_image_bytes("PNG"), "image/png")},
            )
        finally:
            routes.get_model_service = original

        assert response.status_code == 503
        assert "verdict" not in response.json()
