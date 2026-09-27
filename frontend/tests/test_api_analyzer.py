"""Tests for the real backend call in :class:`ApiAnalyzer`.

The HTTP layer is mocked, so none of these need a running FastAPI. What they
pin is the wire contract and the response mapping:

    request   multipart/form-data, one part named "file", holding the exact
              validated bytes the user selected
    response  {"verdict": "real"|"fake", "confidence": 0.0-1.0}
              -> AnalysisResult(verdict=Verdict, confidence=float)

The 0.9945 confidence from the real backend is used throughout as the
canonical case, because it is what the actual model returns for the sample
AI-generated image and it exercises the display path (99.4% / 99.45).
"""

from __future__ import annotations

import logging
import unittest
from unittest import mock

import requests

from core.analyzer import AnalyzerUnavailableError
from core.config import (
    API_BASE_URL,
    interpret_result,
)
from core.models import AnalysisResult, Verdict, format_confidence
from core.validation import validate_image
from services.api_analyzer import ApiAnalyzer, ApiAnalyzerError, UPLOAD_FIELD_NAME

from tests._helpers import make_png_bytes

POST_TARGET = "services.api_analyzer.requests.post"

# What the real backend returns for ai_model/images/1_Atlantis.png.
REAL_BACKEND_PAYLOAD = {"verdict": "fake", "confidence": 0.9945}


def setUpModule() -> None:
    """Silence the analyzer's error logging for the duration of these tests.

    The failure paths below log on purpose; without this they print to stderr
    and bury the test results. Production logging is untouched.
    """
    logger = logging.getLogger("services.api_analyzer")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False


def fake_response(status_code: int, payload=None, *, raw: str | None = None) -> mock.Mock:
    """A stand-in for ``requests.Response``.

    ``payload=None`` makes ``.json()`` raise, reproducing a body that is not
    JSON at all.
    """
    response = mock.Mock()
    response.status_code = status_code

    if payload is None:
        response.json.side_effect = ValueError("Expecting value: line 1 column 1")
    else:
        response.json.return_value = payload

    response.text = raw if raw is not None else ""
    return response


class RequestShapeTest(unittest.TestCase):
    """The outgoing request must match the backend's route signature."""

    def setUp(self) -> None:
        self.bytes = make_png_bytes(24, 24)
        self.analyzer = ApiAnalyzer()

    def test_bytes_are_sent_as_multipart_field_named_file(self):
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(200, REAL_BACKEND_PAYLOAD)
            self.analyzer.analyze(self.bytes)

        files = post.call_args.kwargs["files"]
        self.assertIn("file", files)
        self.assertEqual(list(files), [UPLOAD_FIELD_NAME])
        self.assertEqual(UPLOAD_FIELD_NAME, "file")

    def test_exact_uploaded_bytes_are_transmitted_unmodified(self):
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(200, REAL_BACKEND_PAYLOAD)
            self.analyzer.analyze(self.bytes)

        filename, payload, _mime = post.call_args.kwargs["files"]["file"]
        self.assertEqual(payload, self.bytes)

    def test_request_goes_to_the_analyze_endpoint(self):
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(200, REAL_BACKEND_PAYLOAD)
            self.analyzer.analyze(self.bytes)

        self.assertEqual(post.call_args.args[0], f"{API_BASE_URL}/analyze")
        self.assertEqual(post.call_args.args[0], "http://localhost:8000/analyze")

    def test_filename_and_mime_come_from_image_info(self):
        info = validate_image("photo.png", self.bytes, "image/png")
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(200, REAL_BACKEND_PAYLOAD)
            self.analyzer.analyze(self.bytes, info)

        filename, _bytes, mime = post.call_args.kwargs["files"]["file"]
        self.assertEqual(filename, "photo.png")
        self.assertEqual(mime, "image/png")

    def test_missing_image_info_falls_back_to_a_usable_filename(self):
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(200, REAL_BACKEND_PAYLOAD)
            self.analyzer.analyze(self.bytes, None)

        filename, _bytes, mime = post.call_args.kwargs["files"]["file"]
        self.assertEqual(filename, "upload")
        self.assertEqual(mime, "application/octet-stream")

    def test_configured_timeout_is_applied(self):
        analyzer = ApiAnalyzer(timeout=12.5)
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(200, REAL_BACKEND_PAYLOAD)
            analyzer.analyze(self.bytes)
        self.assertEqual(post.call_args.kwargs["timeout"], 12.5)

    def test_base_url_override_is_respected(self):
        analyzer = ApiAnalyzer(base_url="https://api.example.com/")
        self.assertEqual(analyzer.url, "https://api.example.com/analyze")


class ResponseMappingTest(unittest.TestCase):
    """The response becomes the existing AnalysisResult, unchanged."""

    def setUp(self) -> None:
        self.analyzer = ApiAnalyzer()
        self.bytes = make_png_bytes()

    def _analyze(self, payload):
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(200, payload)
            return self.analyzer.analyze(self.bytes)

    def test_returns_the_existing_analysisresult_type(self):
        result = self._analyze(REAL_BACKEND_PAYLOAD)
        self.assertIsInstance(result, AnalysisResult)

    def test_confidence_0_9945_stays_a_fraction_and_renders_as_a_percentage(self):
        """0.9945 from the API is the 0.0-1.0 form AnalysisResult already wants.

        Stored unchanged as 0.9945; the display layer multiplies by 100. Note
        0.9945 * 100 is 99.450000000000003 in binary floating point, so the
        one-decimal rendering is 99.5%.
        """
        result = self._analyze(REAL_BACKEND_PAYLOAD)

        self.assertEqual(result.confidence, 0.9945)
        self.assertAlmostEqual(result.confidence_percent, 99.45, places=6)
        self.assertEqual(format_confidence(result.confidence), "99.5%")

    def test_confidence_is_not_multiplied_into_a_percentage(self):
        """Guards against 0.9945 arriving as 99.45 and breaking the range check."""
        result = self._analyze(REAL_BACKEND_PAYLOAD)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreaterEqual(result.confidence, 0.0)

    def test_fake_verdict_reaches_the_result_card(self):
        result = self._analyze(REAL_BACKEND_PAYLOAD)
        self.assertIs(result.verdict, Verdict.FAKE)
        self.assertEqual(result.verdict.label, "Fake")
        self.assertEqual(result.verdict.css_class, "fake")
        self.assertIn("Fake", interpret_result(result.verdict, result.confidence))
        self.assertIn("99.5%", interpret_result(result.verdict, result.confidence))

    def test_real_verdict_reaches_the_result_card(self):
        result = self._analyze({"verdict": "real", "confidence": 0.9066})
        self.assertIs(result.verdict, Verdict.REAL)
        self.assertEqual(result.verdict.label, "Real")
        self.assertEqual(result.verdict.css_class, "real")
        self.assertIn("90.7%", format_confidence(result.confidence))

    def test_verdict_is_case_insensitive(self):
        result = self._analyze({"verdict": "REAL", "confidence": 0.5})
        self.assertIs(result.verdict, Verdict.REAL)

    def test_boundary_confidences_are_accepted(self):
        for value in (0.0, 1.0):
            result = self._analyze({"verdict": "fake", "confidence": value})
            self.assertEqual(result.confidence, value)

    def test_metadata_marks_the_api_as_the_source(self):
        """Distinguishable from MockAnalyzer, whose source is "demo"."""
        result = self._analyze(REAL_BACKEND_PAYLOAD)
        self.assertEqual(result.metadata["source"], "api")
        self.assertNotEqual(result.metadata["source"], "demo")

    def test_extra_backend_fields_are_ignored(self):
        """An additive contract change must not break the frontend."""
        result = self._analyze(
            {
                "verdict": "fake",
                "confidence": 0.9945,
                "model": "convnext_base",
                "classes": ["full_synthetic", "real", "tampered"],
            }
        )
        self.assertIs(result.verdict, Verdict.FAKE)
        self.assertEqual(result.confidence, 0.9945)


class MalformedResponseTest(unittest.TestCase):
    """A broken contract is refused, never guessed at."""

    def setUp(self) -> None:
        self.analyzer = ApiAnalyzer()
        self.bytes = make_png_bytes()

    def _assert_rejected(self, payload=None, raw=None, status=200):
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(status, payload, raw=raw)
            with self.assertRaises(ApiAnalyzerError):
                self.analyzer.analyze(self.bytes)

    def test_unparseable_json_is_rejected(self):
        self._assert_rejected(raw="<html>gateway error</html>")

    def test_json_that_is_not_an_object_is_rejected(self):
        self._assert_rejected(payload=[1, 2, 3])

    def test_missing_verdict_is_rejected(self):
        self._assert_rejected(payload={"confidence": 0.9})

    def test_missing_confidence_is_rejected(self):
        self._assert_rejected(payload={"verdict": "fake"})

    def test_unknown_verdict_is_rejected(self):
        self._assert_rejected(payload={"verdict": "maybe", "confidence": 0.9})

    def test_non_numeric_confidence_is_rejected(self):
        self._assert_rejected(payload={"verdict": "fake", "confidence": "0.99"})

    def test_null_confidence_is_rejected(self):
        self._assert_rejected(payload={"verdict": "fake", "confidence": None})

    def test_boolean_confidence_is_rejected(self):
        """True is an int in Python; it is not a confidence."""
        self._assert_rejected(payload={"verdict": "fake", "confidence": True})

    def test_out_of_range_confidence_is_rejected(self):
        self._assert_rejected(payload={"verdict": "fake", "confidence": 1.7})

    def test_negative_confidence_is_rejected(self):
        self._assert_rejected(payload={"verdict": "fake", "confidence": -0.2})

    def test_empty_body_with_200_is_rejected(self):
        self._assert_rejected(raw="")

    def test_error_messages_never_contain_a_traceback(self):
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(200, None, raw="boom")
            with self.assertRaises(ApiAnalyzerError) as ctx:
                self.analyzer.analyze(self.bytes)
        self.assertNotIn("Traceback", str(ctx.exception))
        self.assertNotIn("File \"", str(ctx.exception))


class HttpStatusTest(unittest.TestCase):
    """Each documented status maps onto the right user-facing error state."""

    def setUp(self) -> None:
        self.analyzer = ApiAnalyzer()
        self.bytes = make_png_bytes()

    def _analyze(self, status, payload=None, raw=""):
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(status, payload, raw=raw)
            return self.analyzer.analyze(self.bytes)

    def test_503_reports_the_service_as_unavailable(self):
        """Model not loaded: the UI's 'Analysis service unavailable' state."""
        with self.assertRaises(AnalyzerUnavailableError):
            self._analyze(503, {"detail": "The analysis model is not available."})

    def test_400_reports_a_failed_analysis(self):
        with self.assertRaises(ApiAnalyzerError) as ctx:
            self._analyze(400, {"detail": "The file could not be read as an image."})
        self.assertIn("400", str(ctx.exception))

    def test_413_reports_a_failed_analysis(self):
        with self.assertRaises(ApiAnalyzerError) as ctx:
            self._analyze(413, {"detail": "Image is larger than the 15 MB upload limit."})
        self.assertIn("413", str(ctx.exception))

    def test_422_reports_a_failed_analysis(self):
        with self.assertRaises(ApiAnalyzerError) as ctx:
            self._analyze(422, {"detail": "Field required"})
        self.assertIn("422", str(ctx.exception))

    def test_500_reports_a_failed_analysis(self):
        with self.assertRaises(ApiAnalyzerError) as ctx:
            self._analyze(500, {"detail": "Analysis failed."})
        self.assertIn("500", str(ctx.exception))

    def test_404_reports_a_failed_analysis(self):
        with self.assertRaises(ApiAnalyzerError):
            self._analyze(404, {"detail": "Not Found"})

    def test_201_is_not_treated_as_success(self):
        """Only 200 carries the agreed schema."""
        with self.assertRaises(ApiAnalyzerError):
            self._analyze(201, {"verdict": "fake", "confidence": 0.9})

    def test_backend_detail_is_captured_for_logs(self):
        with self.assertRaises(ApiAnalyzerError) as ctx:
            self._analyze(400, {"detail": "The file could not be read as an image."})
        self.assertIn("could not be read", str(ctx.exception))

    def test_status_errors_without_json_body_still_succeed(self):
        """A non-JSON error body must not mask the status code."""
        with self.assertRaises(ApiAnalyzerError) as ctx:
            self._analyze(502, raw="Bad Gateway")
        self.assertIn("502", str(ctx.exception))


class TransportFailureTest(unittest.TestCase):
    """An unreachable backend is 'unavailable', not a failed analysis."""

    def setUp(self) -> None:
        self.analyzer = ApiAnalyzer()
        self.bytes = make_png_bytes()

    def test_connection_error_is_unavailable(self):
        with mock.patch(POST_TARGET, side_effect=requests.ConnectionError("refused")):
            with self.assertRaises(AnalyzerUnavailableError):
                self.analyzer.analyze(self.bytes)

    def test_timeout_is_unavailable(self):
        with mock.patch(POST_TARGET, side_effect=requests.Timeout("timed out")):
            with self.assertRaises(AnalyzerUnavailableError):
                self.analyzer.analyze(self.bytes)

    def test_generic_request_exception_is_unavailable(self):
        with mock.patch(POST_TARGET, side_effect=requests.RequestException("boom")):
            with self.assertRaises(AnalyzerUnavailableError):
                self.analyzer.analyze(self.bytes)

    def test_unavailable_message_names_the_backend(self):
        analyzer = ApiAnalyzer(base_url="http://elsewhere:9000")
        with mock.patch(POST_TARGET, side_effect=requests.ConnectionError("refused")):
            with self.assertRaises(AnalyzerUnavailableError) as ctx:
                analyzer.analyze(self.bytes)
        self.assertIn("http://elsewhere:9000", str(ctx.exception))

    def test_unavailable_error_is_not_an_api_analyzer_error(self):
        """The two states must stay distinguishable by the UI."""
        self.assertFalse(issubclass(AnalyzerUnavailableError, ApiAnalyzerError))
        self.assertFalse(issubclass(ApiAnalyzerError, AnalyzerUnavailableError))

    def test_empty_bytes_never_reach_the_network(self):
        with mock.patch(POST_TARGET) as post:
            with self.assertRaises(ApiAnalyzerError):
                self.analyzer.analyze(b"")
        post.assert_not_called()


class RealBackendFaithfulnessTest(unittest.TestCase):
    """Values recorded from the real backend must map to the real result."""

    def test_recorded_atlantis_prediction(self):
        """ai_model/images/1_Atlantis.png: full_synthetic -> fake, 99.45%."""
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(
                200, {"verdict": "fake", "confidence": 0.9945141077041626}
            )
            result = ApiAnalyzer().analyze(make_png_bytes())

        self.assertIs(result.verdict, Verdict.FAKE)
        self.assertEqual(format_confidence(result.confidence), "99.5%")

    def test_recorded_jonny_gios_prediction(self):
        """ai_model/images/jonny-gios-...jpg: real -> real, 90.66%."""
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(
                200, {"verdict": "real", "confidence": 0.9066187739372253}
            )
            result = ApiAnalyzer().analyze(make_png_bytes())

        self.assertIs(result.verdict, Verdict.REAL)
        self.assertEqual(format_confidence(result.confidence), "90.7%")


class MockAnalyzerIsolationTest(unittest.TestCase):
    """API mode must never fall back to the demo analyzer."""

    def test_api_analyzer_never_reports_the_demo_source(self):
        with mock.patch(POST_TARGET) as post:
            post.return_value = fake_response(200, REAL_BACKEND_PAYLOAD)
            result = ApiAnalyzer().analyze(make_png_bytes())
        self.assertNotEqual(result.metadata.get("source"), "demo")
        self.assertNotIn("MockAnalyzer", str(result.metadata))

    def test_api_analyzer_never_sleeps_like_the_mock(self):
        """The mock's artificial delay must not appear in the API path."""
        analyzer = ApiAnalyzer()
        with mock.patch("services.api_analyzer.requests.post") as post:
            post.return_value = fake_response(200, REAL_BACKEND_PAYLOAD)
            with mock.patch("time.sleep") as sleep:
                analyzer.analyze(make_png_bytes())
        sleep.assert_not_called()

    def test_api_failure_does_not_fall_back_to_the_mock(self):
        """A broken backend must surface an error, never a demo verdict."""
        from services.mock_analyzer import MockAnalyzer

        with mock.patch(POST_TARGET, side_effect=requests.ConnectionError("refused")):
            with self.assertRaises(AnalyzerUnavailableError):
                ApiAnalyzer().analyze(make_png_bytes())
        # The mock still exists and still works when asked for directly.
        self.assertIsInstance(MockAnalyzer(delay_range=(0, 0)).analyze(b"x"), AnalysisResult)


if __name__ == "__main__":
    unittest.main()
