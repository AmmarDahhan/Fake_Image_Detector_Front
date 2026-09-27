import unittest
from unittest import mock

from core.analyzer import create_analyzer
from core.config import API_BASE_URL, interpret_result
from core.models import AnalysisResult, Verdict
from services.api_analyzer import ApiAnalyzer
from services.mock_analyzer import MockAnalyzer


class TestAnalyzerFactory(unittest.TestCase):
    def test_mock_backend(self):
        self.assertIsInstance(create_analyzer("mock"), MockAnalyzer)

    def test_api_backend(self):
        analyzer = create_analyzer("api")
        self.assertIsInstance(analyzer, ApiAnalyzer)
        self.assertTrue(analyzer.is_available)

    def test_api_backend_uses_configured_base_url(self):
        self.assertEqual(create_analyzer("api").base_url, API_BASE_URL)

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            create_analyzer("nonsense")


class TestApiSeam(unittest.TestCase):
    """The api backend is now implemented, so it must not be a dead stub."""

    def test_analyze_does_not_raise_unimplemented(self):
        """Regression guard: analyze() must attempt a real request.

        The HTTP layer itself is covered in test_api_analyzer.py; here we only
        assert the seam is wired, because a stub that raised
        AnalyzerUnavailableError unconditionally used to live in this slot.
        """
        with mock.patch("services.api_analyzer.requests.post") as post:
            post.return_value = _fake_response(200, {"verdict": "real", "confidence": 0.5})
            result = ApiAnalyzer().analyze(b"fake-bytes")
        self.assertIsInstance(result, AnalysisResult)

    def test_mock_analyzer_is_still_reachable(self):
        """Flipping ANALYZER_BACKEND back to "mock" must keep working."""
        self.assertIsInstance(create_analyzer("mock"), MockAnalyzer)


def _fake_response(status_code, payload=None, text=""):
    """Minimal stand-in for requests.Response."""
    response = mock.Mock()
    response.status_code = status_code
    response.json = mock.Mock(
        side_effect=lambda: (_ for _ in ()).throw(ValueError("no json"))
        if payload is None
        else payload
    )
    response.text = text
    return response


class TestInterpretation(unittest.TestCase):
    def test_messages_mention_confidence(self):
        real = interpret_result(Verdict.REAL, 0.91)
        fake = interpret_result(Verdict.FAKE, 0.88)
        self.assertIn("Real", real)
        self.assertIn("91.0%", real)
        self.assertIn("Fake", fake)
        self.assertIn("88.0%", fake)


if __name__ == "__main__":
    unittest.main()