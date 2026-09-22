import unittest

from core.analyzer import AnalyzerUnavailableError, create_analyzer
from core.config import interpret_result
from core.models import Verdict
from services.api_analyzer import ApiAnalyzer
from services.mock_analyzer import MockAnalyzer


class TestAnalyzerFactory(unittest.TestCase):
    def test_mock_backend(self):
        self.assertIsInstance(create_analyzer("mock"), MockAnalyzer)

    def test_api_backend(self):
        analyzer = create_analyzer("api")
        self.assertIsInstance(analyzer, ApiAnalyzer)
        self.assertFalse(analyzer.is_available)

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            create_analyzer("nonsense")


class TestApiSeam(unittest.TestCase):
    def test_unimplemented_raises_clear_error(self):
        with self.assertRaises(AnalyzerUnavailableError):
            ApiAnalyzer().analyze(b"fake-bytes")


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