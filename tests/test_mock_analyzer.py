import unittest

from core.models import Verdict
from services.mock_analyzer import MockAnalyzer

from tests._helpers import make_png_bytes


class TestMockAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MockAnalyzer(delay_range=(0, 0))

    def test_returns_typed_result(self):
        result = self.analyzer.analyze(make_png_bytes())
        self.assertIsInstance(result.verdict, Verdict)
        self.assertIn(result.verdict, (Verdict.REAL, Verdict.FAKE))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreaterEqual(result.confidence, 0.55)

    def test_deterministic_for_same_bytes(self):
        data = make_png_bytes(32, 32)
        first = self.analyzer.analyze(data)
        second = self.analyzer.analyze(data)
        self.assertEqual(first.verdict, second.verdict)
        self.assertEqual(first.confidence, second.confidence)

    def test_metadata_marks_demo_source(self):
        result = self.analyzer.analyze(make_png_bytes())
        self.assertEqual(result.metadata["source"], "demo")


if __name__ == "__main__":
    unittest.main()