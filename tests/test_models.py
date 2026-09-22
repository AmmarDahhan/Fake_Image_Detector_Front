import unittest

from core.models import (
    AnalysisResult,
    Verdict,
    format_bytes,
    format_confidence,
)


class TestVerdict(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(Verdict.REAL.label, "Real")
        self.assertEqual(Verdict.FAKE.label, "Fake")

    def test_css_classes(self):
        self.assertEqual(Verdict.REAL.css_class, "real")
        self.assertEqual(Verdict.FAKE.css_class, "fake")

    def test_parse(self):
        self.assertIs(Verdict.parse("real"), Verdict.REAL)
        self.assertIs(Verdict.parse("  FAKE "), Verdict.FAKE)
        with self.assertRaises(ValueError):
            Verdict.parse("synthetic")


class TestAnalysisResult(unittest.TestCase):
    def test_valid_result(self):
        result = AnalysisResult(verdict=Verdict.REAL, confidence=0.9)
        self.assertAlmostEqual(result.confidence_percent, 90.0)
        self.assertIsNone(result.message)
        self.assertIsNone(result.metadata)

    def test_wrong_verdict_type(self):
        with self.assertRaises(TypeError):
            AnalysisResult(verdict="real", confidence=0.5)

    def test_confidence_out_of_range(self):
        for bad in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                AnalysisResult(verdict=Verdict.REAL, confidence=bad)


class TestFormatting(unittest.TestCase):
    def test_format_confidence(self):
        self.assertEqual(format_confidence(0.5), "50.0%")
        self.assertEqual(format_confidence(0.8756), "87.6%")
        self.assertEqual(format_confidence(1.0), "100.0%")
        self.assertEqual(format_confidence(0.0), "0.0%")

    def test_format_confidence_rejects_invalid(self):
        with self.assertRaises(ValueError):
            format_confidence(1.1)
        with self.assertRaises(TypeError):
            format_confidence("high")

    def test_format_bytes(self):
        self.assertEqual(format_bytes(512), "512 B")
        self.assertEqual(format_bytes(1536), "1.5 KB")
        self.assertEqual(format_bytes(3 * 1024 * 1024), "3.0 MB")


if __name__ == "__main__":
    unittest.main()