import unittest

from core.validation import ImageValidationError, validate_image
from tests._helpers import make_png_bytes


class TestValidation(unittest.TestCase):
    def test_valid_png(self):
        data = make_png_bytes(24, 18)
        info = validate_image("photo.png", data, "image/png")
        self.assertEqual(info.filename, "photo.png")
        self.assertEqual(info.width, 24)
        self.assertEqual(info.height, 18)
        self.assertEqual(info.format_name, "PNG")
        self.assertEqual(info.size_bytes, len(data))
        self.assertIn("24", info.dimensions_label)

    def test_unsupported_extension(self):
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image("notes.txt", b"hello", "text/plain")
        self.assertEqual(ctx.exception.code, "unsupported_file")

    def test_oversized(self):
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image("big.png", make_png_bytes(), "image/png", max_bytes=10)
        self.assertEqual(ctx.exception.code, "oversized")

    def test_invalid_image_bytes(self):
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image("fake.png", b"this is not really an image", "image/png")
        self.assertEqual(ctx.exception.code, "invalid_image")

    def test_empty_upload(self):
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image("", b"")
        self.assertEqual(ctx.exception.code, "empty")


if __name__ == "__main__":
    unittest.main()