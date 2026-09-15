"""模块一图像预处理测试（M1-UT-037～M1-UT-039）。"""
import sys
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from homework.preprocessing import prepare_image
from homework.validation import ValidationError


class ImagePreprocessingTests(unittest.TestCase):
    def test_m1_ut_037_grayscale_mode_outputs_rgb(self):
        result = prepare_image(Image.new("RGB", (2, 2), "red"), "grayscale")
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0))[0], result.getpixel((0, 0))[1])

    def test_m1_ut_038_binary_mode_preserves_dark_text(self):
        image = Image.new("RGB", (4, 2), "white")
        image.putpixel((0, 0), (20, 20, 20))
        result = prepare_image(image, "binary")
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((3, 1)), (255, 255, 255))

    def test_m1_ut_039_unknown_preprocessing_mode_is_rejected(self):
        with self.assertRaises(ValidationError):
            prepare_image(Image.new("RGB", (2, 2), "white"), "unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
