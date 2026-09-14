import unittest
from PIL import Image
from homework.preprocessing import prepare_image
from homework.validation import ValidationError
from tools.evaluate_model import metrics


class ToolChecks(unittest.TestCase):
    def test_otsu_preserves_black_text_on_white(self):
        image = Image.new('RGB', (4, 2), 'white')
        image.putpixel((0, 0), (20, 20, 20))
        result = prepare_image(image, 'binary')
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((3, 1)), (255, 255, 255))
        self.assertEqual(image.getpixel((0, 0)), (20, 20, 20))

    def test_preprocessing_validation_and_constant_image(self):
        image = Image.new('RGB', (2, 2), 'white')
        self.assertEqual(prepare_image(image, 'binary').getpixel((0, 0)), (255, 255, 255))
        with self.assertRaises(ValidationError):
            prepare_image(image, 'unknown')

    def test_classification_denominators(self):
        result = metrics(tp=3, tn=4, fp=1, fn=2)
        self.assertAlmostEqual(result['false_positive_rate'], 0.2)
        self.assertAlmostEqual(result['false_negative_rate'], 0.4)
        self.assertAlmostEqual(result['precision'], 0.75)
        self.assertEqual(metrics(0,0,0,0)['f1'], 0)
