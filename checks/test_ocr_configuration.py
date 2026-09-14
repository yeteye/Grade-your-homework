"""Regression checks for project-local OCR configuration and error handling."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from homework import ocr, ocr_config
from homework.ai_grader import GradingUnavailable


class OcrConfigurationChecks(unittest.TestCase):
    def test_local_runtime_and_language_detection_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='ocr path ') as folder:
            root = Path(folder)
            runtime = root / 'runtime/tesseract'
            (runtime / 'tessdata').mkdir(parents=True)
            executable = runtime / 'tesseract.exe'
            executable.write_bytes(b'placeholder')
            (runtime / 'tessdata/eng.traineddata').write_bytes(b'placeholder')
            with patch.object(ocr_config, 'ROOT', root), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(ocr_config.tesseract_command(), executable)
                self.assertEqual(ocr_config.tesseract_languages(), ['eng'])
                with self.assertRaisesRegex(GradingUnavailable, 'chi_sim'):
                    ocr.recognize(root / 'image.png', 'Tesseract', '中文')
                with patch('homework.ocr.subprocess.run') as run:
                    run.return_value.stdout = 'Recognized text\n'
                    self.assertEqual(ocr.recognize(root / 'image.png', 'Tesseract', '英文'), 'Recognized text')
                    args = run.call_args.args[0]
                    self.assertEqual(args[args.index('--tessdata-dir') + 1], str(runtime / 'tessdata'))

    def test_missing_model_files_do_not_report_ready(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'HOMEWORK_OCR_MODEL_DIR': folder}):
            for name in ocr_config.PADDLE_MODELS:
                directory = Path(folder) / name
                directory.mkdir()
                (directory / 'inference.json').write_text('{}')
            self.assertFalse(ocr_config.paddle_models_ready())
            self.assertFalse(ocr.status()['PaddleOCR'])
            for name in ocr_config.PADDLE_MODELS:
                for filename in ocr_config.MODEL_FILES:
                    (Path(folder) / name / filename).write_bytes(b'placeholder')
            self.assertTrue(ocr_config.paddle_models_ready())

    def test_explicit_tesseract_path_is_not_silently_replaced(self):
        with patch.dict(os.environ, {'TESSERACT_CMD': 'missing/tesseract.exe'}):
            self.assertEqual(ocr_config.tesseract_command(), Path('missing/tesseract.exe'))
            self.assertFalse(ocr.status()['Tesseract'])
