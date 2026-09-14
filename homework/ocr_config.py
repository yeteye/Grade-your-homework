"""Project-local OCR paths; importing this module never downloads models."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PADDLE_MODELS = ('PP-OCRv5_mobile_det', 'PP-OCRv5_mobile_rec')
MODEL_FILES = ('inference.json', 'inference.pdiparams', 'inference.yml')


def paddle_model_root():
    return Path(os.environ.get('HOMEWORK_OCR_MODEL_DIR', ROOT / 'models/ocr'))


def paddle_models_ready():
    return all((paddle_model_root() / name / filename).is_file()
               for name in PADDLE_MODELS for filename in MODEL_FILES)


def configure_paddle_cache():
    # Keep runtime caches separate from the immutable downloaded model files.
    cache = ROOT / 'instance/cache'
    os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(cache / 'paddlex'))
    os.environ.setdefault('PADDLE_HOME', str(cache / 'paddle'))
    os.environ.setdefault('HF_HOME', str(cache / 'huggingface'))


def tesseract_command():
    configured = os.environ.get('TESSERACT_CMD')
    if configured:
        return Path(configured)
    candidates = [ROOT / folder / 'tesseract.exe' for folder in
                  ('runtime/tesseract', 'Tesseract-OCR', 'tesseract', 'tools/Tesseract-OCR', '.')]
    found = shutil.which('tesseract')
    if found:
        candidates.append(Path(found))
    candidates.append(Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Tesseract-OCR/tesseract.exe')
    return next((path for path in candidates if path.is_file()), None)


def tesseract_data():
    if os.environ.get('TESSDATA_PREFIX'):
        return Path(os.environ['TESSDATA_PREFIX'])
    command = tesseract_command()
    return command.parent / 'tessdata' if command else None


def tesseract_languages():
    data = tesseract_data()
    return [language for language in ('chi_sim', 'eng')
            if data and (data / (language + '.traineddata')).is_file()]
