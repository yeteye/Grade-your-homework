"""Read-only environment check. Does not call paid APIs or download models."""
import importlib.util
import os
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
print('Python:', sys.version.split()[0])
print('Interpreter:', sys.executable)
for module in ('flask', 'waitress', 'PIL', 'rapidocr_onnxruntime', 'paddle'):
    print(f'{module}:', 'installed' if importlib.util.find_spec(module) else 'missing')
from homework.ocr import details
from homework.ocr_config import tesseract_command, paddle_model_root
for engine, note in details().items():
    print(f'{engine}: {note}')
print('Tesseract executable:', tesseract_command() or 'not found')
print('PaddleOCR model directory:', paddle_model_root())
model = Path(os.environ.get('HOMEWORK_MODEL_PATH', root / 'models/transformer/model_best.pdparams'))
print('Model weights:', 'found' if model.is_file() else 'missing')
print('DeepSeek key:', 'configured (not checked)' if os.environ.get('DEEPSEEK_API_KEY') else 'not configured (optional)')
print('URL: http://127.0.0.1:' + os.environ.get('PORT', '5000'))
