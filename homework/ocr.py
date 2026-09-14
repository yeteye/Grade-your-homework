"""Validated temporary images and lazily loaded OCR adapters."""
import importlib.util
import subprocess
import tempfile
import threading
import warnings
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from .validation import ValidationError
from .ai_grader import GradingUnavailable
from .preprocessing import MODES, prepare_image
from .ocr_config import (PADDLE_MODELS, paddle_model_root, paddle_models_ready,
                         configure_paddle_cache, tesseract_command, tesseract_data,
                         tesseract_languages)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000
_lock = threading.Lock()
_engines = {}


def allowed_file(filename):
    return isinstance(filename, str) and '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def status():
    def installed(module):
        return importlib.util.find_spec(module) is not None
    tess = tesseract_command()
    return {'RapidOCR': installed('rapidocr_onnxruntime'),
        'PaddleOCR': installed('paddleocr') and installed('paddle') and paddle_models_ready(),
        'Tesseract': bool(tess and tess.is_file() and tesseract_languages())}


def details():
    available = status()
    langs = tesseract_languages()
    return {'RapidOCR': '已配置 · 中英文' if available['RapidOCR'] else '未安装依赖',
            'PaddleOCR': '本地模型已就绪 · 中英文 · CPU' if available['PaddleOCR'] else '请运行 setup-paddleocr.bat 安装依赖与模型',
            'Tesseract': '已配置 · ' + ' / '.join({'chi_sim': '中文', 'eng': '英文'}[x] for x in langs)
            if available['Tesseract'] else '未找到引擎或语言包'}


def save_image(upload, path, preprocessing='original'):
    if not upload or not allowed_file(upload.filename):
        raise ValidationError('请上传 JPG、PNG、BMP、TIFF 或 WebP 图片。')
    raw = upload.stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValidationError('单张图片不能超过 8 MB。')
    from io import BytesIO
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise ValidationError('图片不能超过 2000 万像素。')
                if source.format not in ('JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP'):
                    raise ValidationError('文件内容不是支持的图片格式。')
                source.load()
                prepare_image(ImageOps.exif_transpose(source), preprocessing).save(path, 'PNG')
    except ValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValidationError('图片已损坏、格式错误或尺寸过大。') from None


def recognize(path, engine, language):
    if engine == 'RapidOCR':
        if engine not in _engines:
            from rapidocr_onnxruntime import RapidOCR
            _engines[engine] = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
        result, _ = _engines[engine](str(path))
        return '\n'.join(str(line[1]) for line in (result or []))
    if engine == 'PaddleOCR':
        # The multilingual recognition model handles both UI language choices.
        key = (engine, str(paddle_model_root()))
        if key not in _engines:
            if not paddle_models_ready():
                raise GradingUnavailable('缺少 PaddleOCR 模型，请运行 setup-paddleocr.bat。')
            configure_paddle_cache()
            from paddleocr import PaddleOCR
            det, rec = PADDLE_MODELS
            _engines[key] = PaddleOCR(
                text_detection_model_name=det, text_detection_model_dir=str(paddle_model_root() / det),
                text_recognition_model_name=rec, text_recognition_model_dir=str(paddle_model_root() / rec),
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, device='cpu', cpu_threads=2, enable_mkldnn=False)
        return '\n'.join(str(line) for page in _engines[key].predict(str(path)) for line in page['rec_texts'])
    lang = 'chi_sim' if language == '中文' else 'eng'
    if lang not in tesseract_languages():
        raise GradingUnavailable(f'Tesseract 缺少 {lang}.traineddata 语言包，请放入 tessdata 目录。')
    # Pass each argument separately: pytesseract's Windows config parser preserves
    # literal quotes around tessdata paths and fails for directories with spaces.
    result = subprocess.run([str(tesseract_command()), str(path), 'stdout',
        '--tessdata-dir', str(tesseract_data()), '-l', lang], capture_output=True,
        encoding='utf-8', timeout=30, check=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return result.stdout.strip()


def recognize_pair(file1, file2, engine, language, temp_root, preprocessing='original'):
    if preprocessing not in MODES:
        raise ValidationError('不支持的图像预处理方式。')
    if engine not in ('RapidOCR', 'PaddleOCR', 'Tesseract'):
        raise ValidationError('无效的 OCR 引擎。')
    if language not in ('中文', '英文'):
        raise ValidationError('请选择中文或英文。')
    with tempfile.TemporaryDirectory(dir=temp_root, prefix='ocr-') as folder:
        paths = [Path(folder) / 'work.png', Path(folder) / 'answer.png']
        for upload, path in zip((file1, file2), paths):
            save_image(upload, path, preprocessing)
        if not status()[engine]:
            raise GradingUnavailable(f'{engine} 尚未就绪，请检查环境状态或改用文本输入。')
        try:
            with _lock:
                work, answer = [recognize(p, engine, language) for p in paths]
        except GradingUnavailable:
            raise
        except Exception as exc:
            raise GradingUnavailable('OCR 识别失败，请检查引擎、语言包或图片内容。') from exc
    notes = [] if work.strip() and answer.strip() else ['部分图片未识别到文字，请手动补充后批改。']
    return {'workContent': work, 'answerContent': answer, 'warnings': notes}
