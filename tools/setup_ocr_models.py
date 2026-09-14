"""Download official PaddleOCR CPU models once; application inference stays local."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homework.ocr_config import PADDLE_MODELS, MODEL_FILES, paddle_model_root

BASE_URL = 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/'
ARCHIVE_HASHES = {
    'PP-OCRv5_mobile_det': '50446e5d01ac2a73d5319c89513281f6578414c888c602f9af13f93feefffc58',
    'PP-OCRv5_mobile_rec': '566b9512b34e34a9f0db54d87b51fa5a0b9ed2cf1ab7e49728cc0b8b5a64f414',
}


def install():
    root = paddle_model_root()
    root.mkdir(parents=True, exist_ok=True)
    for name in PADDLE_MODELS:
        target = root / name
        if all((target / filename).is_file() for filename in MODEL_FILES):
            print('Already present:', name, flush=True)
            continue
        if target.exists():
            raise RuntimeError(f'Incomplete model directory: {target}. Rename it before retrying.')
        url = BASE_URL + name + '_infer.tar'
        print('Downloading:', url, flush=True)
        with tempfile.TemporaryDirectory(prefix='download-', dir=root) as temporary:
            staging = Path(temporary)
            archive = staging / 'model.tar'
            with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as output:
                shutil.copyfileobj(response, output)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != ARCHIVE_HASHES[name]:
                raise RuntimeError(f'Download checksum mismatch: {name}')
            with tarfile.open(archive) as bundle:
                bundle.extractall(staging / 'unpacked', filter='data')
            configs = list((staging / 'unpacked').rglob('inference.yml'))
            if len(configs) != 1 or not all((configs[0].parent / f).is_file() for f in MODEL_FILES):
                raise RuntimeError('The model archive does not contain the expected inference files.')
            source = configs[0].parent
            manifest = {'source': url, 'archiveSha256': digest,
                        'files': {f: hashlib.sha256((source / f).read_bytes()).hexdigest() for f in MODEL_FILES}}
            (source / 'download.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            # Copy out of TemporaryDirectory so Windows inherits the project's
            # ACL instead of retaining the temporary folder's owner-only ACL.
            shutil.copytree(source, target)
        print('Installed:', target, flush=True)
    print('PaddleOCR models ready. Start run.bat and select PaddleOCR in Image recognition.', flush=True)


if __name__ == '__main__':
    install()
