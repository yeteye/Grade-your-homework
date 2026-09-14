"""Install official Chinese/English language data next to the local executable."""
from pathlib import Path
import os
import shutil
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homework.ocr_config import tesseract_command, tesseract_data


def install():
    command = tesseract_command()
    if not command or not command.is_file():
        raise SystemExit('Install Tesseract into runtime/tesseract first, or set TESSERACT_CMD.')
    folder = tesseract_data()
    folder.mkdir(parents=True, exist_ok=True)
    for lang in ('chi_sim', 'eng'):
        target = folder / (lang + '.traineddata')
        if target.is_file() and target.stat().st_size > 0:
            print('Already present:', target, flush=True)
            continue
        url = f'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/{lang}.traineddata'
        print('Downloading:', url, flush=True)
        descriptor, temporary = tempfile.mkstemp(dir=folder, suffix='.download')
        try:
            with os.fdopen(descriptor, 'wb') as output, urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output)
            if Path(temporary).stat().st_size < 1000:
                raise RuntimeError('Unexpected language download size')
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print('Installed:', target, flush=True)


if __name__ == '__main__':
    install()
