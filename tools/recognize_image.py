"""Recognize a local image using the same configured engines as the web app."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homework.ocr import recognize, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--engine', choices=('RapidOCR', 'PaddleOCR', 'Tesseract'), default='PaddleOCR')
    parser.add_argument('--language', choices=('zh', 'en'), default='zh')
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error('Image file does not exist.')
    if not status()[args.engine]:
        parser.error(f'{args.engine} is not configured. Run tools/check_environment.py.')
    sys.stdout.reconfigure(encoding='utf-8')
    print(recognize(args.image.resolve(), args.engine, '中文' if args.language == 'zh' else '英文'))


if __name__ == '__main__':
    main()
