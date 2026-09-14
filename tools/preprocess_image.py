"""Usage: python tools/preprocess_image.py input.png output.png --mode binary"""
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageOps
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homework.preprocessing import prepare_image, MODES


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess an image using the same code as the OCR service.')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mode', choices=MODES, default='binary')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Choose a separate output path to preserve the original image.')
    if args.output.exists():
        parser.error('Output already exists; choose another filename.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.input) as image:
        prepare_image(ImageOps.exif_transpose(image), args.mode).save(args.output)
    print(args.output)
