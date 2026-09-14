"""Optional ESRGAN inference reusing the original RRDB architecture.

Requires PyTorch and a separately supplied trusted ESRGAN checkpoint.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upscale an image 4x with the preserved ESRGAN RRDB model.')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    if args.output.exists() or args.input.resolve()==args.output.resolve():
        parser.error('Choose a new output path to preserve existing images.')
    if not args.input.is_file() or not args.weights.is_file():
        parser.error('Input image and model weights must exist.')
    try:
        import torch
    except ImportError:
        parser.error('Optional dependency PyTorch is missing. Install it before using ESRGAN.')
    import numpy as np
    from PIL import Image, ImageOps
    from research.super_resolution.esrgan.rrdb_architecture import RRDBNet
    model = RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
    model.eval().to(args.device)
    with Image.open(args.input) as image:
        pixels = np.asarray(ImageOps.exif_transpose(image).convert('RGB'),dtype=np.float32)/255
    tensor = torch.from_numpy(pixels.transpose(2,0,1).copy()).unsqueeze(0).to(args.device)
    with torch.no_grad():
        result = model(tensor).squeeze(0).clamp(0,1).cpu().numpy().transpose(1,2,0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray((result*255).round().astype('uint8')).save(args.output)
    print(args.output)
