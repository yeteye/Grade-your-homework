"""Interpolate two matching RRDB checkpoints into a new file."""
import argparse
from pathlib import Path
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('psnr', type=Path)
    parser.add_argument('esrgan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error('alpha must be in [0, 1].')
    if args.output.exists():
        parser.error('Output already exists; choose a new path.')
    first = torch.load(args.psnr, map_location='cpu', weights_only=True)
    second = torch.load(args.esrgan, map_location='cpu', weights_only=True)
    if first.keys() != second.keys():
        parser.error('Checkpoint parameter names do not match.')
    merged = {}
    for key in first:
        if first[key].shape != second[key].shape:
            parser.error('Parameter shapes do not match: ' + key)
        merged[key] = (1 - args.alpha) * first[key] + args.alpha * second[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(args.output)
