"""Compare two texts through a local engine, without creating a history record."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homework.scoring import lexical_similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two input texts.')
    parser.add_argument('reference')
    parser.add_argument('answer')
    parser.add_argument('--engine', choices=('lexical','transformer'), default='lexical')
    args = parser.parse_args()
    if args.engine == 'transformer':
        from homework.models.transformer import calculate_similarity
        score = calculate_similarity(args.reference, args.answer)
    else:
        score = lexical_similarity(args.reference, args.answer)
    print(json.dumps({'engine':args.engine, 'similarity':round(score, 6)}))
