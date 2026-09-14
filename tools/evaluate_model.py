"""Dataset-level classification metrics, adapted from the legacy HanLP CSV tool."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homework.scoring import lexical_similarity


def metrics(tp, tn, fp, fn):
    def divide(a, b):
        return a / b if b else 0.0
    precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
    return {'samples':tp+tn+fp+fn, 'accuracy':divide(tp+tn,tp+tn+fp+fn),
            'precision':precision, 'recall':recall, 'f1':divide(2*precision*recall,precision+recall),
            'false_positive_rate':divide(fp,fp+tn), 'false_negative_rate':divide(fn,fn+tp),
            'confusion_matrix':{'tp':tp,'tn':tn,'fp':fp,'fn':fn}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate binary text matching on a labelled TSV dataset.')
    parser.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parents[1]/'datasets/lcqmc/test.tsv')
    parser.add_argument('--engine', choices=('lexical','transformer'), default='lexical')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.limit < 1 or not 0 <= args.threshold <= 1:
        parser.error('limit must be positive; threshold must be between 0 and 1.')
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new filename.')
    score = lexical_similarity
    if args.engine == 'transformer':
        from homework.models.transformer import calculate_similarity
        score = calculate_similarity
    counts = {'tp':0,'tn':0,'fp':0,'fn':0}
    with args.dataset.open(encoding='utf-8-sig',newline='') as stream:
        for index,row in enumerate(csv.reader(stream,delimiter='\t')):
            if index >= args.limit:
                break
            if len(row)!=3 or row[2] not in ('0','1'):
                parser.error(f'Invalid labelled TSV row {index+1}.')
            predicted, actual = score(row[0],row[1]) >= args.threshold, row[2]=='1'
            counts[('t' if predicted==actual else 'f')+('p' if predicted else 'n')] += 1
    result = {'engine':args.engine,'threshold':args.threshold,'dataset':str(args.dataset),**metrics(**counts)}
    content = json.dumps(result, ensure_ascii=False, indent=2)
    sys.stdout.reconfigure(encoding='utf-8')
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(content,encoding='utf-8')
