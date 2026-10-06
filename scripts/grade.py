"""Grade completed generations with the preserved ReBalance parser and grader."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reasoning_compression.io import read, save, sha


def validate_predictions(rows, expected, cap):
    ids = [r['problem_sha256'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Missing, duplicate, or unexpected predictions')
    for r in rows:
        n = len(r['token_ids'])
        if not 0 < n <= cap or n != r['tokens'] or r['capped'] != (n >= cap):
            raise ValueError('Invalid token accounting')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='Output root containing RUN.json and generation/')
    p.add_argument('--gold', type=Path, required=True)
    a = p.parse_args()
    plan = read(a.run / 'job/PILOT_PLAN.json')
    complete = read(a.run / 'generation/COMPLETE.json')
    if not complete.get('passed') or complete['manifest_sha256'] != sha(a.run / 'job/MANIFEST.json'):
        raise ValueError('Generation incomplete or from a different input manifest')
    method = plan['method']
    name = method if method in ('unsteered', 'rebalance') else plan['methods'][0]
    raw = [json.loads(s) for s in (a.run / 'generation' / (name+'.jsonl')).read_text(encoding='utf-8').splitlines() if s.strip()]
    expected = [r['problem_sha256'] for r in read(a.run / 'job/inputs/pilot.json')]
    rows = validate_predictions(raw, expected, plan['cap'])
    gold_rows = read(a.gold)
    gold = {r['problem_sha256']: r['gold'] for r in gold_rows}
    if len(gold) != len(gold_rows) or not set(expected).issubset(gold):
        raise ValueError('Duplicate or missing reference answers')
    sys.path.insert(0, str(ROOT / 'third_party/rebalance'))
    from utils.parser import extract_answer
    from utils.grader import check_is_correct
    labels = []
    for r in rows:
        answer = extract_answer(r['text'])
        labels.append(dict(problem_sha256=r['problem_sha256'], dataset_index=r['dataset_index'],
                           prediction=answer, correct=bool(check_is_correct(answer, str(gold[r['problem_sha256']]))),
                           tokens=r['tokens'], thinking_tokens=r['thinking_tokens'],
                           answer_tokens=r['answer_tokens'], capped=r['capped']))
    result = dict(model=plan['model_size'], dataset=plan['dataset'], method=method, seed=plan['seed'], n=len(labels),
                  correct=sum(r['correct'] for r in labels), total_tokens=sum(r['tokens'] for r in labels),
                  thinking_tokens=sum(r['thinking_tokens'] for r in labels), answer_tokens=sum(r['answer_tokens'] for r in labels),
                  capped=sum(r['capped'] for r in labels), data_variant=plan.get('data_variant', False),
                  subset=plan.get('subset_evaluation', False), questions_sha256=sha(a.run / 'job/inputs/pilot.json'),
                  predictions_sha256=sha(a.run / 'generation' / (name+'.jsonl')), gold_sha256=sha(a.gold),
                  grader_sha256=sha(ROOT / 'third_party/rebalance/utils/grader.py'))
    result['accuracy'] = result['correct'] / result['n']
    result['mean_tokens'] = result['total_tokens'] / result['n']
    save(a.run / 'LABELS.json', labels)
    save(a.run / 'SUMMARY.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
