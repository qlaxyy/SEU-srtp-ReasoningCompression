"""Download public benchmark text into ignored data/, separating questions and gold."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reasoning_compression.io import problem_id, save, ROOT, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['math500', 'gsm8k', 'amc23'], required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'data')
    p.add_argument('--revision', help='Optional immutable Hugging Face dataset revision')
    a = p.parse_args()
    from datasets import load_dataset
    spec = {'math500': ('HuggingFaceH4/MATH-500', None, 'test'),
            'gsm8k': ('openai/gsm8k', 'main', 'test'),
            'amc23': ('math-ai/amc23', None, 'test')}[a.dataset]
    ds = load_dataset(spec[0], spec[1], split=spec[2], revision=a.revision)
    rows, gold = [], []
    for i, r in enumerate(ds):
        text = r.get('problem', r.get('question'))
        answer = str(r['answer']).split('####')[-1].strip()
        key = problem_id(text)
        rows.append(dict(dataset_index=i, problem=text, problem_sha256=key, source=spec[0]))
        gold.append(dict(problem_sha256=key, gold=answer))
    expected = read(ROOT / 'configs/datasets' / (a.dataset + '_identities.json'))['ordered_problem_sha256']
    if {r['problem_sha256'] for r in rows} != set(expected) or len(rows) != len(expected):
        raise ValueError('Downloaded benchmark differs from historical question identities; review the dataset revision')
    by_id = {r['problem_sha256']: r for r in rows}
    rows = [dict(by_id[k], dataset_index=i) for i, k in enumerate(expected)]
    target = a.output / a.dataset
    target.mkdir(parents=True, exist_ok=False)
    save(target / 'questions.json', rows)
    save(target / 'gold.json', gold)
    save(target / 'DATA_SOURCE.json', dict(repo=spec[0], config=spec[1], split=spec[2],
        requested_revision=a.revision, dataset_fingerprint=ds._fingerprint, n=len(rows), historical_question_hashes_match=True))
    print(f'Saved {len(rows)} questions and separate gold records to {target}')


if __name__ == '__main__':
    main()
