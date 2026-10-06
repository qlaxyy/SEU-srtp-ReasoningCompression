"""Collect successful scored runs and compare only matching dataset identities/seeds."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('outputs'))
    p.add_argument('--output', type=Path, default=Path('outputs/summary.csv'))
    a = p.parse_args()
    rows = []
    for path in sorted(a.runs.rglob('SUMMARY.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        if 'questions_sha256' not in r: continue
        rows.append(dict(r, run=str(path.parent)))
    keys = lambda r: (r['model'], r['dataset'], r['seed'], r['questions_sha256'])
    baselines = {}
    for r in rows:
        if r['method'] in ('unsteered', 'rebalance'):
            key = (keys(r), r['method'])
            if key in baselines: raise ValueError(f'Duplicate baseline for {key}; use a narrower --runs directory')
            baselines[key] = r
    for r in rows:
        unsteered = baselines.get((keys(r), 'unsteered'))
        rebalance = baselines.get((keys(r), 'rebalance'))
        r['compression_vs_unsteered_percent'] = '' if unsteered is None else 100*(1-r['total_tokens']/unsteered['total_tokens'])
        r['accuracy_vs_rebalance_pp'] = '' if rebalance is None else 100*(r['accuracy']-rebalance['accuracy'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)
    print(f'Summarized {len(rows)} runs into {a.output}')


if __name__ == '__main__':
    main()
