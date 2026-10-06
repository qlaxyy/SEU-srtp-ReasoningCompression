"""Fit calibration assets from caller-supplied training-only states, on CPU."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reasoning_compression.calibration import fit_answer_vector, fit_penalty_shape
from reasoning_compression.io import ROOT, read, save, sha
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['penalty', 'answer'], required=True)
    p.add_argument('--input', type=Path, required=True, help='NPZ; see docs/calibration.md')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, default=16)
    a = p.parse_args()
    test_ids = set()
    for path in (ROOT / 'configs/datasets').glob('*_identities.json'):
        test_ids.update(read(path)['ordered_problem_sha256'])
    with np.load(a.input, allow_pickle=False) as z:
        fit_ids = z['question_ids'].astype(str)
        if not len(fit_ids) or any(len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in fit_ids):
            raise ValueError('Expected normalized SHA-256 training question identities')
        if test_ids.intersection(fit_ids):
            raise ValueError('Fitting/evaluation overlap')
        if a.kind == 'penalty':
            receipt = fit_penalty_shape(z['strengths'], fit_ids)
            vector = None
        else:
            x, positive = z['features'], z['positive']
            if len(x) != len(fit_ids) or positive.dtype != np.bool_ or positive.shape != (len(x),):
                raise ValueError('Aligned per-trajectory features and Boolean SRQ selection labels required')
            vector, receipt = fit_answer_vector(x[positive], x[~positive], a.rank)
    a.output.mkdir(parents=True, exist_ok=False)
    if vector is not None:
        np.save(a.output / 'pca_mean16.npy', vector, allow_pickle=False)
        receipt['vector_sha256'] = sha(a.output / 'pca_mean16.npy')
    receipt.update(input_sha256=sha(a.input), fit_ids=sorted(set(fit_ids)), kind=a.kind,
                   gpu_used=False, benchmark_outcomes_used=False)
    save(a.output / 'FIT.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != 'fit_ids'}, indent=2))


if __name__ == '__main__':
    main()
