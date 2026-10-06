"""Training-only calibration: original CPU formulas with input validation."""
import numpy as np


def fit_penalty_shape(strengths, questions):
    s = np.asarray(strengths, dtype=np.float64)
    q = np.asarray(questions)
    if s.ndim != 1 or s.shape != q.shape or not len(s):
        raise ValueError('Expected aligned nonempty strengths and question identities')
    if not np.isfinite(s).all() or np.any(s <= 0) or np.any(s > 1):
        raise ValueError('Negative-controller strengths must lie in (0, 1]')
    ids, counts = np.unique(q, return_counts=True)
    counts = dict(zip(ids, counts))
    weights = np.array([1. / counts[x] for x in q])
    order = np.argsort(s, kind='stable')
    threshold = .5 * len(ids) - 1e-12 * max(1, len(ids))
    m = float(s[order[np.searchsorted(np.cumsum(weights[order]), threshold)]])
    if not 1e-4 < m < 1. - 1e-4:
        raise ValueError('Degenerate median; do not silently clamp')
    return dict(median_strength=m, shape=m / (1. - m), questions=len(ids), states=len(s))


def fit_answer_vector(positive, negative, rank=16):
    p, n = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    if p.ndim != 2 or n.ndim != 2 or p.shape[1] != n.shape[1] or min(len(p), len(n)) < 2:
        raise ValueError('Two nonempty feature matrices with a shared hidden dimension required')
    if not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError('Nonfinite hidden states')
    stack = np.concatenate(((p-p.mean(0))/np.sqrt(len(p)), (n-n.mean(0))/np.sqrt(len(n))))
    if not isinstance(rank, int) or not 1 <= rank <= min(stack.shape):
        raise ValueError('Invalid PCA rank')
    _, singular, basis = np.linalg.svd(stack, full_matrices=False)
    if np.sum(singular**2) <= 1e-20:
        raise ValueError('No within-class variation to identify a PCA subspace')
    contrast = p.mean(0)-n.mean(0)
    norm = np.linalg.norm(contrast)
    projected = (contrast @ basis[:rank].T) @ basis[:rank]
    if norm <= 1e-12 or np.linalg.norm(projected) / norm <= 1e-8:
        raise ValueError('Degenerate contrast/projection')
    retained = np.linalg.norm(projected) / norm
    return (projected / retained).astype(np.float32), dict(rank=rank, caa_norm=float(norm),
        projected_contrast_norm_fraction=float(retained),
        top_rank_variance_fraction=float((singular[:rank]**2).sum()/(singular**2).sum()))
