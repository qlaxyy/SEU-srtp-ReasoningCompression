"""Small, regularized conditional direction bank; NumPy, no model dependency.

This fits a mapping; the base LLM stays frozen. It does NOT estimate correctness.
All transforms are fitted on fit questions only. No outcome gate in version 1.
"""
from dataclasses import dataclass
import numpy as np


def finite_2d(x, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError(name+' must be a nonempty finite matrix')
    return x


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x/np.maximum(n, 1e-12)


def softmax(x):
    e = np.exp(x-x.max(axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)


def spherical_clusters(directions, k, seed):
    """Seeded farthest-first initialization; no repeated best-of-N search."""
    x = unit(directions)
    if np.any(np.linalg.norm(x, axis=1) < .99):
        raise ValueError('Zero response difference is not a direction')
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(x)))]
    for _ in range(1, k):
        distance = 1-(x@x[selected].T).max(axis=1)
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    centers = x[selected].copy()
    old_labels = None
    for _ in range(50):
        labels = (x@centers.T).argmax(axis=1)
        counts = np.bincount(labels, minlength=k)
        if np.any(counts == 0):
            raise ValueError('Empty direction cluster; no automatic retuning of K')
        if old_labels is not None and np.array_equal(old_labels, labels):
            break
        centers = unit(np.stack([x[labels == j].mean(axis=0) for j in range(k)]))
        old_labels = labels.copy()
    return centers, labels


@dataclass
class Mixture:
    center: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    mapper: np.ndarray
    shuffled_mapper: np.ndarray
    prototypes: np.ndarray
    global_direction: np.ndarray
    reference_norm: float
    fit_ids: tuple

    def weights(self, queries, shuffled=False):
        q = finite_2d(queries, 'queries')
        if q.shape[1] != len(self.center):
            raise ValueError('Query dimension mismatch')
        z = ((q-self.center)@self.basis)/self.scale
        z = np.column_stack([np.ones(len(z)), z])
        return softmax(z@(self.shuffled_mapper if shuffled else self.mapper))

    def directions(self, queries, method='conditional'):
        q = finite_2d(queries, 'queries')
        if method == 'off':
            return np.zeros_like(q)
        if method == 'global':
            return np.broadcast_to(self.global_direction, q.shape).copy()
        if method not in ('conditional', 'shuffled'):
            raise ValueError('Unknown method')
        mixed = self.weights(q, method == 'shuffled')@self.prototypes
        return unit(mixed)*self.reference_norm

    def save(self, path):
        with open(path, 'xb') as f:
            np.savez_compressed(f, center=self.center, basis=self.basis, scale=self.scale,
                mapper=self.mapper, shuffled_mapper=self.shuffled_mapper,
                prototypes=self.prototypes, global_direction=self.global_direction,
                reference_norm=np.array(self.reference_norm), fit_ids=np.asarray(self.fit_ids))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            m = cls(**{k: d[k].copy() for k in ('center','basis','scale','mapper',
                'shuffled_mapper','prototypes','global_direction')},
                reference_norm=float(d['reference_norm']), fit_ids=tuple(d['fit_ids'].tolist()))
        for value in (m.center,m.basis,m.scale,m.mapper,m.shuffled_mapper,m.prototypes,m.global_direction):
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite fitted artifact')
        if not np.isfinite(m.reference_norm) or m.reference_norm <= 0 or np.any(m.scale <= 0):
            raise ValueError('Invalid fitted scale')
        if m.basis.shape[0] != len(m.center) or m.prototypes.shape[1] != len(m.center):
            raise ValueError('Fitted artifact dimension mismatch')
        if m.mapper.shape != (m.basis.shape[1]+1, len(m.prototypes)) or m.mapper.shape != m.shuffled_mapper.shape:
            raise ValueError('Mapper dimension mismatch')
        if m.global_direction.shape != m.center.shape or m.scale.shape != (m.basis.shape[1],):
            raise ValueError('Scale dimension mismatch')
        return m


def fit(queries, differences, ids, *, k=4, rank=16, ridge=10., seed=42,
        minimum_cluster=8, minimum_questions=48):
    q, delta = finite_2d(queries, 'queries'), finite_2d(differences, 'differences')
    if q.shape != delta.shape or len(ids) != len(q) or len(set(ids)) != len(ids):
        raise ValueError('Requires one paired row per distinct training question')
    if len(q) < minimum_questions or k < 2 or rank < 1 or ridge <= 0:
        raise ValueError('Insufficient fit data or invalid fixed hyperparameters')
    center = q.mean(axis=0)
    _, s, vt = np.linalg.svd(q-center, full_matrices=False)
    usable = min(rank, int((s > s[0]*1e-8).sum()), len(q)-1)
    if usable != rank:
        raise ValueError('Query rank insufficient; do not silently change the experiment')
    basis = vt[:rank].T
    z = (q-center)@basis
    scale = z.std(axis=0, ddof=1)
    z = np.column_stack([np.ones(len(q)), z/scale])
    prototypes, labels = spherical_clusters(delta, k, seed)
    counts = np.bincount(labels, minlength=k)
    if np.any(counts < minimum_cluster):
        raise ValueError('Direction cluster too small; collect data or reject version')
    targets = 4.*(unit(delta)@prototypes.T)
    targets -= targets.mean(axis=1, keepdims=True)
    penalty = np.eye(rank+1)*ridge
    penalty[0, 0] = 0.
    normal = z.T@z+penalty
    mapper = np.linalg.solve(normal, z.T@targets)
    # Same direction bank, same fit questions, shuffled input/target association.
    permutation = np.random.default_rng(seed+1).permutation(len(q))
    shuffled_mapper = np.linalg.solve(normal, z.T@targets[permutation])
    global_direction = delta.mean(axis=0)
    ref = float(np.linalg.norm(global_direction))
    if ref <= 1e-8:
        raise ValueError('Global control is degenerate; cannot match intervention magnitude')
    fitted = Mixture(center, basis, scale, mapper, shuffled_mapper, prototypes,
                     global_direction, ref, tuple(ids))
    report = dict(fit_questions=len(q), hidden_dim=q.shape[1], k=k, rank=rank,
        ridge=ridge, seed=seed, cluster_counts=counts.tolist(), reference_norm=ref,
        mapper_parameters=int(mapper.size), base_model_trained=False,
        fitted_statistics_parameters=int(center.size+basis.size+scale.size+prototypes.size),
        no_correctness_predictor=True, no_benefit_gate=True)
    return fitted, report

