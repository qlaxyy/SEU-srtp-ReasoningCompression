"""Experiment-local correction for steering workload missing from compile hash.

No edits to the shared runtime or deletion of earlier compiled artifacts.
"""
import hashlib
import json


def workload_hash(config, original_hash):
    algorithms = config.algorithms
    if isinstance(algorithms, list):
        algorithms = sorted(set(algorithms))
    text = json.dumps(dict(version='srq_answer_workload_identity_v1',
        original=original_hash, algorithms=algorithms, multi_vector=config.multi_vector),
        sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def install():
    from vllm.config import SteerVectorConfig
    original = SteerVectorConfig.compute_hash
    def compute_hash(self):
        return workload_hash(self, original(self))
    SteerVectorConfig.compute_hash = compute_hash
    return dict(scope='Current process only; shared runtime files untouched',
        reason='Pinned SteerVectorConfig.compute_hash omits algorithms/multi_vector',
        correction='Include normalized workload in cache identity; keep inference configuration')
