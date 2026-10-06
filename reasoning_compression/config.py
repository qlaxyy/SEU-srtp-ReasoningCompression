"""Frozen inference settings; no test-set-dependent parameter selection."""
import math
from .io import ROOT, read

METHODS = ('unsteered', 'rebalance', 'l27', 'dynamic32', 'full')
DATASETS = ('math500', 'gsm8k', 'amc23', 'aime2025', 'custom')
MODELS = ('1p5b', '7b')


def make_plan(model, dataset, method, seed=42):
    if model not in MODELS or dataset not in DATASETS or method not in METHODS:
        raise ValueError('Unknown model, dataset or method')
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be in [0, 2**32)')
    small = model == '1p5b'
    cal = read(ROOT / 'assets' / model / 'penalty_calibration.json')
    if not math.isfinite(cal['shape']) or cal['shape'] <= 0:
        raise ValueError('Invalid frozen calibration')
    runtime = dict(max_model_len=32768 if small else 17920,
                   max_num_seqs=256 if small else (64 if dataset == 'gsm8k' else 32),
                   max_num_batched_tokens=32768 if small else 4096,
                   gpu_memory_utilization=.9 if small else .95,
                   chunked_prefill=not small, async_scheduling=small,
                   dtype='bfloat16', seed=seed, temperature=.7, top_p=.95, max_tokens=16000)
    return dict(model='DeepSeek-R1-Distill-Qwen-' + ('1.5B' if small else '7B'),
                model_size=model, dataset=dataset, method=method, seed=seed, cap=16000,
                runtime=runtime, methods=['srq_pca16' if method == 'full' else 'off'],
                answer_strength=.25 if method == 'full' else 0.,
                maximum=1 if method == 'l27' else 32, calibration_shape=cal['shape'],
                gpu_limit_seconds=10800, protocol='public-v0.1',
                historical_result_reproduction='Configuration matched; packaging must pass GPU engineering gates')
