import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from reasoning_compression.io import ROOT, load_questions, problem_id, read, sha
from reasoning_compression.config import make_plan, METHODS
from reasoning_compression.cli import check_rows, prepare_job
from reasoning_compression.calibration import fit_answer_vector, fit_penalty_shape


def test_frozen_assets():
    for model, dim in [('1p5b', 1536), ('7b', 3584)]:
        root = ROOT / 'assets' / model
        receipt = read(root / 'SRQ_VECTOR_RECEIPT.json')
        vector = np.load(root / 'pca_mean16.npy', allow_pickle=False)
        assert vector.shape == (dim,) and np.isfinite(vector).all()
        assert sha(root / 'pca_mean16.npy') == receipt['vector_sha256']
        assert sha(root / 'original_vector.pt') == read(root / 'original_fit.json')['vector_sha256']
        assert len(set(receipt['fit_ids'])) == 271
        cal = read(root / 'penalty_calibration.json')
        assert len(set(cal['fit_ids'])) == 300
        assert cal['shape'] == pytest.approx(cal['median_strength'] / (1-cal['median_strength']))


def test_reference_pca_matches_explicit_cartesian_difference():
    rng = np.random.default_rng(4)
    pos, neg = rng.normal(size=(9, 7)), rng.normal(size=(5, 7))
    pos[:, :2] += 1.
    vector, stats = fit_answer_vector(pos, neg, rank=3)
    differences = (pos[:, None, :] - neg[None, :, :]).reshape(-1, 7)
    _, _, basis = np.linalg.svd(differences-differences.mean(0), full_matrices=False)
    contrast = pos.mean(0)-neg.mean(0)
    projected = contrast @ basis[:3].T @ basis[:3]
    expected = projected / np.linalg.norm(projected)*np.linalg.norm(contrast)
    np.testing.assert_allclose(vector, expected, rtol=1e-6, atol=1e-6)
    assert np.linalg.norm(vector) == pytest.approx(stats['caa_norm'])


def test_calibration_balances_questions_not_number_of_steps():
    a = fit_penalty_shape([.2, .3, .7], ['a', 'a', 'b'])
    b = fit_penalty_shape([.2, .3]*40 + [.7], ['a']*80 + ['b'])
    assert a['shape'] == b['shape']
    m, rho = a['median_strength'], a['shape']
    assert m/(m+rho*(1-m)) == pytest.approx(.5)
    with pytest.raises(ValueError): fit_penalty_shape([float('nan')], ['a'])
    with pytest.raises(ValueError): fit_penalty_shape([1.], ['a'])


def test_questions_strip_gold_and_reject_duplicates(tmp_path):
    p = tmp_path / 'input.json'
    p.write_text(json.dumps([dict(problem='3 + 4', answer='7', private_metadata='not an input')]))
    rows = load_questions(p)
    assert set(rows[0]) == {'dataset_index', 'problem_sha256', 'problem', 'source'}
    assert problem_id('３ +\n 4') == problem_id('3+4')
    p.write_text(json.dumps([dict(problem='3+4'), dict(problem='3 + 4')]))
    with pytest.raises(ValueError): load_questions(p)


def test_fitting_overlap_and_dataset_mismatch_rejected():
    plan = make_plan('1p5b', 'custom', 'full')
    key = read(ROOT / 'assets/1p5b/SRQ_VECTOR_RECEIPT.json')['fit_ids'][0]
    with pytest.raises(ValueError): check_rows([dict(problem_sha256=key)], plan)
    with pytest.raises(ValueError): check_rows(load_questions(ROOT / 'examples/questions.jsonl'), make_plan('7b', 'math500', 'full'))


def test_all_protocol_dry_runs_import_no_gpu_modules():
    for model in ['1p5b', '7b']:
        for method in METHODS:
            result = subprocess.run([sys.executable, '-m', 'reasoning_compression.cli', '--model', model,
                '--dataset', 'custom', '--method', method, '--input', str(ROOT/'examples/questions.jsonl'), '--dry-run'],
                cwd=ROOT, capture_output=True, text=True, check=True)
            output = json.loads(result.stdout)
            assert output['n'] == 2 and output['gpu_started'] is False
            assert output['plan']['runtime']['max_tokens'] == 16000


def test_job_export_and_native_runner_dry_run(tmp_path):
    from types import SimpleNamespace
    import os
    for model in ['1p5b', '7b']:
        args = SimpleNamespace(output=tmp_path/model, model=model, model_path=tmp_path/'weights', allow_data_variant=False)
        plan = make_plan(model, 'custom', 'full')
        rows = load_questions(ROOT/'examples/questions.jsonl')
        target, job, commit = prepare_job(args, rows, plan)
        exported = read(job/'inputs/pilot.json')
        assert len(exported) == 2 and not any('gold' in r for r in exported)
        for n,h in read(job/'MANIFEST.json')['files'].items(): assert sha(job/n) == h
        result = subprocess.run([sys.executable, str(ROOT/'runtime'/model/'run_pilot.py'), '--dry-run', '--runtime-root', str(tmp_path)],
            env=dict(os.environ, RC_JOB_DIR=str(job)), capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_result_accounting_rejects_duplicates_and_missing_tokens():
    spec = importlib.util.spec_from_file_location('public_grade', ROOT/'scripts/grade.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    r = dict(problem_sha256='a', token_ids=[1, 2], tokens=2, capped=False)
    assert mod.validate_predictions([r], ['a'], 10) == [r]
    with pytest.raises(ValueError): mod.validate_predictions([r, r], ['a'], 10)
    with pytest.raises(ValueError): mod.validate_predictions([dict(r, tokens=1)], ['a'], 10)


def test_lexical_opening_and_multitoken_phrase():
    from reasoning_compression.lexicon import Automaton, expanded_terms, compact_tables
    words = read(ROOT/'configs/l27_terms.json')['terms']
    assert len(words)==27
    matcher=Automaton(expanded_terms(words),opening=True)
    pieces=['Wait',' when','think',' again','The result is wait',' carefully','waiting']
    tr,hits=matcher.compile_tokens(pieces)
    tables=compact_tables(tr,hits,matcher.final,True)
    assert hits[0,0] and not hits[0,1]
    assert not hits[0,2] and hits[tr[0,2],3]
    assert not hits[0,4] and hits[0,5] and not hits[0,6]
    np.testing.assert_array_equal(tables['transition'][:,tables['token_classes']],tr)
    np.testing.assert_array_equal(tables['hits'][:,tables['token_classes']],hits)
