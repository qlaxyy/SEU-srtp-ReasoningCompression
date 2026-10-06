"""Prepare auditable jobs on CPU; run GPU inference only with --execute."""
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from .config import MODELS, DATASETS, METHODS, make_plan
from .io import ROOT, load_questions, read, save, sha


def check_rows(rows, plan, allow_variant=False):
    assets = ROOT / 'assets' / plan['model_size']
    fitted = set(read(assets / 'SRQ_VECTOR_RECEIPT.json')['fit_ids'])
    fitted.update(read(assets / 'penalty_calibration.json')['fit_ids'])
    ids = [r['problem_sha256'] for r in rows]
    if fitted.intersection(ids):
        raise ValueError('Evaluation questions overlap released fitting identities')
    path = ROOT / 'configs/datasets' / (plan['dataset'] + '_identities.json')
    if path.exists():
        expected = read(path)['ordered_problem_sha256']
        if ids != expected and not allow_variant:
            raise ValueError('Dataset text/order differs from historical protocol. Use --allow-data-variant only for explicitly labeled new evaluations.')
    return rows


def prepare_job(args, rows, plan):
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    job = target / 'job'
    job.mkdir()
    shutil.copytree(ROOT / 'assets' / args.model, job / 'assets')
    save(job / 'inputs/pilot.json', rows)
    plan.update(n=len(rows), inputs_sha256=sha(job / 'inputs/pilot.json'), data_variant=args.allow_data_variant)
    save(job / 'PILOT_PLAN.json', plan)
    save(job / 'assets/FROZEN_7B_PROTOCOL.json', plan['runtime'])
    model_identity = read(job / 'assets/model_hashes.json')
    save(job / 'assets/source_identity.json', dict(
        assets=dict(model_path=str(args.model_path.resolve()), model_files=model_identity['files']),
        runtime_hashes=read(ROOT / 'runtime/expected_hashes.json')))
    save(job / 'MANIFEST.json', dict(files={str(p.relative_to(job)).replace('\\', '/'): sha(p)
        for p in sorted(job.rglob('*')) if p.is_file()}))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    versions = {}
    for name in ['torch', 'transformers', 'vllm', 'numpy', 'triton', 'easysteer']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    save(target / 'RUN.json', dict(git_commit=commit, working_tree_dirty=bool(dirty),
        command=sys.argv, python=sys.version, platform=platform.platform(), dependencies=versions,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', 'default'), plan=plan,
        code_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
            for folder in ['runtime', 'reasoning_compression'] for p in sorted((ROOT / folder).rglob('*.py'))}))
    return target, job, commit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--method', choices=METHODS, default='full')
    p.add_argument('--input', type=Path, help='JSON/JSONL with problem text; no gold needed')
    p.add_argument('--model-path', type=Path)
    p.add_argument('--runtime-root', type=Path, default=ROOT / '.runtime')
    p.add_argument('--output', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-samples', '--max_samples', type=int)
    p.add_argument('--allow-data-variant', action='store_true')
    action = p.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    plan = make_plan(a.model, a.dataset, a.method, a.seed)
    rows = check_rows(load_questions(a.input), plan, a.allow_data_variant) if a.input else None
    if a.max_samples is not None:
        if a.max_samples < 1: p.error('--max-samples must be positive')
        if rows is not None: rows = rows[:a.max_samples]
        plan['subset_evaluation'] = True
    if not a.execute:
        print(json.dumps(dict(plan=plan, n=None if rows is None else len(rows), gpu_started=False,
                              data_checked=rows is not None), ensure_ascii=False, indent=2))
        return
    if os.name != 'posix': p.error('GPU execution requires Linux')
    if not a.input or not a.output or not a.model_path: p.error('--execute requires --input, --output and --model-path')
    if sys.flags.optimize: p.error('Do not disable runtime assertion gates with python -O')
    if not a.model_path.is_dir(): p.error('--model-path must be a local model snapshot directory')
    import fcntl
    # One model process at a time per selected CUDA device set.
    import hashlib
    key = hashlib.sha256(os.environ.get('CUDA_VISIBLE_DEVICES', 'default').encode()).hexdigest()[:16]
    lock_path = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'seu-reasoning-compression' / (key + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target, job, commit = prepare_job(a, rows, plan)
        runner = ROOT / 'runtime' / a.model / 'run_pilot.py'
        command = [sys.executable, str(runner), '--execute', '--runtime-root', str(a.runtime_root.resolve()), '--output', str(target / 'generation')]
        if a.method in ('unsteered', 'rebalance'):
            command = [sys.executable, str(ROOT / 'runtime/run_standard.py'), '--method', a.method,
                       '--runtime-root', str(a.runtime_root.resolve()), '--output', str(target / 'generation')]
        env = dict(os.environ, RC_JOB_DIR=str(job), SOURCE_COMMIT=commit,
                   RC_MODEL_SIZE=a.model, GPU_LOCK_FD=str(lock.fileno()), PYTHONUNBUFFERED='1')
        save(target / 'COMMAND.json', command)
        subprocess.run(command, env=env, check=True, pass_fds=(lock.fileno(),))


if __name__ == '__main__':
    main()
