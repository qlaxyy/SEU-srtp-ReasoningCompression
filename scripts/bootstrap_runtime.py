"""Fetch pinned upstream code and apply audited overlays; no models or GPU calls."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAMS = {
    'vllm': ('https://github.com/ZJU-REAL/EasySteer-vllm-v1.git', '6267ca0cfc9c6e93b1427d36b1d655821d6d6f9b', 'sources/EasySteer/vllm-steer'),
}


def run(args, cwd):
    subprocess.run(args, cwd=cwd, check=True, env=dict(os.environ, GIT_LFS_SKIP_SMUDGE='1'))


def verify(dest):
    expected = json.loads((ROOT / 'runtime/expected_hashes.json').read_text())
    bad = []
    for name, h in expected.items():
        p = dest / name
        if not p.is_file() or hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != h:
            bad.append(name)
    if bad:
        raise ValueError('Runtime snapshot mismatch: ' + ', '.join(bad[:20]))
    print(f'Runtime verified: {len(expected)} source hashes match the final experiment snapshot.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dest', type=Path, default=ROOT / '.runtime')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(dict(upstreams=UPSTREAMS, dest=str(a.dest), downloads_model_weights=False), indent=2))
        return
    dest = a.dest.resolve()
    if a.verify_only:
        verify(dest)
        return
    # Bundle only the small, hash-verified EasySteer Python library, not its
    # checkpoints, notebooks, frontend or example outputs.
    for f in (ROOT / 'third_party/EasySteer').rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts:
            out = dest / 'sources/EasySteer' / f.relative_to(ROOT / 'third_party/EasySteer')
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and out.read_bytes().replace(b'\r\n', b'\n') != f.read_bytes().replace(b'\r\n', b'\n'):
                raise ValueError(f'Existing EasySteer file differs: {out}; use a new --dest')
            shutil.copyfile(f, out)
    for label, (url, commit, rel) in UPSTREAMS.items():
        path = dest / rel
        path.mkdir(parents=True, exist_ok=True)
        if not (path / '.git').is_dir():
            if list(path.iterdir()):
                raise ValueError(f'Refusing to initialize nonempty directory {path}')
            run(['git', 'init'], path)
            run(['git', 'remote', 'add', 'origin', url], path)
            run(['git', 'fetch', '--depth', '1', 'origin', commit], path)
            run(['git', '-c', 'core.autocrlf=false', 'checkout', '--detach', 'FETCH_HEAD'], path)
        else:
            actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
            if actual != commit:
                raise ValueError(f'Existing checkout must be at {commit}; use a new --dest')
        for f in (ROOT / 'runtime/overlays' / label).rglob('*'):
            if f.is_file():
                out = path / f.relative_to(ROOT / 'runtime/overlays' / label)
                out.parent.mkdir(parents=True, exist_ok=True)
                # Do not overwrite unknown user edits on repeat setup.
                if out.exists() and out.read_bytes().replace(b'\r\n', b'\n') != f.read_bytes():
                    tracked_file = subprocess.run(['git', 'ls-files', '--error-unmatch', '--', str(out.relative_to(path))],
                                                  cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if tracked_file.returncode != 0:
                        raise ValueError(f'Untracked overlay destination already exists: {out}')
                    tracked = subprocess.run(['git', 'diff', '--quiet', '--', str(out.relative_to(path))], cwd=path)
                    if tracked.returncode != 0:
                        raise ValueError(f'Locally edited overlay destination: {out}')
                shutil.copyfile(f, out)
    verify(dest)


if __name__ == '__main__':
    main()
