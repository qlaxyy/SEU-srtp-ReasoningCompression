"""Read-only release checks over tracked and unignored files; never prints secret values."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    files = [ROOT / n for n in sorted(set(names.decode().split('\0'))) if n and (ROOT/n).is_file()]
    bad = []
    patterns = {
        'api_key': re.compile(r'\bsk-[A-Za-z0-9_-]{20,}'),
        'github_token': re.compile(r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}'),
        'private_key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
        'private_deployment': re.compile(r'(?:connect\.[a-z0-9]+\.seetacloud\.com|/root/autodl-tmp|[Ee]:[/\\]srtp|[Cc]:[/\\]Users[/\\])'),
    }
    for p in files:
        rel = str(p.relative_to(ROOT)).replace('\\','/')
        if p.stat().st_size > 2*1024*1024: bad.append((rel,'unexpected large file'))
        if p.suffix in ['.pt','.npy','.npz']: continue
        try: text = p.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            bad.append((rel,'unreviewed binary'));continue
        if rel != 'scripts/check_release.py':
            for label, pattern in patterns.items():
                if pattern.search(text): bad.append((rel,label))
        if p.suffix == '.py':
            try: ast.parse(text, filename=rel)
            except SyntaxError: bad.append((rel,'syntax error'))
    for rel, record in json.loads((ROOT/'runtime/provenance.json').read_text()).items():
        if record.get('portable_wrapper_changes') is False:
            b=(ROOT/rel).read_bytes().replace(b'\r\n',b'\n')
            if hashlib.sha256(b).hexdigest()!=record['historical_sha256']:bad.append((rel,'historical core changed'))
    report=dict(files=len(files),bytes=sum(p.stat().st_size for p in files),issues=bad,
                gpu_validation_performed=False)
    print(json.dumps(report,indent=2))
    if bad:raise SystemExit(1)


if __name__=='__main__':main()
