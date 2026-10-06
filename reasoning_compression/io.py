"""Portable identities and strict evaluation input contracts."""
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def problem_id(text):
    return hashlib.sha256(''.join(unicodedata.normalize('NFKC', text).split()).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def load_questions(path):
    """Discard non-input columns: reference answers never enter the inference job."""
    path = Path(path)
    raw = read(path) if path.suffix == '.json' else [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not isinstance(raw, list) or not raw:
        raise ValueError('Expected a nonempty JSON list or JSONL dataset')
    rows = []
    for i, r in enumerate(raw):
        text = r.get('problem', r.get('question'))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Invalid problem at row {i}')
        key = problem_id(text)
        if r.get('problem_sha256', key) != key:
            raise ValueError(f'Problem identity mismatch at row {i}')
        rows.append(dict(dataset_index=i, problem_sha256=key, problem=text, source=str(r.get('source', 'custom'))))
    if len({r['problem_sha256'] for r in rows}) != len(rows):
        raise ValueError('Duplicate normalized questions')
    return rows
