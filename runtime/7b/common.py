"""Local experiment identities; importable without torch or vLLM."""
import os
import hashlib
import json
from pathlib import Path

ROOT=Path(os.environ['RC_JOB_DIR']).resolve()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def verify():
    manifest=read(ROOT/'MANIFEST.json')
    for n,h in manifest['files'].items():
        if sha(ROOT/n)!=h:raise ValueError('Manifest changed: '+n)
def prompt_ids(tok,row):
    messages=[dict(role='system',content='Please reason step by step, and put your final answer within \\boxed{}.'),dict(role='user',content=row['problem'])]
    prompt=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
    if 'expected_prompt_sha256' in row and hashlib.sha256(str(prompt).encode()).hexdigest()!=row['expected_prompt_sha256']:
        raise ValueError('Frozen prompt differs')
    if 'saved_prefix_sha256' in row and digest(prompt+row['reasoning_ids'])!=row['saved_prefix_sha256']:
        raise ValueError('Frozen prefix differs')
    if len(prompt)+16000>17920:raise ValueError('Context budget changed')
    return prompt
