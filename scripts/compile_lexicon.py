"""Compile the L27 opening automaton for a local tokenizer, entirely on CPU."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reasoning_compression.io import ROOT, read
from reasoning_compression.lexicon import Automaton, expanded_terms, compact_tables


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compare', type=Path, help='Optional frozen opening.npz for exact array comparison')
    a=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    terms=expanded_terms(read(ROOT/'configs/l27_terms.json')['terms'])
    matcher=Automaton(terms, opening=True)
    size=max(tokenizer.get_vocab().values())+1
    pieces=[tokenizer.decode([i]) for i in range(size)]
    transitions,hits=matcher.compile_tokens(pieces)
    tables=compact_tables(transitions,hits,matcher.final,True)
    if a.compare:
        with np.load(a.compare,allow_pickle=False) as old:
            if set(old.files)!=set(tables) or any(not np.array_equal(old[k],v) for k,v in tables.items()):
                raise ValueError('Compiled tables differ from the frozen tokenizer/matcher')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:np.savez_compressed(f,**tables)
    print(json.dumps(dict(vocabulary=size,states=len(matcher.states),candidate_ids=len(tables['candidate_ids']),
                          exact_comparison=a.compare is not None,gpu_used=False)))


if __name__=='__main__':main()
