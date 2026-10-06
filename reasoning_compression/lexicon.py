"""Finite-state L27 matcher, with token transitions compiled before generation.

Full-step mode delays a word-end match until a boundary is observed. Opening
mode scores a candidate token as if it were the end of the current prefix,
matching large_lexicon_reference. No tokenizer encoding assumptions are used.
"""
from collections import deque
import numpy as np

ALPHABET = 'abcdefghijklmnopqrstuvwxyz' + ' -!#'


def characters(text):
    for c in text:
        if c in '–—': c = '-'
        # Author calls .lower() before regex IGNORECASE: capital dotted I
        # expands to i + combining dot and is NOT equivalent to plain i.
        if c == 'İ':
            yield 'i'; yield '!'; continue
        if c == 'ı': c = 'i'
        if c == 'ſ': c = 's'
        if c == 'K': c = 'k'
        c = c.lower()
        if len(c) == 1 and 'a' <= c <= 'z': yield c
        elif c.isspace(): yield ' '
        elif c == '-': yield '-'
        elif c.isalnum() or c == '_': yield '#'
        else: yield '!'


def expanded_terms(base):
    variants = {'alternative':['alternative','alternatives','alternatively'],
        'confusing':['confuse','confuses','confused','confusing'],
        'differently':['different','differently'], 'careful':['careful','carefully'],
        'sometimes':['sometime','sometimes'],
        'alternate':['alternate','alternates','alternated','alternating']}
    return sorted({v for term in base for v in variants.get(term,[term])})


class Automaton:
    def __init__(self, terms, opening=False):
        self.terms = set(terms)
        self.prefixes = {s[:i] for s in terms for i in range(1,len(s)+1)}
        self.opening = opening
        # prefix-set, preceding character is word, still at initial whitespace
        start = ((), False, True)
        self.states = [start]
        mapping = {start:0}
        transitions, hits = [], []
        for state in self.states:
            tr, hr = [], []
            for c in ALPHABET:
                nxt, hit = self.advance(state,c)
                if nxt not in mapping:
                    mapping[nxt] = len(self.states); self.states.append(nxt)
                tr.append(mapping[nxt]); hr.append(hit)
            transitions.append(tr); hits.append(hr)
        self.transition = np.asarray(transitions,dtype=np.int32)
        self.hits = np.asarray(hits,dtype=bool)
        self.final = np.asarray([any(p in self.terms for p in s[0]) for s in self.states])

    def advance(self, state, c):
        prefixes, prev_word, initial = state
        word = c.isalpha() or c == '#'
        hit = not word and any(p in self.terms for p in prefixes)
        nxt = set()
        for p in prefixes:
            if c in ' -':
                q = p if p.endswith(' ') else p+' '
            else: q = p+c
            if q in self.prefixes: nxt.add(q)
        can_start = initial if self.opening else not prev_word
        if can_start and c in self.prefixes: nxt.add(c)
        initial = initial and c == ' '
        return (tuple(sorted(nxt)),word,initial),hit

    def consume(self, state, text):
        hit = False
        for c in characters(text):
            col = ALPHABET.index(c)
            hit |= bool(self.hits[state,col])
            state = int(self.transition[state,col])
        return state,hit

    def compile_tokens(self, pieces):
        n,v = len(self.states),len(pieces)
        table = np.empty((n,v),dtype=np.int32)
        hits = np.empty((n,v),dtype=bool)
        initial = np.arange(n)
        for token,piece in enumerate(pieces):
            state = initial.copy(); hit = np.zeros(n,dtype=bool)
            for c in characters(piece):
                col = ALPHABET.index(c)
                hit |= self.hits[state,col]
                state = self.transition[state,col]
            table[:,token] = state
            hits[:,token] = hit | self.final[state] if self.opening else hit
        return table,hits


def compact_tables(transition,hits,final,opening):
    """Deduplicate equivalent token transitions without approximating matches."""
    states=transition.shape[0]
    assert states < 32768
    packed=(transition+hits.astype(np.int32)*states).T.astype(np.uint16)
    unique,classes=np.unique(packed,axis=0,return_inverse=True)
    tr=(unique.T % states).astype(np.int32)
    hs=(unique.T >= states)
    # Opening terminal states are closed immediately on accepting that token;
    # their completion edges are unreachable for subsequent suppression.
    candidate_classes=hs[~final].any(axis=0) if opening else np.zeros(len(unique),dtype=bool)
    return dict(transition=tr,hits=hs,final=final,
        token_classes=classes.astype(np.int32),candidate_ids=np.flatnonzero(candidate_classes[classes]).astype(np.int32))
