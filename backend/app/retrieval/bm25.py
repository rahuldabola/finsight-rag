"""A small Okapi BM25 over an inverted index.

Dense embeddings are good at paraphrase ("how much cash did they generate" ->
"free cash flow") but weak on exact tokens that matter in filings: tickers,
"20-F", "Q1", "EBITDA", specific figures. BM25 covers exactly that gap.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*%?")
STOPWORDS = frozenset(
    """a an and are as at be by did do does for from had has have how in is it its
    of on or that the their them they this to was were what when where which who
    why will with our we you your than then there these those been being into
    about over under per vs versus much many""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = np.zeros(self.n, dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            counts = Counter(tokenize(doc))
            self.doc_len[i] = sum(counts.values())
            for term, tf in counts.items():
                postings[term].append((i, tf))
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.postings = {
            t: (np.array([p[0] for p in ps], dtype=np.int32), np.array([p[1] for p in ps], dtype=np.float32))
            for t, ps in postings.items()
        }

    def idf(self, term: str) -> float:
        df = len(self.postings[term][0]) if term in self.postings else 0
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            out[ids] += self.idf(term) * tf * (self.k1 + 1) / (tf + norm[ids])
        return out
