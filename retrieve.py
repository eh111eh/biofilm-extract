"""Small BM25 retriever over section-tagged chunks (no external deps, runs offline).

Swap-in point for dense embeddings later: keep the `search(query, k, section)` signature.
"""
import math
import re
from collections import Counter

from ingest import Chunk

_TOKEN = re.compile(r"\w+", re.UNICODE)


_CANON = [  # map PDF glyph soup and query shorthand onto shared tokens
    (r"g\s*(?:′′|''|″)\s*0?", " gdprime "),
    (r"g\s*[′'’]\s*0|g0_prime(?:_pa)?|\bg_?0\b", " gprime0 "),
    (r"g\s*[′'’]", " gprime "),
    (r"tan[\s_]*(?:δ|delta)[\s_]*0?", " tandelta0 "),
    (r"γ\s*_?\s*y\b|gamma_?y", " gammay "),
    (r"γ\s*_?\s*f\b|gamma_?f", " gammaf "),
    (r"weak strain overshoot", " wso "),
    (r"(\d+)\s*[◦°]\s*c\b", r" \1c "),
]


def tokenize(s: str) -> list[str]:
    s = s.lower()
    for pat, rep in _CANON:
        s = re.sub(pat, rep, s)
    return _TOKEN.findall(s)


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # section title is indexed with the body so "yield behaviour" queries hit the right section
        self.docs = [tokenize(c.section + " " + c.text) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}

    def _score(self, q: list[str], i: int) -> float:
        tf, dl, s = self.tf[i], len(self.docs[i]), 0.0
        for t in q:
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int = 5, section: str = "") -> list[tuple[float, Chunk]]:
        q = tokenize(query)
        sec = section.strip().lower()
        scored = [
            (self._score(q, i), c)
            for i, c in enumerate(self.chunks)
            if not sec or sec in c.section.lower()
        ]
        scored.sort(key=lambda x: -x[0])
        return [(s, c) for s, c in scored[:k] if s > 0]
