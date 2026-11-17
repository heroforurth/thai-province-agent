"""Okapi BM25 in ~50 lines (no dependency). Strong on exact names - "Doi Inthanon",
"Bhumibol Dam", "TH-57" - which is exactly where dense embeddings are weakest."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

STOPWORDS = set(
    "a an and are as at be by for from has have he her his how in is it its of on or "
    "she that the their there these they this to was were what when where which who "
    "why will with does did do i me my you your can could would should into than then "
    "also about over under s".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, text in enumerate(docs):
            tf = Counter(tokenize(text))
            self.doc_len.append(sum(tf.values()))
            for term, n in tf.items():
                self.postings[term].append((i, n))
        self.n = len(docs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def scores(self, query: str) -> list[float]:
        out = [0.0] * self.n
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
