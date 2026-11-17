"""Retrieval: BM25, dense (bge-m3 cosine), or hybrid (Reciprocal Rank Fusion).

Hybrid fuses the two *rankings*, not their scores: BM25 scores are unbounded and
cosine lives in [-1, 1], so adding them is meaningless. RRF(d) = sum 1/(60 + rank).

The `min_score` floor answers "is this question about my corpus at all?": if the
best dense cosine is below the floor we return no hits and the app refuses instead
of answering from irrelevant context. Its value was chosen on labelled in-corpus vs
off-corpus questions (eval/eval_retrieval.py), not guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app import config
from app.bm25 import BM25
from app.chunking import Chunk, chunk_docs, load_docs
from app.embeddings import EmbedFn, EmbeddingCache

RRF_K = 60
CANDIDATES = 50  # how deep each ranker goes before fusion


@dataclass
class Hit:
    chunk: Chunk
    score: float  # the score of the active retriever (bm25 / cosine / rrf)
    cosine: float | None = None


@dataclass
class RetrievalResult:
    query: str
    mode: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    top_cosine: float | None = None


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        *,
        embed_fn: EmbedFn | None = None,
        cache: EmbeddingCache | None = None,
        fallback_to_bm25: bool = False,
    ):
        self.chunks = chunks
        # Serving only: if the embedding API fails for a query, degrade to BM25 instead of
        # failing the request. Kept OFF in evaluation so results never mix retrievers silently.
        self.fallback_to_bm25 = fallback_to_bm25
        self.bm25 = BM25([c.index_text for c in chunks])
        self.embed_fn = embed_fn
        self.cache = cache
        self.vectors: np.ndarray | None = None
        if embed_fn is not None:
            self.cache = cache or EmbeddingCache()
            self.vectors = self.cache.get_many([c.index_text for c in chunks], embed_fn)

    # ------------------------------------------------------------------ rankers
    def _bm25_rank(self, query: str) -> list[tuple[int, float]]:
        s = np.asarray(self.bm25.scores(query))
        order = np.argsort(-s)[:CANDIDATES]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]

    def _cosines(self, query: str) -> np.ndarray:
        if self.vectors is None or self.embed_fn is None:
            raise RuntimeError("dense retrieval needs embeddings (set OPENAI_API_KEY and run ingest)")
        q = self.cache.get_many([query], self.embed_fn)[0]
        return self.vectors @ q

    # ------------------------------------------------------------------ public
    def search(
        self, query: str, k: int | None = None, mode: str | None = None, min_score: float | None = None
    ) -> RetrievalResult:
        k = k or config.TOP_K  # resolved at call time so config/env overrides apply
        mode = mode or config.RETRIEVER
        res = RetrievalResult(query=query, mode=mode)
        if mode == "bm25":
            res.hits = [Hit(self.chunks[i], s) for i, s in self._bm25_rank(query)[:k]]
            return res

        try:
            cos = self._cosines(query)
        except Exception:
            if not self.fallback_to_bm25:
                raise
            res = self.search(query, k=k, mode="bm25")
            res.mode = "bm25 (embedding API unavailable)"
            return res
        res.top_cosine = float(cos.max()) if len(cos) else None
        if min_score is not None and (res.top_cosine is None or res.top_cosine < min_score):
            res.refused = True
            return res

        dense = np.argsort(-cos)[:CANDIDATES]
        if mode == "dense":
            res.hits = [Hit(self.chunks[i], float(cos[i]), float(cos[i])) for i in dense[:k]]
            return res
        if mode != "hybrid":
            raise ValueError(f"unknown retrieval mode {mode!r}")

        fused: dict[int, float] = {}
        for rank, i in enumerate(dense):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (i, _) in enumerate(self._bm25_rank(query)):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        res.hits = [Hit(self.chunks[i], s, float(cos[i])) for i, s in best]
        return res


def build_retriever(
    *, dense: bool = True, embed_fn: EmbedFn | None = None, fallback_to_bm25: bool = False, **chunk_kwargs
) -> Retriever:
    """Chunk the corpus with the configured settings and build the retriever."""
    opts = {"size": config.CHUNK_WORDS, "overlap": config.CHUNK_OVERLAP, **chunk_kwargs}
    chunks = chunk_docs(load_docs(config.DOCS_DIR), **opts)
    if dense and embed_fn is None:
        from app import llm

        embed_fn = llm.embed
    return Retriever(chunks, embed_fn=embed_fn if dense else None, fallback_to_bm25=fallback_to_bm25)
