"""Metrics used by the evaluation scripts (pure functions, unit-tested in tests/)."""
from __future__ import annotations

import re


def norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.\- ]", " ", s)).strip()


# ---------------------------------------------------------------- retrieval


def is_relevant(chunk, qrel: dict) -> bool:
    """A chunk is relevant iff it comes from the labelled article AND contains the
    labelled evidence span. Evidence-based labels survive re-chunking, so the same
    qrels can score every chunk size / strategy fairly."""
    return chunk.doc_id == qrel["doc_id"] and norm(qrel["evidence"]) in norm(chunk.text)


def first_relevant_rank(chunks: list, qrel: dict) -> int | None:
    for rank, c in enumerate(chunks, 1):
        if is_relevant(c, qrel):
            return rank
    return None


def retrieval_metrics(ranks: list[int | None], doc_ranks: list[int | None]) -> dict:
    """ranks[i] = 1-based rank of the first relevant chunk for query i (None = miss).
    Each query has one answer-bearing passage, so Recall@k == Hit@k here."""
    n = len(ranks) or 1

    def hit(k, rs):
        return sum(1 for r in rs if r is not None and r <= k) / n

    return {
        "recall@1": hit(1, ranks),
        "recall@3": hit(3, ranks),
        "recall@5": hit(5, ranks),
        "mrr@10": sum(1 / r for r in ranks if r is not None and r <= 10) / n,
        "doc_recall@5": hit(5, doc_ranks),
        "n": len(ranks),
    }


# ---------------------------------------------------------------- answers

_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
REFUSAL_MARKERS = (
    "could not find", "couldn't find", "can't find", "cannot find", "don't say", "do not say",
    "does not say", "not in my sources", "no information", "not mentioned", "do not contain",
    "don't contain", "does not contain", "doesn't contain", "not available", "unable to find",
    "i don't know", "not provided", "no relevant", "can only answer",
)


def extract_numbers(text: str) -> list[float]:
    """'896,000' -> 896000; '1.8 million' -> 1800000; '2,565 m' -> 2565; '25%' -> 25."""
    out = []
    for m in re.finditer(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s*(thousand|million|billion))?", text.lower()):
        val = float(m.group(1).replace(",", ""))
        if m.group(2):
            val *= _SCALE[m.group(2)]
        out.append(val)
    return out


def is_refusal(text: str) -> bool:
    t = text.lower().replace("’", "'")
    return any(m in t for m in REFUSAL_MARKERS)


def score_answer(pred: str, item: dict) -> bool:
    """Deterministic correctness: the answer is right if it contains any gold alias
    (case/punctuation-insensitive), or any number within the item's relative
    tolerance of the gold number. 'Unanswerable' items are right only on refusal."""
    if item.get("refuse"):
        return is_refusal(pred)
    p = norm(pred)
    if any(norm(a) in p for a in item.get("answers", [])):
        return True
    if "numeric" in item:
        gold, tol = float(item["numeric"]), float(item.get("tol", 0.01))
        return any(abs(x - gold) <= max(tol * abs(gold), 1e-9) for x in extract_numbers(pred))
    return False
