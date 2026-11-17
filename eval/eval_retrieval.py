"""Retrieval evaluation: Recall@k / MRR on the labelled set, baselines, a chunk-size
sweep, an ablation, and the min_score floor chosen from in- vs off-corpus questions.

    python -m eval.eval_retrieval            # everything (needs OPENAI_API_KEY for dense/hybrid)
    python -m eval.eval_retrieval --bm25-only  # the parts that need no API key

Writes eval/results/retrieval.json and eval/results/retrieval.md.
Embeddings are cached (data/cache/), so re-runs are fast and give identical numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from app import config
from app.chunking import chunk_docs, load_docs
from app.retrieval import Retriever
from eval.metrics import first_relevant_rank, is_relevant, retrieval_metrics

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

NAIVE = dict(strategy="fixed", size=200, overlap=0, header=False, drop_boilerplate=False)
OURS = dict(strategy="section", size=config.CHUNK_WORDS, overlap=config.CHUNK_OVERLAP, header=True, drop_boilerplate=True)


def load_qrels() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(l) for l in open(HERE / "data" / "retrieval_qrels.jsonl", encoding="utf-8")]
    return [r for r in rows if r["doc_id"]], [r for r in rows if not r["doc_id"]]


def evaluate(retriever: Retriever, qrels: list[dict], mode: str, depth: int = 10) -> tuple[dict, list[dict]]:
    ranks, doc_ranks, per_query = [], [], []
    unreachable = 0
    for q in qrels:
        if not any(is_relevant(c, q) for c in retriever.chunks):
            unreachable += 1  # evidence split across a chunk boundary in this config
        res = retriever.search(q["query"], k=depth, mode=mode)
        chunks = [h.chunk for h in res.hits]
        r = first_relevant_rank(chunks, q)
        docs = list(dict.fromkeys(c.doc_id for c in chunks))
        dr = docs.index(q["doc_id"]) + 1 if q["doc_id"] in docs else None
        ranks.append(r)
        doc_ranks.append(dr)
        per_query.append({"id": q["id"], "style": q["style"], "rank": r, "doc_rank": dr,
                          "top1": chunks[0].chunk_id if chunks else None})
    m = retrieval_metrics(ranks, doc_ranks)
    m["unreachable"] = unreachable
    for style in ("implicit", "explicit"):
        sub = [p for p in per_query if p["style"] == style]
        m[f"recall@5_{style}"] = sum(1 for p in sub if p["rank"] and p["rank"] <= 5) / max(1, len(sub))
    return m, per_query


def sweep_min_score(retriever: Retriever, in_q: list[dict], off_q: list[dict]) -> dict:
    """For each threshold t: in-corpus questions kept (top cosine >= t) and off-corpus
    questions refused (top cosine < t). Pick t maximising balanced accuracy; ties ->
    the LOWER t (refusing a real question is worse than one extra off-topic answer)."""
    def top_cos(q):
        return float((retriever.vectors @ retriever.cache.get_many([q], retriever.embed_fn)[0]).max())

    ins = [top_cos(q["query"]) for q in in_q]
    offs = [top_cos(q["query"]) for q in off_q]
    rows = []
    for t in np.round(np.arange(0.30, 0.75, 0.025), 3):
        kept = sum(c >= t for c in ins) / len(ins)
        refused = sum(c < t for c in offs) / len(offs)
        rows.append({"threshold": float(t), "in_corpus_kept": kept, "off_corpus_refused": refused,
                     "balanced_acc": (kept + refused) / 2})
    best = max(rows, key=lambda r: (round(r["balanced_acc"], 6), -r["threshold"]))
    return {"in_corpus_top_cosine": {"min": min(ins), "median": float(np.median(ins)), "max": max(ins)},
            "off_corpus_top_cosine": {"min": min(offs), "median": float(np.median(offs)), "max": max(offs)},
            "sweep": rows, "best": best,
            "per_query": {"in": dict(zip([q["id"] for q in in_q], ins)), "off": dict(zip([q["id"] for q in off_q], offs))}}


def fmt(m: dict) -> str:
    return (f"| {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | {m['mrr@10']:.3f} | "
            f"{m['doc_recall@5']:.3f} | {m['recall@5_implicit']:.3f} | {m['recall@5_explicit']:.3f} | {m['unreachable']} |")


HEADER = ("| config | R@1 | R@3 | R@5 | MRR@10 | doc R@5 | R@5 implicit | R@5 explicit | unreachable |\n"
          "|---|---|---|---|---|---|---|---|---|")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25-only", action="store_true", help="skip dense/hybrid (no API key needed)")
    args = ap.parse_args()
    dense = not args.bm25_only
    if dense and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -> running --bm25-only", file=sys.stderr)
        dense = False

    in_q, off_q = load_qrels()
    docs = load_docs(config.DOCS_DIR)
    embed_fn = None
    if dense:
        from app import llm
        embed_fn = llm.embed
    modes = ["bm25", "dense", "hybrid"] if dense else ["bm25"]
    results: dict = {"n_in_corpus": len(in_q), "n_off_corpus": len(off_q), "experiments": {}}
    md = [f"# Retrieval evaluation\n\n{len(in_q)} labelled in-corpus queries ({sum(q['style']=='implicit' for q in in_q)} implicit = "
          f"province not named), {len(off_q)} off-corpus queries. Relevant = chunk from the labelled article that contains "
          f"the labelled evidence span. Generated by `python -m eval.eval_retrieval`.\n"]
    t0 = time.time()

    def run(name: str, chunk_opts: dict, mode: str):
        r = Retriever(chunk_docs(docs, **chunk_opts), embed_fn=embed_fn if mode != "bm25" else None)
        m, per = evaluate(r, in_q, mode)
        results["experiments"][name] = {"chunking": chunk_opts, "mode": mode, "chunks": len(r.chunks), "metrics": m, "per_query": per}
        print(f"{name:40s} R@5={m['recall@5']:.3f} MRR={m['mrr@10']:.3f} ({len(r.chunks)} chunks)")
        return m

    # 1) Baselines vs ours: retriever x chunking
    md += ["## 1. Retriever x chunking (baselines vs ours)\n", HEADER]
    for label, opts in (("naive fixed-200", NAIVE), (f"section-{OURS['size']}+header", OURS)):
        for mode in modes:
            md.append(f"| {mode} / {label} " + fmt(run(f"{mode}|{label}", opts, mode)))

    best_mode = "hybrid" if dense else "bm25"
    # 2) Chunk-size sweep (our strategy, best retriever)
    md += [f"\n## 2. Chunk size sweep ({best_mode}, section + header, overlap = size/4)\n", HEADER]
    for size in (60, 120, 200, 300):
        opts = {**OURS, "size": size, "overlap": size // 4}
        md.append(f"| {best_mode} / size {size} " + fmt(run(f"{best_mode}|size{size}", opts, best_mode)))

    # 3) Ablation of the two chunking ideas
    md += [f"\n## 3. Ablation ({best_mode}, size {OURS['size']})\n", HEADER]
    for label, opts in (("no contextual header", {**OURS, "header": False}),
                        ("keep boilerplate", {**OURS, "drop_boilerplate": False})):
        md.append(f"| {label} " + fmt(run(f"{best_mode}|{label}", opts, best_mode)))

    # 4) min_score floor
    if dense:
        r = Retriever(chunk_docs(docs, **OURS), embed_fn=embed_fn)
        ms = sweep_min_score(r, in_q, off_q)
        results["min_score"] = ms
        b = ms["best"]
        md += ["\n## 4. min_score floor (top dense cosine, in- vs off-corpus)\n",
               f"In-corpus top cosine: min {ms['in_corpus_top_cosine']['min']:.3f}, median {ms['in_corpus_top_cosine']['median']:.3f}. "
               f"Off-corpus: median {ms['off_corpus_top_cosine']['median']:.3f}, max {ms['off_corpus_top_cosine']['max']:.3f}.\n",
               "| threshold | in-corpus kept | off-corpus refused | balanced acc |", "|---|---|---|---|"]
        for row in ms["sweep"]:
            mark = " **<- chosen**" if row is b else ""
            md.append(f"| {row['threshold']:.3f} | {row['in_corpus_kept']:.3f} | {row['off_corpus_refused']:.3f} | {row['balanced_acc']:.3f}{mark} |")
        md.append(f"\nNaive default (no floor, threshold 0): in-corpus kept 1.000, off-corpus refused 0.000, balanced acc 0.500.")
        print(f"best min_score = {b['threshold']} (balanced acc {b['balanced_acc']:.3f})")

    RESULTS.mkdir(exist_ok=True)
    suffix = "" if dense else "_bm25only"
    (RESULTS / f"retrieval{suffix}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (RESULTS / f"retrieval{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote eval/results/retrieval{suffix}.md in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
