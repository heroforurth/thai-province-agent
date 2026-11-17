"""Retrieval evaluation: Recall@k / MRR on the labelled set, baselines, a chunk-size
sweep, an ablation, and the min_score floor chosen from in- vs off-corpus questions.

    python -m eval.eval_retrieval            # everything (needs OPENAI_API_KEY for dense/hybrid)
    python -m eval.eval_retrieval --bm25-only  # the parts that need no API key

Writes eval/results/retrieval.json and eval/results/retrieval.md.
Embeddings are cached (data/cache/), so re-runs are fast and give identical numbers.
"""
from __future__ import annotations

import argparse
import logging
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
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
    results: dict = {"n_in_corpus": len(in_q), "n_off_corpus": len(off_q), "experiments": {}, "skipped": []}
    RESULTS.mkdir(exist_ok=True)
    suffix = "" if dense else "_bm25only"
    t0 = time.time()

    def save():  # written after every experiment, so a proxy outage never loses finished work
        (RESULTS / f"retrieval{suffix}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (RESULTS / f"retrieval{suffix}.md").write_text(render_md(results, in_q, off_q), encoding="utf-8")

    def run(name: str, group: str, chunk_opts: dict, mode: str):
        try:
            r = Retriever(chunk_docs(docs, **chunk_opts), embed_fn=embed_fn if mode != "bm25" else None)
            m, per = evaluate(r, in_q, mode)
        except Exception as e:  # e.g. the proxy is down: record and move on
            results["skipped"].append({"name": name, "error": str(e)[:200]})
            print(f"{name:40s} SKIPPED ({str(e)[:80]})")
            save()
            return
        results["experiments"][name] = {"group": group, "chunking": chunk_opts, "mode": mode, "chunks": len(r.chunks),
                                        "metrics": m, "per_query": per}
        print(f"{name:40s} R@5={m['recall@5']:.3f} MRR={m['mrr@10']:.3f} ({len(r.chunks)} chunks)")
        save()

    best_mode = "hybrid" if dense else "bm25"
    ours_label = f"section-{OURS['size']}+header"
    # Ordered so the cheapest / most important experiments (already-embedded chunks) run first.
    for mode in modes:
        run(f"{mode}|{ours_label}", "main", OURS, mode)
    if dense:
        try:
            r = Retriever(chunk_docs(docs, **OURS), embed_fn=embed_fn)
            results["min_score"] = sweep_min_score(r, in_q, off_q)
            print(f"best min_score = {results['min_score']['best']['threshold']}")
            save()
        except Exception as e:
            results["skipped"].append({"name": "min_score", "error": str(e)[:200]})
    for mode in modes:
        run(f"{mode}|naive fixed-200", "main", NAIVE, mode)
    run(f"{best_mode}|keep boilerplate", "ablation", {**OURS, "drop_boilerplate": False}, best_mode)
    for size in (200, 300, 80):
        if size != OURS["size"]:
            run(f"{best_mode}|size{size}", "sweep", {**OURS, "size": size, "overlap": size // 4}, best_mode)
    run(f"{best_mode}|no contextual header", "ablation", {**OURS, "header": False}, best_mode)
    save()
    print(f"wrote eval/results/retrieval{suffix}.md in {time.time() - t0:.0f}s; skipped: {[x['name'] for x in results['skipped']]}")
    return 0


def render_md(results: dict, in_q: list, off_q: list) -> str:
    NL = chr(10)
    ex = results["experiments"]
    n_impl = sum(q["style"] == "implicit" for q in in_q)
    md = [f"# Retrieval evaluation{NL}",
          f"{len(in_q)} labelled in-corpus queries ({n_impl} implicit = province not named), {len(off_q)} off-corpus "
          "queries. Relevant = chunk from the labelled article that contains the labelled evidence span. "
          f"Generated by `python -m eval.eval_retrieval`.{NL}"]
    titles = {"main": "1. Retriever x chunking (baselines vs ours)",
              "sweep": "2. Chunk size sweep (section + header, overlap = size/4)",
              "ablation": "3. Ablation of the chunking ideas"}
    best_mode = "hybrid" if any(x["mode"] == "hybrid" for x in ex.values()) else "bm25"
    for group, title in titles.items():
        rows = [(k, v) for k, v in ex.items() if v["group"] == group]
        if group == "sweep":  # include the default size from the main block for comparison
            rows += [(k, v) for k, v in ex.items()
                     if v["group"] == "main" and "section" in k and v["mode"] == best_mode]
            rows.sort(key=lambda kv: kv[1]["chunking"]["size"])
        if not rows:
            continue
        md += [f"{NL}## {title}{NL}", HEADER]
        md += [f"| {k.replace('|', ' / ')} ({v['chunks']} chunks) " + fmt(v["metrics"]) for k, v in rows]
    ms = results.get("min_score")
    if ms:
        b = ms["best"]
        md += [f"{NL}## 4. min_score floor (top dense cosine, in- vs off-corpus){NL}",
               f"In-corpus top cosine: min {ms['in_corpus_top_cosine']['min']:.3f}, "
               f"median {ms['in_corpus_top_cosine']['median']:.3f}. Off-corpus: median "
               f"{ms['off_corpus_top_cosine']['median']:.3f}, max {ms['off_corpus_top_cosine']['max']:.3f}.{NL}",
               "| threshold | in-corpus kept | off-corpus refused | balanced acc |", "|---|---|---|---|"]
        for row in ms["sweep"]:
            mark = " **<- chosen**" if row["threshold"] == b["threshold"] else ""
            md.append(f"| {row['threshold']:.3f} | {row['in_corpus_kept']:.3f} | "
                      f"{row['off_corpus_refused']:.3f} | {row['balanced_acc']:.3f}{mark} |")
        md.append(f"{NL}Naive default (no floor): in-corpus kept 1.000, off-corpus refused 0.000, balanced acc 0.500.")
    if results.get("skipped"):
        md.append(f"{NL}**Skipped (proxy errors):** " + ", ".join(x["name"] for x in results["skipped"]))
    return NL.join(md) + NL


if __name__ == "__main__":
    raise SystemExit(main())
