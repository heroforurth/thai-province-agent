"""End-to-end answer evaluation on eval/data/qa_gold.jsonl (34 questions).

Systems compared (same model, temperature 0):
  closed_book - baseline: the model alone, no retrieval, no tools
  rag         - single-shot hybrid RAG with citations (no tools)
  agent       - the tool-calling agent (search_docs + run_sql + calculator)

Metric: deterministic answer accuracy (gold alias contained, or number within the
item's tolerance; unanswerable items must be refused) - see eval/metrics.py.
Also reported for the agent: citation validity, tool calls, JSON repairs, stop reasons.

    python -m eval.eval_answers                      # run all systems (needs OPENAI_API_KEY)
    python -m eval.eval_answers --systems agent      # one system
    python -m eval.eval_answers --rescore            # recompute metrics from saved outputs, no API calls

Raw predictions are saved to eval/results/answers_<system>.jsonl so the numbers can be
re-scored and audited without calling the model again.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from eval.metrics import score_answer

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SYSTEMS = ("closed_book", "rag", "agent")
TYPES = ("text", "sql", "multi", "unanswerable")


def load_gold() -> list[dict]:
    return [json.loads(l) for l in open(HERE / "data" / "qa_gold.jsonl", encoding="utf-8")]


def run_system(name: str, gold: list[dict]) -> list[dict]:
    from app import agent, llm, rag
    from app.retrieval import build_retriever
    from app.sqldb import connect_readonly
    from app.tools import ToolBox

    retriever = build_retriever(dense=True) if name != "closed_book" else None
    if name == "agent":
        toolbox = ToolBox(retriever, connect_readonly())
    out = []
    for item in gold:
        t0 = time.time()
        row = {"id": item["id"], "type": item["type"], "question": item["question"]}
        try:
            if name == "closed_book":
                row.update(rag.closed_book_answer(item["question"], llm.chat))
            elif name == "rag":
                res = rag.rag_answer(item["question"], retriever, llm.chat)
                row.update(answer=res["answer"], refused=res["refused"], sources=[s["doc_id"] for s in res["sources"]])
            else:
                final, trace = agent.answer(item["question"], toolbox, llm.chat)
                row.update(
                    answer=final["answer"], citations=[c["doc_id"] for c in final["citations"]],
                    invalid_citations=final["invalid_citations"], used_sql=final["used_sql"],
                    tool_calls=final["tool_calls"], repairs=final["repairs"], stop_reason=final["stop_reason"],
                    tools=[e["action"] for e in trace if e["type"] == "step"],
                    sql=[e["args"].get("sql") for e in trace if e["type"] == "step" and e["action"] == "run_sql"],
                )
        except Exception as e:  # a crash is a wrong answer, not a missing row
            row.update(answer=f"ERROR: {e}", error=True)
        row["latency_s"] = round(time.time() - t0, 2)
        row["correct"] = score_answer(row["answer"], item)
        print(f"  [{name}] {item['id']} {'OK ' if row['correct'] else 'XX '} {row['answer'][:90]!r}")
        out.append(row)
    return out


def summarise(name: str, rows: list[dict], gold: dict[str, dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        r["correct"] = score_answer(r["answer"], gold[r["id"]])  # always re-score from text
        by[r["type"]].append(r["correct"])
    s = {"system": name, "n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
         "by_type": {t: (sum(v) / len(v), len(v)) for t, v in by.items()},
         "median_latency_s": sorted(r["latency_s"] for r in rows)[len(rows) // 2]}
    if name == "agent":
        answered = [r for r in rows if gold[r["id"]]["type"] != "unanswerable"]
        s["grounded_rate"] = sum(1 for r in answered if r.get("citations") or r.get("used_sql")) / max(1, len(answered))
        s["answers_with_invalid_citation"] = sum(1 for r in rows if r.get("invalid_citations"))
        s["mean_tool_calls"] = sum(r.get("tool_calls", 0) for r in rows) / len(rows)
        s["total_json_repairs"] = sum(r.get("repairs", 0) for r in rows)
        stops = defaultdict(int)
        for r in rows:
            stops[r.get("stop_reason", "?")] += 1
        s["stop_reasons"] = dict(stops)
    return s


def to_markdown(summaries: list[dict]) -> str:
    lines = ["# Answer evaluation\n", "34 questions: 14 text (articles), 11 sql (statistics table), 7 multi-hop (both), "
             "2 unanswerable. Metric: deterministic accuracy (see eval/metrics.py). Model temperature 0.\n",
             "| system | overall | text | sql | multi | unanswerable | median latency (s) |", "|---|---|---|---|---|---|---|"]
    for s in summaries:
        cells = []
        for t in TYPES:
            acc, n = s["by_type"].get(t, (0.0, 0))
            cells.append(f"{acc:.2f} ({round(acc * n)}/{n})")
        lines.append(f"| {s['system']} | **{s['accuracy']:.3f}** ({round(s['accuracy'] * s['n'])}/{s['n']}) | "
                     + " | ".join(cells) + f" | {s['median_latency_s']} |")
    ag = next((s for s in summaries if s["system"] == "agent"), None)
    if ag:
        lines += ["\n## Agent behaviour\n",
                  f"- grounded answers (>=1 valid citation or SQL), answerable items: {ag['grounded_rate']:.2f}",
                  f"- answers containing an invalid (hallucinated) citation number: {ag['answers_with_invalid_citation']}",
                  f"- mean tool calls per question: {ag['mean_tool_calls']:.2f}",
                  f"- JSON repairs needed (total): {ag['total_json_repairs']}",
                  f"- stop reasons: {ag['stop_reasons']}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    ap.add_argument("--rescore", action="store_true", help="re-score saved outputs, no API calls")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    gold_list = load_gold()[: args.limit]
    gold = {g["id"]: g for g in gold_list}
    RESULTS.mkdir(exist_ok=True)
    if not args.rescore:
        for name in args.systems:
            print(f"running {name} ...")
            rows = run_system(name, gold_list)
            with open(RESULTS / f"answers_{name}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summarise every system that has saved outputs (so the table always shows all of them).
    summaries = []
    for name in SYSTEMS:
        path = RESULTS / f"answers_{name}.jsonl"
        if path.exists():
            rows = [r for r in (json.loads(l) for l in open(path, encoding="utf-8")) if r["id"] in gold]
            summaries.append(summarise(name, rows, gold))
    if not summaries:
        print("no saved outputs to score")
        return 1
    md = to_markdown(summaries)
    (RESULTS / "answers.md").write_text(md, encoding="utf-8")
    (RESULTS / "answers.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
