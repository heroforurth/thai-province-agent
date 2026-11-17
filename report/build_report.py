"""Build report/report.pdf from report/report.html.tpl + eval/results/*.json.

Tables are generated from the saved evaluation results, so the numbers in the PDF are
exactly the numbers the eval scripts produced.  Placeholders: {{name}} in the template.

    python report/build_report.py
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "report"
RES = ROOT / "eval" / "results"


def table(headers: list[str], rows: list[list], bold_row: int | None = None) -> str:
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = ""
    for i, r in enumerate(rows):
        cls = ' class="best"' if i == bold_row else ""
        body += f"<tr{cls}>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>"
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def f3(x: float) -> str:
    return f"{x:.3f}"


def retrieval_tables(ret: dict) -> dict[str, str]:
    ex = ret["experiments"]
    cols = ["Config", "chunks", "R@1", "R@3", "R@5", "MRR@10", "R@5 implicit", "R@5 explicit"]

    def row(label, key):
        e = ex[key]
        m = e["metrics"]
        return [label, e["chunks"], f3(m["recall@1"]), f3(m["recall@3"]), f3(m["recall@5"]), f3(m["mrr@10"]),
                f3(m["recall@5_implicit"]), f3(m["recall@5_explicit"])]

    main_keys = [k for k in ex if "|naive" in k or "|section" in k]
    main_rows = [row(k.replace("|", " / "), k) for k in main_keys]
    best = max(range(len(main_keys)), key=lambda i: (ex[main_keys[i]]["metrics"]["mrr@10"]))
    sweep_keys = [k for k in ex if "|size" in k]
    abl_keys = [k for k in ex if k.endswith("no contextual header") or k.endswith("keep boilerplate")]
    out = {
        "retrieval_main": table(cols, main_rows, best),
        "retrieval_sweep": table(cols, [row(k.split("|")[1].replace("size", "size "), k) for k in sweep_keys]),
        "retrieval_ablation": table(cols, [row(k.split("|")[1], k) for k in abl_keys]),
    }
    ms = ret.get("min_score")
    if ms:
        rows = [[f"{r['threshold']:.3f}", f3(r["in_corpus_kept"]), f3(r["off_corpus_refused"]), f3(r["balanced_acc"])]
                for r in ms["sweep"] if 0.35 <= r["threshold"] <= 0.70]
        best_i = next(i for i, r in enumerate(x for x in ms["sweep"] if 0.35 <= x["threshold"] <= 0.70)
                      if r["threshold"] == ms["best"]["threshold"])
        out["min_score_table"] = table(["min_score", "in-corpus kept", "off-corpus refused", "balanced acc."], rows, best_i)
        out["min_score_best"] = f"{ms['best']['threshold']:.3f}"
    return out


def answer_tables(ans: list[dict]) -> dict[str, str]:
    rows = []
    for s in ans:
        cells = [s["system"], f"<b>{s['accuracy']:.3f}</b> ({round(s['accuracy'] * s['n'])}/{s['n']})"]
        for t in ("text", "sql", "multi", "unanswerable"):
            acc, n = s["by_type"].get(t, (0, 0))
            cells.append(f"{acc:.2f} ({round(acc * n)}/{n})")
        cells.append(s["median_latency_s"])
        rows.append(cells)
    out = {"answers_table": table(["System", "Overall", "Text (14)", "SQL (11)", "Multi-hop (7)", "Unanswerable (2)", "Median s"], rows,
                                  max(range(len(ans)), key=lambda i: ans[i]["accuracy"]))}
    ag = next((s for s in ans if s["system"] == "agent"), None)
    if ag:
        out["agent_stats"] = table(
            ["Grounded (cites or SQL)", "Answers w/ invalid citation", "Mean tool calls", "JSON repairs", "Stop reasons"],
            [[f"{ag['grounded_rate']:.2f}", ag["answers_with_invalid_citation"], f"{ag['mean_tool_calls']:.2f}",
              ag["total_json_repairs"], ", ".join(f"{k}: {v}" for k, v in ag["stop_reasons"].items())]])
    return out


def main() -> int:
    tpl = (HERE / "report.html.tpl").read_text(encoding="utf-8")
    vals: dict[str, str] = {}
    vals.update(retrieval_tables(json.loads((RES / "retrieval.json").read_text(encoding="utf-8"))))
    vals.update(answer_tables(json.loads((RES / "answers.json").read_text(encoding="utf-8"))))
    svg = (HERE / "architecture.svg").read_text(encoding="utf-8")
    vals["architecture"] = f'<img alt="architecture" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'
    for k, v in vals.items():
        tpl = tpl.replace("{{" + k + "}}", str(v))
    if "{{" in tpl:
        print("WARNING: unfilled placeholder near:", tpl[tpl.index("{{"):][:60], file=sys.stderr)
    html = HERE / "report.html"
    html.write_text(tpl, encoding="utf-8")

    browser = next((p for p in [
        shutil.which("chrome"), shutil.which("msedge"), shutil.which("chromium"), shutil.which("google-chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ] if p and Path(p).exists()), None)
    if not browser:
        print(f"wrote {html} (no Chrome/Edge found: print it to PDF from a browser)")
        return 0
    pdf = HERE / "report.pdf"
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.resolve().as_uri()], check=True, capture_output=True, timeout=120)
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
