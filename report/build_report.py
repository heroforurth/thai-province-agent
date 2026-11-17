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
    cols = ["Config", "Chunks", "R@1", "R@3", "R@5", "MRR@10", "R@5 implicit (30)", "R@5 explicit (13)"]

    def row(label, e):
        m = e["metrics"]
        return [label, e["chunks"], f3(m["recall@1"]), f3(m["recall@3"]), f3(m["recall@5"]), f3(m["mrr@10"]),
                f3(m["recall@5_implicit"]), f3(m["recall@5_explicit"])]

    def label(k, e):
        c = e["chunking"]
        if c["strategy"] == "fixed":
            return f"{e['mode']} · naive fixed {c['size']} words, no header"
        bits = [f"section {c['size']}/{c['overlap']}", "header" if c["header"] else "<b>no header</b>"]
        if not c["drop_boilerplate"]:
            bits.append("<b>boilerplate kept</b>")
        return f"{e['mode']} · " + ", ".join(bits)

    main = [(k, e) for k, e in ex.items() if e["group"] == "main"]
    sweep = [(k, e) for k, e in ex.items() if e["group"] == "sweep" or (e["group"] == "main" and e["chunking"]["strategy"] == "section")]
    sweep.sort(key=lambda kv: kv[1]["chunking"]["size"])
    default = next(((k, e) for k, e in main if e["chunking"]["strategy"] == "section"), None)
    abl = ([default] if default else []) + [(k, e) for k, e in ex.items() if e["group"] == "ablation"]
    out = {
        "retrieval_main": table(cols, [row(label(k, e), e) for k, e in main]),
        "retrieval_sweep": table(cols, [row(label(k, e), e) for k, e in sweep]),
        "retrieval_ablation": table(cols, [row(label(k, e), e) for k, e in abl]),
    }
    return out


def answer_tables(ans: list[dict]) -> dict[str, str]:
    rows = []
    for s in ans:
        cells = [s["system"].replace("_", "-"), f"<b>{s['accuracy']:.3f}</b> ({round(s['accuracy'] * s['n'])}/{s['n']})",
                 f"{s.get('auto_accuracy', s['accuracy']):.3f}"]
        for t in ("text", "sql", "multi", "unanswerable"):
            acc, n = s["by_type"].get(t, (0, 0))
            cells.append(f"{acc:.2f} ({round(acc * n)}/{n})")
        cells.append(s["median_latency_s"])
        rows.append(cells)
    out = {"answers_table": table(["System", "Audited acc.", "Automatic", "Text (14)", "SQL (11)", "Multi-hop (7)",
                                   "Unanswerable (2)", "Median s"], rows,
                                  max(range(len(ans)), key=lambda i: ans[i]["accuracy"]))}
    ag = next((s for s in ans if s["system"] == "agent"), None)
    if ag:
        out["agent_stats"] = table(
            ["Grounded (citation or SQL)", "Hallucinated citation numbers", "Mean tool calls", "JSON repairs", "Stop reasons"],
            [[f"{ag['grounded_rate']:.2f}", ag["answers_with_invalid_citation"], f"{ag['mean_tool_calls']:.2f}",
              ag["total_json_repairs"], ", ".join(f"{k}: {v}" for k, v in ag["stop_reasons"].items())]])
    return out


def main() -> int:
    tpl = (HERE / "report.html.tpl").read_text(encoding="utf-8")
    vals: dict[str, str] = {}
    # Use the full (dense + hybrid) results if they exist, else the BM25-only run.
    full = RES / "retrieval.json"
    ret = json.loads(full.read_text(encoding="utf-8")) if full.exists() else {}
    if not any(e["mode"] != "bm25" for e in ret.get("experiments", {}).values()):
        ret = json.loads((RES / "retrieval_bm25only.json").read_text(encoding="utf-8"))
    vals.update(retrieval_tables(ret))
    vals.update(answer_tables(json.loads((RES / "answers.json").read_text(encoding="utf-8"))))
    author_file = HERE / "author.txt"  # one line, e.g. "Name Surname (Student ID 67xxxxxx)"
    vals["author"] = author_file.read_text(encoding="utf-8").strip() if author_file.exists() else "[Your name, student ID]"
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
