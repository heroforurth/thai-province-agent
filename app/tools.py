"""The agent's tools and the allow-list that guards them.

Only the names in TOOLS can ever run, and each call's arguments have already been
validated by the Pydantic models in app/schemas.py (the agent action schema is a
discriminated union on `action`). Tool *results* are untrusted too: they are
truncated and clearly framed as data before being handed back to the model.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from app import config
from app.calculator import CalcError, calculate
from app.retrieval import Retriever
from app.schemas import CalculatorCall, RunSqlCall, SearchDocsCall, Source
from app.sqldb import SqlError, connect_readonly, run_query

TOOL_DOCS = {
    "search_docs": 'search_docs(query: str, k: int = 4) - search Wikipedia articles about Thai provinces '
    "(history, geography, landmarks, festivals, economy, name meanings). Returns numbered sources [n].",
    "run_sql": "run_sql(sql: str) - run ONE read-only SQLite SELECT over the table below. Use it for "
    "population, area, density, region, capital, ISO code, rankings, counts, sums and averages.",
    "calculator": "calculator(expression: str) - evaluate arithmetic, e.g. '22311 / 547' or '(266000 + 484000)'.",
}
TOOLS = frozenset(TOOL_DOCS)
MAX_OBSERVATION_CHARS = 2500


@dataclass
class Observation:
    tool: str
    text: str  # what the model reads
    ok: bool = True
    sources: list[Source] = field(default_factory=list)
    table: dict | None = None


class ToolBox:
    """Holds the resources tools need and the numbered sources seen so far (so the
    final answer's citations can be checked against what was actually retrieved)."""

    def __init__(self, retriever: Retriever, db: sqlite3.Connection | None = None):
        self.retriever = retriever
        self.db = db or connect_readonly()
        self.sources: dict[int, Source] = {}
        self._by_chunk: dict[str, int] = {}

    def reset(self) -> None:
        self.sources.clear()
        self._by_chunk.clear()

    # ------------------------------------------------------------------ dispatch
    def run(self, call) -> Observation:
        if call.action not in TOOLS:  # unreachable after schema validation; kept as a belt
            return Observation(call.action, f"ERROR: unknown tool {call.action!r}", ok=False)
        if isinstance(call, SearchDocsCall):
            return self._search(call.args.query, call.args.k)
        if isinstance(call, RunSqlCall):
            return self._sql(call.args.sql)
        if isinstance(call, CalculatorCall):
            return self._calc(call.args.expression)
        return Observation(call.action, "ERROR: not a tool", ok=False)

    # ------------------------------------------------------------------ tools
    def _search(self, query: str, k: int) -> Observation:
        res = self.retriever.search(query, k=k, min_score=config.MIN_SCORE)
        if res.refused or not res.hits:
            return Observation("search_docs", "No relevant passages found in the province articles.", sources=[])
        new_sources, lines = [], []
        for hit in res.hits:
            cid = hit.chunk.chunk_id
            if cid not in self._by_chunk:
                n = len(self.sources) + 1
                self._by_chunk[cid] = n
                self.sources[n] = Source(
                    n=n, doc_id=hit.chunk.doc_id, title=hit.chunk.title, section=hit.chunk.section,
                    url=hit.chunk.url, score=round(hit.score, 4), snippet=hit.chunk.text[:600],
                )
            src = self.sources[self._by_chunk[cid]]
            new_sources.append(src)
            lines.append(f"[{src.n}] {src.title} > {src.section}: {hit.chunk.text[:550]}")
        return Observation("search_docs", _clip("\n".join(lines)), sources=new_sources)

    def _sql(self, sql: str) -> Observation:
        try:
            out = run_query(self.db, sql)
        except SqlError as e:
            return Observation("run_sql", f"ERROR: {e}", ok=False)
        text = json.dumps({"columns": out["columns"], "rows": out["rows"]}, ensure_ascii=False)
        if out["truncated"]:
            text += f"\n(truncated to {config.SQL_ROW_LIMIT} rows)"
        if not out["rows"]:
            text += "\n(0 rows - check spelling of names; use LIKE '%...%' for partial matches)"
        return Observation("run_sql", _clip(text), table={"sql": sql, **out})

    def _calc(self, expression: str) -> Observation:
        try:
            return Observation("calculator", f"{expression} = {calculate(expression)}")
        except CalcError as e:
            return Observation("calculator", f"ERROR: {e}", ok=False)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_OBSERVATION_CHARS else text[:MAX_OBSERVATION_CHARS] + " ...(truncated)"
