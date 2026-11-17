"""HTTP layer only: routes, request validation, and Server-Sent Events framing.
All logic lives in app/agent.py, app/rag.py and app/retrieval.py.

Run:  uvicorn app.main:app --port 8000     ->  http://localhost:8000
"""
from __future__ import annotations

import functools
import json
import logging
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import agent, config, llm, rag
from app.retrieval import build_retriever
from app.schemas import AskRequest, SearchResponse
from app.sqldb import connect_readonly, ensure_db
from app.tools import ToolBox

log = logging.getLogger("app")
WEB_DIR = config.ROOT / "web"
STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_db()  # (re)built from the committed CSV when missing or stale
    if not os.environ.get("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY is not set: starting in BM25-only mode; /api/ask will fail")
        STATE["retriever"] = build_retriever(dense=False, fallback_to_bm25=True)
        STATE["mode"] = "bm25"
    else:
        try:  # embeds only chunks not already in the cache; queries retry briefly, then fall back to BM25
            STATE["retriever"] = build_retriever(
                dense=True, embed_fn=functools.partial(llm.embed, attempts=2), fallback_to_bm25=True)
            STATE["mode"] = config.RETRIEVER
        except Exception:
            log.exception("embedding API unavailable at startup: serving with BM25 retrieval")
            STATE["retriever"] = build_retriever(dense=False, fallback_to_bm25=True)
            STATE["mode"] = "bm25 (embedding API unavailable)"
    yield
    STATE.clear()


app = FastAPI(title="Thai Province Agent", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    r = STATE.get("retriever")
    return {"ok": r is not None, "chunks": len(r.chunks) if r else 0, "retriever": STATE.get("mode"),
            "chat_model": config.CHAT_MODEL, "embed_model": config.EMBED_MODEL,
            "min_score": config.MIN_SCORE, "top_k": config.TOP_K}


@app.get("/api/search", response_model=SearchResponse)
def search(q: str = Query(min_length=2, max_length=300), k: int = Query(config.TOP_K, ge=1, le=10)) -> SearchResponse:
    """Retrieval only (no LLM call for BM25 mode) - handy for debugging the index."""
    retriever = STATE["retriever"]
    mode = config.RETRIEVER if retriever.vectors is not None else "bm25"
    res = retriever.search(q, k=k, mode=mode, min_score=None if mode == "bm25" else config.MIN_SCORE)
    return SearchResponse(query=q, retriever=res.mode, refused=res.refused, sources=rag.sources_from(res))


def _sse(events: Iterator[dict]) -> Iterator[str]:
    try:
        for ev in events:
            yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
    except Exception as e:  # surface failures to the UI instead of a silently dropped stream
        log.exception("stream failed")
        yield f"event: fatal\ndata: {json.dumps({'type': 'fatal', 'message': str(e)[:300]})}\n\n"


def _agent_events(question: str, retriever) -> Iterator[dict]:
    db = connect_readonly()  # per request: its own SQL connection and source numbering
    try:
        yield from agent.run_agent(question, ToolBox(retriever, db), llm.chat)
    finally:
        db.close()


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY is not configured on the server")
    retriever = STATE["retriever"]
    if req.mode == "rag":
        events = rag.rag_stream(req.question, retriever, llm.chat_stream)
    else:
        events = _agent_events(req.question, retriever)
    return StreamingResponse(_sse(events), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
