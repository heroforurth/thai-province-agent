"""Agent-loop, retrieval and API tests with a scripted fake model and a fake embedder."""
import json

import pytest
from fastapi.testclient import TestClient

from app import agent, config
from app.chunking import chunk_docs, load_docs
from app.retrieval import Retriever
from app.sqldb import build_db, connect_readonly
from app.tools import ToolBox
from tests.conftest import ScriptedChat, fake_embed


@pytest.fixture(scope="module")
def chunks():
    return chunk_docs(load_docs(config.DOCS_DIR))


@pytest.fixture
def retriever(chunks, tmp_cache):
    return Retriever(chunks, embed_fn=fake_embed)


@pytest.fixture
def toolbox(retriever, tmp_path):
    p = tmp_path / "p.db"
    build_db(p)
    return ToolBox(retriever, connect_readonly(p))


# ------------------------------------------------------------------ retrieval


def test_hybrid_finds_river_kwai(retriever):
    res = retriever.search("Bridge over the River Kwai", k=5, mode="hybrid")
    assert res.hits[0].chunk.doc_id == "kanchanaburi"


def test_min_score_refuses_off_topic(retriever):
    res = retriever.search("zzqx blorf wibble", k=5, mode="hybrid", min_score=0.5)
    assert res.refused and res.hits == []


def test_bm25_mode_needs_no_embeddings(chunks):
    r = Retriever(chunks)
    assert r.search("Doi Inthanon", mode="bm25", k=3).hits[0].chunk.doc_id == "chiang-mai"
    with pytest.raises(RuntimeError):
        r.search("Doi Inthanon", mode="dense")


# ------------------------------------------------------------------ agent loop


def _j(**kw):
    return json.dumps(kw)


def test_agent_multi_hop_search_then_sql(toolbox, monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.0)
    chat = ScriptedChat([
        _j(thought="find province", action="search_docs", args={"query": "Bridge over the River Kwai"}),
        _j(thought="population", action="run_sql", args={"sql": "SELECT population FROM provinces WHERE name='Kanchanaburi'"}),
        _j(action="final_answer", args={"answer": "Kanchanaburi has 896,000 people [1].", "citations": [1, 42], "used_sql": True}),
    ])
    final, trace = agent.answer("population of the River Kwai province?", toolbox, chat)
    assert final["stop_reason"] == "final_answer" and final["tool_calls"] == 2
    assert [c["n"] for c in final["citations"]] == [1] and final["invalid_citations"] == [42]
    sql_obs = [e for e in trace if e["type"] == "observation" and e["tool"] == "run_sql"][0]
    assert sql_obs["table"]["rows"] == [[896000]]
    # tool results are framed as untrusted data when fed back to the model
    assert "untrusted data" in chat.calls[-1][-1]["content"]


def test_agent_rejects_answer_without_tool_use(toolbox):
    chat = ScriptedChat([
        _j(action="final_answer", args={"answer": "From memory: Chiang Mai."}),
        _j(action="run_sql", args={"sql": "SELECT name FROM provinces ORDER BY area_km2 DESC LIMIT 1"}),
        _j(action="final_answer", args={"answer": "Chiang Mai.", "used_sql": True}),
    ])
    final, trace = agent.answer("largest province?", toolbox, chat)
    assert final["answer"] == "Chiang Mai." and final["tool_calls"] == 1
    assert any(e["type"] == "error" and "without using a tool" in e["message"] for e in trace)


def test_agent_stops_on_repeated_call_and_forces_answer(toolbox):
    call = _j(action="calculator", args={"expression": "1+1"})
    chat = ScriptedChat([call, call, _j(action="final_answer", args={"answer": "2"})])
    final, _ = agent.answer("1+1?", toolbox, chat)
    assert final["stop_reason"] == "forced:repeated_call" and final["answer"] == "2"


def test_agent_max_steps_then_fallback(toolbox):
    chat = ScriptedChat([_j(action="calculator", args={"expression": f"{i}+1"}) for i in range(3)] + ["garbage"])
    final, _ = agent.answer("loop forever", toolbox, chat, max_steps=3)
    assert final["stop_reason"] == "fallback:max_steps" and final["tool_calls"] == 3


def test_agent_survives_invalid_json_with_repair(toolbox):
    chat = ScriptedChat([
        "I think I should search!",  # invalid -> repair turn
        _j(action="search_docs", args={"query": "Doi Inthanon"}),
        _j(action="final_answer", args={"answer": "Chiang Mai [1]", "citations": [1]}),
    ])
    final, _ = agent.answer("Where is Doi Inthanon?", toolbox, chat)
    assert final["repairs"] == 1 and final["stop_reason"] == "final_answer"


def test_sql_tool_errors_are_returned_not_raised(toolbox):
    chat = ScriptedChat([
        _j(action="run_sql", args={"sql": "DROP TABLE provinces"}),
        _j(action="final_answer", args={"answer": "could not find it"}),
    ])
    _, trace = agent.answer("drop it", toolbox, chat)
    obs = [e for e in trace if e["type"] == "observation"][0]
    assert not obs["ok"] and "only SELECT" in obs["text"]


# ------------------------------------------------------------------ API (SSE)


@pytest.fixture
def client(monkeypatch, tmp_cache):
    from app import llm, main

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(llm, "embed", fake_embed)
    monkeypatch.setattr(config, "MIN_SCORE", 0.0)
    replies = ScriptedChat([
        _j(action="search_docs", args={"query": "Doi Inthanon"}),
        _j(action="final_answer", args={"answer": "Chiang Mai [1].", "citations": [1]}),
    ])
    monkeypatch.setattr(llm, "chat", replies)
    monkeypatch.setattr(llm, "chat_stream", lambda msgs, **kw: iter(["Chiang ", "Mai [1]."]))
    with TestClient(main.app) as c:
        yield c


def _events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_api_health_and_search(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["chunks"] > 1000
    s = client.get("/api/search", params={"q": "Doi Inthanon"}).json()
    assert "chiang-mai" in [x["doc_id"] for x in s["sources"]]


def test_api_agent_stream(client):
    resp = client.post("/api/ask", json={"question": "Where is Doi Inthanon?", "mode": "agent"})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    evs = _events(resp)
    assert [e["type"] for e in evs] == ["step", "observation", "answer"]
    assert evs[-1]["citations"][0]["n"] == 1 and evs[-1]["citations"][0]["url"].startswith("https://")


def test_api_rag_stream(client):
    evs = _events(client.post("/api/ask", json={"question": "Where is Doi Inthanon?", "mode": "rag"}))
    assert evs[0]["type"] == "sources" and "".join(e["text"] for e in evs if e["type"] == "token") == "Chiang Mai [1]."


def test_api_validates_input(client):
    assert client.post("/api/ask", json={"question": "x", "mode": "agent"}).status_code == 422
    assert client.post("/api/ask", json={"question": "hello there", "mode": "shell"}).status_code == 422


def test_serving_falls_back_to_bm25_when_embedding_api_fails(chunks, tmp_cache):
    api = {"up": True}

    def flaky_embed(texts):
        if not api["up"]:
            raise ConnectionError("proxy down")
        return fake_embed(texts)

    r = Retriever(chunks, embed_fn=flaky_embed, fallback_to_bm25=True)  # chunks embed while the API is up
    strict = Retriever(chunks, embed_fn=flaky_embed)
    api["up"] = False  # ...then the proxy goes down before the queries arrive
    res = r.search("Doi Inthanon", k=3, mode="hybrid")
    assert res.mode.startswith("bm25") and res.hits[0].chunk.doc_id == "chiang-mai"
    with pytest.raises(ConnectionError):
        strict.search("another query", mode="hybrid")
