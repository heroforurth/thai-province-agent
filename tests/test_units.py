"""Unit tests for the pure building blocks (no network, no API key)."""
import pytest
from pydantic import TypeAdapter, ValidationError

from app.bm25 import BM25
from app.calculator import CalcError, calculate
from app.chunking import Doc, chunk_docs, find_boilerplate
from app.schemas import AgentAction, FinalAnswer, RunSqlCall
from app.sqldb import SqlError, build_db, check_sql, connect_readonly, run_query
from app.structured import StructuredOutputError, complete_structured, extract_json
from eval.metrics import extract_numbers, is_refusal, retrieval_metrics, score_answer

# ------------------------------------------------------------------ structured output


def test_extract_json_from_fenced_and_chatty_reply():
    raw = 'Sure! Here you go:\n```json\n{"action": "run_sql", "args": {"sql": "SELECT 1"}}\n```'
    assert extract_json(raw)["action"] == "run_sql"


def test_extract_json_repairs_trailing_comma_and_braces_in_strings():
    raw = '{"thought": "use {braces}", "action": "calculator", "args": {"expression": "1+1",},}'
    assert extract_json(raw)["args"]["expression"] == "1+1"


def test_extract_json_raises_without_object():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_complete_structured_repairs_once_then_succeeds():
    replies = iter(['{"action": "delete_everything", "args": {}}',
                    '{"action": "calculator", "args": {"expression": "2*3"}}'])
    obj, raw, repairs = complete_structured(lambda m: next(replies), [], AgentAction, max_repairs=1)
    assert obj.action == "calculator" and repairs == 1


def test_complete_structured_gives_up_after_max_repairs():
    with pytest.raises(StructuredOutputError):
        complete_structured(lambda m: "not json", [], AgentAction, max_repairs=1)


def test_action_schema_is_the_tool_allow_list():
    ta = TypeAdapter(AgentAction)
    assert isinstance(ta.validate_python({"action": "run_sql", "args": {"sql": "SELECT 1 x"}}), RunSqlCall)
    with pytest.raises(ValidationError):
        ta.validate_python({"action": "shell", "args": {"cmd": "rm -rf /"}})
    with pytest.raises(ValidationError):  # args validated too
        ta.validate_python({"action": "search_docs", "args": {"query": "x", "k": 99}})


def test_final_answer_citation_coercion():
    fa = TypeAdapter(AgentAction).validate_python(
        {"action": "final_answer", "args": {"answer": "x", "citations": ["[1]", 2, "two"]}})
    assert isinstance(fa, FinalAnswer) and fa.args.citations == [1, 2]


# ------------------------------------------------------------------ SQL guard


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "p.db"
    build_db(path)
    return connect_readonly(path)


def test_sql_select_works(db):
    out = run_query(db, "SELECT name, population FROM provinces ORDER BY population DESC LIMIT 2")
    assert out["columns"] == ["name", "population"] and out["rows"][0][0] == "Bangkok"


def test_sql_counts_regions(db):
    assert run_query(db, "SELECT COUNT(*) FROM provinces")["rows"][0][0] == 77
    assert run_query(db, "SELECT COUNT(*) FROM provinces WHERE region='Northeastern'")["rows"][0][0] == 20


@pytest.mark.parametrize("bad", [
    "DROP TABLE provinces",
    "SELECT 1; DROP TABLE provinces",
    "PRAGMA table_info(provinces)",
    "WITH x AS (SELECT 1) DELETE FROM provinces",
    "UPDATE provinces SET population = 0",
])
def test_sql_rejects_non_select(db, bad):
    with pytest.raises(SqlError):
        run_query(db, bad)


def test_sql_authorizer_blocks_other_tables(db):
    with pytest.raises(SqlError):
        run_query(db, "SELECT * FROM sqlite_master")


def test_sql_row_limit(db):
    out = run_query(db, "SELECT name FROM provinces", row_limit=10)
    assert len(out["rows"]) == 10 and out["truncated"]


def test_check_sql_strips_fences():
    assert check_sql("```sql\nSELECT 1\n```").upper() == "SELECT 1"


# ------------------------------------------------------------------ calculator


@pytest.mark.parametrize("expr,val", [("22311 / 547", 40.787934), ("(266,000 + 484,000)", 750000), ("2^10", 1024), ("round(10/3, 2)", 3.33)])
def test_calculator(expr, val):
    assert calculate(expr) == pytest.approx(val, rel=1e-6)


@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "open('f')", "9**9999", "1/0", "a+1"])
def test_calculator_rejects(bad):
    with pytest.raises(CalcError):
        calculate(bad)


# ------------------------------------------------------------------ chunking & bm25

DOCS = [
    Doc("alpha", "Alpha", "u", "Intro about Alpha town.\n## History\nAlpha was founded long ago by river traders.\n## Human achievement index 2022\nSince 2003, the United Nations Development Programme has tracked progress in Alpha."),
    Doc("beta", "Beta", "u", "Beta is mountainous.\n## Geography\nBeta has the tallest peak.\n## Human achievement index 2022\nSince 2003, United Nations Development Programme has tracked progress in Beta."),
] + [Doc(f"d{i}", f"D{i}", "u", f"Town {i}.\n## X\nSince 2003, the United Nations Development Programme has tracked progress in D{i}.") for i in range(4)]


def test_boilerplate_detected_despite_small_wording_changes():
    assert len(find_boilerplate(DOCS)) == 1
    chunks = chunk_docs(DOCS, size=50, overlap=10)
    assert not any("Development Programme" in c.text for c in chunks)
    assert any("Development Programme" in c.text for c in chunk_docs(DOCS, size=50, overlap=10, drop_boilerplate=False))


def test_section_chunks_carry_header():
    c = next(c for c in chunk_docs(DOCS, size=50, overlap=10) if c.doc_id == "alpha" and "founded" in c.text)
    assert c.section == "History" and c.index_text.startswith("Alpha > History")


def test_windows_overlap():
    doc = Doc("x", "X", "u", " ".join(f"w{i}" for i in range(100)))
    chunks = chunk_docs([doc], strategy="fixed", size=40, overlap=10, drop_boilerplate=False)
    assert len(chunks) == 3 and chunks[1].text.split()[0] == "w30"


def test_bm25_ranks_exact_term_first():
    bm = BM25(["the bridge over the river kwai", "a mountain in the north", "river delta rice"])
    s = bm.scores("river kwai bridge")
    assert max(range(3), key=lambda i: s[i]) == 0


# ------------------------------------------------------------------ metrics


def test_extract_numbers():
    assert extract_numbers("about 896,000 people") == [896000]
    assert extract_numbers("1.8 million residents and 25%") == [1800000, 25]


def test_score_answer_alias_numeric_refusal():
    assert score_answer("It is in Kanchanaburi province [1].", {"answers": ["Kanchanaburi"]})
    assert score_answer("Phuket has 430,000 people.", {"numeric": 430000, "tol": 0.01})
    assert not score_answer("Phuket has 416,582 people.", {"numeric": 430000, "tol": 0.01})
    assert score_answer("Roughly a quarter.", {"numeric": 25, "tol": 0, "answers": ["quarter"]})
    assert score_answer("I could not find that in my sources.", {"refuse": True})
    assert not score_answer("It is 820,000.", {"refuse": True})
    assert not score_answer("Northeastern", {"answers": ["Northern"]})


def test_is_refusal():
    assert is_refusal("The sources don't say.") and not is_refusal("Chiang Mai.")


def test_retrieval_metrics():
    m = retrieval_metrics([1, 2, None, 6], [1, 1, None, 2])
    assert m["recall@1"] == 0.25 and m["recall@3"] == 0.5 and m["recall@5"] == 0.5
    assert m["mrr@10"] == pytest.approx((1 + 0.5 + 1 / 6) / 4)
