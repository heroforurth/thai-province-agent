"""All configuration in one place, read from the environment (and an optional .env).

Secrets (OPENAI_API_KEY / OPENAI_BASE_URL) are never read here directly: the OpenAI
client picks them up from the environment itself. Nothing secret is ever committed.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemma-4-E4B-it")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
EMBED_CONCURRENCY = _i("EMBED_CONCURRENCY", 2)
LLM_RETRY_ATTEMPTS = _i("LLM_RETRY_ATTEMPTS", 10)  # the class proxy has flaky spells

RAW_DIR = ROOT / "data" / "raw"
DOCS_DIR = RAW_DIR / "provinces"
TABLE_CSV = RAW_DIR / "provinces_table.csv"
INDEX_DIR = Path(os.environ.get("INDEX_DIR", ROOT / "data" / "index"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", ROOT / "data" / "cache"))
SQL_DB_PATH = Path(os.environ.get("SQL_DB_PATH", ROOT / "data" / "provinces.db"))

# Retrieval knobs. Starting values; eval/eval_retrieval.py measures them against alternatives
# (see README "Evaluation" and eval/results/): do not change them without re-running it.
# Default = the configuration the answer evaluation actually measured (BM25; the class
# proxy's embedding endpoint was down during the eval). hybrid/dense are implemented and
# unit-tested; enable with RETRIEVER=hybrid once eval/eval_retrieval.py has measured them.
RETRIEVER = os.environ.get("RETRIEVER", "bm25")  # bm25 | dense | hybrid
CHUNK_WORDS = _i("CHUNK_WORDS", 120)
CHUNK_OVERLAP = _i("CHUNK_OVERLAP", 30)
TOP_K = _i("TOP_K", 5)
MIN_SCORE = _f("MIN_SCORE", 0.45)  # cosine floor on the best dense hit ("not in corpus")

# Agent knobs.
MAX_STEPS = _i("MAX_STEPS", 6)
SQL_ROW_LIMIT = _i("SQL_ROW_LIMIT", 50)
