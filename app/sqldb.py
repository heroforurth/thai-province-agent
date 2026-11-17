"""The structured half of the corpus: a SQLite table of province statistics, plus a
*guarded* query runner the agent's text-to-SQL tool uses.

Model-written SQL is untrusted input. Defence in depth:
  1. textual pre-check: one statement, must start with SELECT/WITH, no PRAGMA/ATTACH;
  2. the connection is opened read-only (mode=ro), so writes fail at the engine level;
  3. an SQLite authorizer allow-lists the operations (SELECT, READ, FUNCTION) and only
     the `provinces` table, so even a clever query can't touch anything else;
  4. a progress handler aborts runaway queries; results are capped at N rows.
"""
from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

from app import config

SCHEMA = """CREATE TABLE provinces(
  doc_id TEXT PRIMARY KEY,      -- slug, e.g. 'chiang-mai' (matches the article id)
  name TEXT NOT NULL,           -- English name, e.g. 'Chiang Mai'
  name_thai TEXT,
  region TEXT NOT NULL,         -- six-region system: Northern, Northeastern, Central, Eastern, Western, Southern
  capital TEXT,
  population INTEGER,           -- registered population, December 2024
  area_km2 INTEGER,
  density_per_km2 INTEGER,      -- people per km2
  iso_code TEXT                 -- ISO 3166-2, e.g. 'TH-50'
)"""

SCHEMA_FOR_PROMPT = re.sub(r"\s+", " ", SCHEMA)
ALLOWED_TABLES = {"provinces"}
_FORBIDDEN = re.compile(r"\b(pragma|attach|detach|insert|update|delete|drop|alter|create|replace|vacuum|reindex)\b", re.I)


class SqlError(ValueError):
    pass


def build_db(path: Path = config.SQL_DB_PATH, csv_path: Path = config.TABLE_CSV) -> Path:
    """(Re)create the SQLite file from the committed CSV. Idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.execute(SCHEMA)
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    db.executemany(
        "INSERT INTO provinces VALUES(?,?,?,?,?,?,?,?,?)",
        [
            (r["doc_id"], r["name"], r["name_thai"], r["region"], r["capital"],
             int(r["population"]), int(r["area_km2"]), int(r["density_per_km2"]), r["iso_code"])
            for r in rows
        ],
    )
    db.commit()
    db.close()
    return path


def ensure_db(path: Path = config.SQL_DB_PATH, csv_path: Path = config.TABLE_CSV) -> Path:
    """Build the SQLite file only if it is missing or older than the CSV."""
    if not path.exists() or path.stat().st_mtime < csv_path.stat().st_mtime:
        build_db(path, csv_path)
    return path


def _authorizer(action, arg1, arg2, dbname, source):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if arg1 in ALLOWED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def connect_readonly(path: Path = config.SQL_DB_PATH) -> sqlite3.Connection:
    ensure_db(path)
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    db.set_authorizer(_authorizer)
    return db


def check_sql(sql: str) -> str:
    """Textual allow-list checks. Returns the cleaned single statement."""
    s = sql.strip().rstrip(";").strip()
    s = re.sub(r"^```(?:sql)?|```$", "", s, flags=re.I).strip()
    if ";" in s:
        raise SqlError("only a single SQL statement is allowed")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise SqlError("only SELECT queries are allowed")
    if _FORBIDDEN.search(s):
        raise SqlError("query contains a forbidden keyword")
    return s


def run_query(db: sqlite3.Connection, sql: str, *, row_limit: int = config.SQL_ROW_LIMIT) -> dict:
    s = check_sql(sql)
    steps = {"n": 0}

    def _guard():  # called every 1000 VM instructions
        steps["n"] += 1
        return 1 if steps["n"] > 5000 else 0  # non-zero aborts

    db.set_progress_handler(_guard, 1000)
    try:
        cur = db.execute(s)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(row_limit + 1)
    except sqlite3.Error as e:
        raise SqlError(f"SQL error: {e}") from e
    finally:
        db.set_progress_handler(None, 0)
    truncated = len(rows) > row_limit
    return {"columns": cols, "rows": [list(r) for r in rows[:row_limit]], "truncated": truncated}
