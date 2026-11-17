"""Build everything the app needs at runtime:
  1. the SQLite statistics table (data/provinces.db) from the committed CSV,
  2. the chunks + their bge-m3 embeddings (cached in data/cache/, so this only calls
     the embedding API for chunks it has never seen).

    python scripts/ingest.py
"""
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config  # noqa: E402
from app.retrieval import build_retriever  # noqa: E402
from app.sqldb import build_db  # noqa: E402  (always rebuild here)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    path = build_db()
    print(f"SQL table -> {path}")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (put it in .env): skipping embeddings.")
        return 1
    t0 = time.time()
    r = build_retriever(dense=True)
    print(f"{len(r.chunks)} chunks embedded with {config.EMBED_MODEL} in {time.time() - t0:.1f}s "
          f"(chunk={config.CHUNK_WORDS} words, overlap={config.CHUNK_OVERLAP})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
