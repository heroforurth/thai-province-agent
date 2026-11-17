"""A content-addressed embedding cache (SQLite) in front of llm.embed.

Key = sha1(model + text). Re-chunking, sweeping chunk sizes in the evaluation, or
restarting the server only embeds texts that were never seen before, so experiments
are cheap and the numbers are reproducible run-to-run.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app import config

EmbedFn = Callable[[list[str]], list[list[float]]]


class EmbeddingCache:
    def __init__(self, path: Path | None = None, model: str | None = None):
        self.model = model or config.EMBED_MODEL
        path = path or config.CACHE_DIR / f"embeddings_{self.model.replace('/', '_')}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb(key TEXT PRIMARY KEY, vec BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}\x00{text}".encode()).hexdigest()

    def get_many(self, texts: list[str], embed_fn: EmbedFn) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors, embedding misses."""
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                batch = keys[i : i + 500]
                q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
                for k, blob in self.db.execute(q, batch):
                    found[k] = np.frombuffer(blob, dtype=np.float32)
        missing = [(k, t) for k, t in zip(keys, texts) if k not in found]
        if missing:
            uniq = dict(missing)  # dedupe identical texts
            vecs = embed_fn(list(uniq.values()))
            rows = []
            for k, v in zip(uniq.keys(), vecs):
                arr = np.asarray(v, dtype=np.float32)
                found[k] = arr
                rows.append((k, arr.tobytes()))
            with self._lock:
                self.db.executemany("INSERT OR REPLACE INTO emb VALUES(?, ?)", rows)
                self.db.commit()
        mat = np.stack([found[k] for k in keys]) if keys else np.zeros((0, 1), np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1, norms)
