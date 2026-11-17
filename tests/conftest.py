import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fake_embed(texts):
    """Deterministic bag-of-words hashing embedding: texts sharing words are similar.
    Lets us test dense/hybrid retrieval and the min_score floor with no API."""
    out = []
    for t in texts:
        v = np.zeros(256, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
        out.append(v.tolist())
    return out


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


class ScriptedChat:
    """A fake chat model that returns pre-written replies in order and records calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, **kw):
        self.calls.append(messages)
        if not self.replies:
            return '{"action": "final_answer", "args": {"answer": "out of script"}}'
        return self.replies.pop(0)
