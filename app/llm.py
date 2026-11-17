"""The ONLY module that talks to a model.

`OpenAI()` reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment, so the same
code talks to the class LiteLLM proxy (chat: gemma-4-E4B-it, embeddings: bge-m3).
Everything else in the app calls the three functions below, which also makes them
the single seam that tests replace with fakes.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI

from app import config


log = logging.getLogger(__name__)
T = TypeVar("T")


def _with_retry(fn: Callable[[], T], attempts: int = 6) -> T:
    """The class proxy intermittently answers with a spurious 401 ("column t.tpd_limit
    does not exist" - a proxy-side DB error), which the SDK does not retry. Retry any
    API error a bounded number of times with exponential backoff, then give up."""
    for i in range(attempts):
        try:
            return fn()
        except (APIStatusError, APIConnectionError) as e:
            if i == attempts - 1:
                raise
            wait = min(30, 2 * 2**i)
            log.warning("proxy error (%s), retry %d/%d in %ss", str(e)[:80], i + 1, attempts - 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


@lru_cache(maxsize=1)
def client() -> OpenAI:
    # Created lazily so importing the app (tests, BM25-only eval) never needs a key.
    return OpenAI(timeout=120, max_retries=3)


def chat(messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 700) -> str:
    """One blocking completion. temperature=0 by default: everything we evaluate must
    be repeatable."""
    resp = _with_retry(lambda: client().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    ))
    return (resp.choices[0].message.content or "").strip()


def chat_stream(
    messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 700
) -> Iterator[str]:
    """Yield the reply token by token (used by the SSE endpoint)."""
    stream = _with_retry(lambda: client().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    ))
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def embed(texts: list[str], *, batch_size: int = 64) -> list[list[float]]:
    """Embed texts with bge-m3. Queries and chunks MUST use the same model."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _with_retry(lambda: client().embeddings.create(model=config.EMBED_MODEL, input=batch))
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        log.info("embedded %d/%d", min(i + batch_size, len(texts)), len(texts))
    return out
