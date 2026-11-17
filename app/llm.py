"""The ONLY module that talks to a model.

`OpenAI()` reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment, so the same
code talks to the class LiteLLM proxy (chat: gemma-4-E4B-it, embeddings: bge-m3).
Everything else in the app calls the three functions below, which also makes them
the single seam that tests replace with fakes.
"""
from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from openai import OpenAI

from app import config


@lru_cache(maxsize=1)
def client() -> OpenAI:
    # Created lazily so importing the app (tests, BM25-only eval) never needs a key.
    return OpenAI(timeout=120, max_retries=3)


def chat(messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 700) -> str:
    """One blocking completion. temperature=0 by default: everything we evaluate must
    be repeatable."""
    resp = client().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def chat_stream(
    messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 700
) -> Iterator[str]:
    """Yield the reply token by token (used by the SSE endpoint)."""
    stream = client().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def embed(texts: list[str], *, batch_size: int = 64) -> list[list[float]]:
    """Embed texts with bge-m3. Queries and chunks MUST use the same model."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = client().embeddings.create(model=config.EMBED_MODEL, input=texts[i : i + batch_size])
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out
