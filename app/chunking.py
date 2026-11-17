"""Turn the raw province articles into retrievable chunks.

Two strategies, so the evaluation can compare them (see eval/eval_retrieval.py):

  "fixed"   - the naive default: slide a fixed word window over the whole article.
  "section" - split on the article's headings first, then window inside each section,
              and prefix every chunk with a contextual header "Province > Section".
              Short questions like "what does the name mean?" rarely mention the
              province, but the chunk header always does, which helps both BM25
              and dense retrieval.

Boilerplate paragraphs (identical text repeated in many articles, e.g. the generic
"Human achievement index" blurb whose table was stripped) are dropped at this stage:
they carry no province-specific information and only steal top-k slots.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    url: str
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    url: str
    text: str  # what we show / cite

    @property
    def index_text(self) -> str:  # what we embed / BM25-index
        return f"{self.title} > {self.section}\n{self.text}" if self.section else self.text


def load_docs(docs_dir: Path) -> list[Doc]:
    docs = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip()
        m = re.search(r"^Source: (\S+)", raw, re.M)
        body = raw.split("\n\n", 2)[2] if raw.count("\n\n") >= 2 else raw
        docs.append(Doc(path.stem, title, m.group(1) if m else "", body))
    return docs


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n+", text) if p.strip()]


def _para_key(p: str) -> str:
    """Normalised first ~10 words, so near-identical templated paragraphs collide."""
    words = re.sub(r"[^a-z0-9 ]", " ", p.lower()).split()
    words = [w for w in words if w not in {"the", "a", "an"}]
    return " ".join(words[:10])


def find_boilerplate(docs: list[Doc], min_docs: int = 5) -> set[str]:
    """Keys of paragraphs that appear (near-verbatim) in >= min_docs different articles."""
    seen: Counter[str] = Counter()
    for d in docs:
        seen.update({_para_key(p) for p in _paragraphs(d.body) if not p.startswith("#") and len(p) > 60})
    return {k for k, n in seen.items() if n >= min_docs}


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words] if words else []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
    return out


def _sections(body: str) -> list[tuple[str, str]]:
    """[(section_path, text)] following the markdown headings (##, ###, ####)."""
    out: list[tuple[str, str]] = []
    path: list[str] = []
    buf: list[str] = []
    for line in body.splitlines():
        m = re.match(r"^(#{2,4}) (.+)$", line)
        if m:
            out.append((" > ".join(path), "\n".join(buf)))
            level = len(m.group(1)) - 2  # ## -> 0, ### -> 1, #### -> 2
            path = path[:level] + [m.group(2).strip()]
            buf = []
        else:
            buf.append(line)
    out.append((" > ".join(path), "\n".join(buf)))
    return [(s or "Overview", t) for s, t in out if t.strip()]


def chunk_docs(
    docs: list[Doc],
    *,
    strategy: str = "section",
    size: int = 120,
    overlap: int = 30,
    drop_boilerplate: bool = True,
    header: bool = True,
) -> list[Chunk]:
    boiler = find_boilerplate(docs) if drop_boilerplate else set()
    chunks: list[Chunk] = []
    for d in docs:
        if strategy == "fixed":
            parts = [("", "\n".join(p for p in _paragraphs(d.body) if not p.startswith("#")))]
        elif strategy == "section":
            parts = _sections(d.body)
        else:
            raise ValueError(f"unknown chunking strategy {strategy!r}")
        i = 0
        for section, text in parts:
            paras = [p for p in _paragraphs(text) if _para_key(p) not in boiler]
            words = " ".join(paras).split()
            for win in _windows(words, size, overlap):
                if len(win) < 8:  # a stray caption / table fragment
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{d.doc_id}#{i}",
                        doc_id=d.doc_id,
                        title=d.title,
                        section=section if header else "",
                        url=d.url,
                        text=" ".join(win),
                    )
                )
                i += 1
    return chunks
