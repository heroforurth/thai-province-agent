"""Structured output: get JSON out of the model and validate it against a schema.

The class proxy ignores `response_format`, so we can't rely on a JSON mode. Instead:
  1. ask for JSON in the prompt,
  2. extract the first balanced {...} object from the reply (models wrap JSON in
     ``` fences or add a sentence before it),
  3. validate it with a Pydantic model,
  4. on failure, show the model its own output + the validation error and ask again
     (parse -> validate -> repair, bounded number of attempts).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

T = TypeVar("T")


class StructuredOutputError(ValueError):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def _balanced_object(text: str, start: int) -> str | None:
    """Return the {...} starting at `start`, respecting strings/escapes, or None."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _light_repair(s: str) -> str:
    """Fix the two most common small-model JSON slips without changing meaning."""
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    s = s.replace("“", '"').replace("”", '"')  # smart quotes
    return s


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply. Raises ValueError if none."""
    text = re.sub(r"```(?:json)?", "", text)
    for m in re.finditer(r"\{", text):
        candidate = _balanced_object(text, m.start())
        if candidate is None:
            continue
        for attempt in (candidate, _light_repair(candidate)):
            try:
                obj = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("no JSON object found in model output")


def parse_as(text: str, schema) -> object:
    """extract_json + validate against a Pydantic model or a TypeAdapter-able type."""
    obj = extract_json(text)
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_validate(obj)
    return TypeAdapter(schema).validate_python(obj)


def complete_structured(
    chat_fn: Callable[[list[dict]], str],
    messages: list[dict],
    schema,
    *,
    max_repairs: int = 1,
) -> tuple[object, str, int]:
    """Call the model, validate its JSON against `schema`, repair on failure.

    Returns (validated_object, raw_text, repairs_used). Raises StructuredOutputError
    if the output is still invalid after `max_repairs` repair turns.
    """
    convo = list(messages)
    raw = chat_fn(convo)
    for attempt in range(max_repairs + 1):
        try:
            return parse_as(raw, schema), raw, attempt
        except (ValueError, ValidationError) as err:
            if attempt == max_repairs:
                raise StructuredOutputError(f"invalid structured output: {err}", raw) from err
            error_text = str(err).splitlines()[0] if isinstance(err, ValueError) else _short(err)
            convo += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your reply was not valid: {error_text}\n"
                        "Reply again with ONLY one JSON object that follows the required format."
                    ),
                },
            ]
            raw = chat_fn(convo)
    raise AssertionError("unreachable")


def _short(err: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()[:3])
