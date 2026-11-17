"""Pydantic schemas for everything that crosses a boundary:
  - what the model is allowed to emit in the agent loop (tool calls / final answer),
  - the arguments each tool accepts (validated BEFORE the tool runs),
  - the HTTP request/response bodies of the API.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- tool arguments


class SearchDocsArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(default=4, ge=1, le=8)


class RunSqlArgs(BaseModel):
    sql: str = Field(min_length=6, max_length=1000)


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


# ---------------------------------------------------------------- agent actions
# The model must reply with exactly ONE of these JSON objects per turn. The
# discriminator `action` doubles as the tool allow-list: anything else fails
# validation and is never executed.


class _Step(BaseModel):
    thought: str = Field(default="", max_length=600)


class SearchDocsCall(_Step):
    action: Literal["search_docs"]
    args: SearchDocsArgs


class RunSqlCall(_Step):
    action: Literal["run_sql"]
    args: RunSqlArgs


class CalculatorCall(_Step):
    action: Literal["calculator"]
    args: CalculatorArgs


class FinalAnswerArgs(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)
    citations: list[int] = Field(default_factory=list, max_length=10)
    used_sql: bool = False

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce(cls, v):
        # Small models sometimes write ["1", "[2]"]; accept them but keep ints only.
        out = []
        for x in v or []:
            s = str(x).strip("[] ")
            if s.isdigit():
                out.append(int(s))
        return out


class FinalAnswer(_Step):
    action: Literal["final_answer"]
    args: FinalAnswerArgs


AgentAction = Annotated[
    Union[SearchDocsCall, RunSqlCall, CalculatorCall, FinalAnswer],
    Field(discriminator="action"),
]

# ---------------------------------------------------------------- API


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    mode: Literal["agent", "rag"] = "agent"


class Source(BaseModel):
    n: int
    doc_id: str
    title: str
    section: str
    url: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    retriever: str
    refused: bool
    sources: list[Source]
