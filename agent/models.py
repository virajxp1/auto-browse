from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )


class Interactable(StrictModel):
    kind: Literal["input", "button", "link"]
    label: str
    selector: str
    href: str | None = None


class PageState(StrictModel):
    url: str
    title: str
    markdown: str
    interactables: list[Interactable]


class AgentDecision(StrictModel):
    action: Literal["extract", "type_and_submit", "click", "navigate", "fail"]
    step_summary: str
    next_step: str
    reason: str | None = None

    answer: str | None = None
    evidence: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

    selector: str | None = None
    text: str | None = None

    url: str | None = None


class AgentStepTrace(StrictModel):
    step: int
    url: str
    title: str
    decision: AgentDecision


class AgentResult(StrictModel):
    answer: str | None = None
    source_url: str | None = None
    evidence: str | None = None
    confidence: float | None = None
    error: str | None = None
    trace: list[AgentStepTrace]
