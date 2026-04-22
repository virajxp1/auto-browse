from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PRICE_PATTERN = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d{2})?)|(?:\d[\d,]*(?:\.\d{2})?\s?(?:usd|eur|gbp))",
    re.IGNORECASE,
)


PageArchetype = Literal[
    "generic",
    "form",
    "search_results",
    "product_detail",
    "quote",
    "travel_search",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )


class Interactable(StrictModel):
    ref: str | None = None
    kind: Literal["input", "button", "link", "select", "checkbox", "radio"]
    label: str
    selector: str
    href: str | None = None
    field_type: str | None = None
    options: list[str] | None = None
    checked: bool | None = None
    region: Literal["main", "form", "body", "header", "nav", "footer", "aside"] | None = None
    context_text: str | None = None


class PageState(StrictModel):
    url: str
    title: str
    markdown: str
    interactables: list[Interactable]
    page_archetype: PageArchetype = "generic"
    page_hints: list[str] = Field(default_factory=list)


class AgentDecision(StrictModel):
    action: Literal[
        "extract",
        "analyze",
        "verify",
        "complete",
        "type_and_submit",
        "fill",
        "submit",
        "select_option",
        "check",
        "wait_for",
        "click",
        "navigate",
        "fail",
    ]
    step_summary: str
    next_step: str
    reason: str | None = None
    status: Literal["completed", "blocked", "requires_user_input", "requires_approval"] | None = None
    goal_summary: str | None = None
    result_data: dict[str, Any] | None = None
    analysis: str | None = None
    verified: bool | None = None

    answer: str | None = None
    structured_data: dict[str, str | None] | None = None
    evidence: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

    interactable_ref: str | None = None
    selector: str | None = None
    text: str | None = None
    value: str | None = None
    wait_state: Literal["attached", "visible", "hidden", "detached"] | None = None
    timeout_ms: int | None = None

    url: str | None = None


class AgentStepTrace(StrictModel):
    step: int
    url: str
    title: str
    decision: AgentDecision


class AgentResult(StrictModel):
    status: Literal["completed", "blocked", "requires_user_input", "requires_approval", "failed"] | None = None
    goal_summary: str | None = None
    result_data: dict[str, Any] | None = None
    answer: str | None = None
    structured_data: dict[str, str | None] | None = None
    source_url: str | None = None
    final_url: str | None = None
    final_title: str | None = None
    evidence: str | None = None
    confidence: float | None = None
    error: str | None = None
    trace: list[AgentStepTrace]
