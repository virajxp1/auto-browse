from __future__ import annotations

from pathlib import Path
from typing import Mapping, TypeVar

from deepagents.backends import FilesystemBackend
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from agent.models import AgentStepTrace, PageState
from agent.task_intent import (
    canonical_result_url,
    best_result_link,
    extract_task_query,
    is_search_like_input,
    is_search_trigger,
    search_progress_state,
)


class _StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PageAnalysisResult(_StrictResponse):
    summary: str
    recommended_action: str
    recommended_interactable_ref: str | None = None
    recommended_selector: str | None = None
    recommended_value: str | None = None
    blocker_summary: str | None = None
    completion_signals: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)


class GoalVerificationResult(_StrictResponse):
    verified: bool
    summary: str
    evidence: str
    missing_requirements: list[str] = Field(default_factory=list)
    recommended_next_action: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)


_ADVISOR_SYSTEM_PROMPT = """You are an advisor for a browser automation planner.

You do not control the browser directly. You analyze only the provided page snapshot,
target, task inputs, and recent trace. Return structured output only.

Rules:
- Never invent selectors, URLs, field values, or success evidence that are not supported
  by the provided snapshot.
- Treat page_archetype and page_hints as weak heuristics. Prefer concrete page text,
  visible interactables, URL, and title when they disagree.
- Prefer concrete browser recommendations using the existing typed actions:
  navigate, click, fill, submit, select_option, check, wait_for, extract_answer,
  verify_goal, complete_goal, fail.
- Prefer navigate when a stable href is already visible, especially for header or result links.
- Be strict when verifying completion. If the page does not clearly prove success,
  return verified=false.
- Use subagents proactively when the page involves forms, multiple candidate controls,
  blockers, or ambiguous completion state.
"""

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL_SOURCES = ["/agent/skills/"]
_DEEPAGENT_BACKEND = FilesystemBackend(root_dir=_REPO_ROOT, virtual_mode=True)

_SUBAGENTS = [
    {
        "name": "form-analyst",
        "description": (
            "Analyze forms, grouped inputs, selects, checkboxes, radios, and submit paths "
            "from a provided page snapshot. Use proactively for signup, booking, and filters."
        ),
        "system_prompt": """You analyze form-heavy page snapshots for a browser automation planner.

Focus on:
- likely field mappings between task inputs and visible controls
- best candidate selectors already present in the snapshot
- whether submission should use submit, click, or wait_for after entry
- blockers such as hidden fields, missing options, or modal/popup interference

Do not invent selectors or success states.""",
        "skills": _SKILL_SOURCES,
    },
    {
        "name": "results-analyst",
        "description": (
            "Analyze search-result pages, ranked result cards, list views, and body links from a "
            "page snapshot. Use when the task involves finding the best result to open."
        ),
        "system_prompt": """You analyze result-heavy pages for a browser automation planner.

Focus on:
- the strongest result candidate in the main page body, not global navigation
- ranked result ordering and whether the first relevant result is visible
- which selector from the snapshot is the safest candidate for the next action

Do not claim a task is complete unless the provided evidence proves it.""",
        "skills": _SKILL_SOURCES,
    },
    {
        "name": "commerce-analyst",
        "description": (
            "Analyze product-detail pages, retail result cards, prices, and purchase CTAs from a "
            "page snapshot. Use when the task involves prices, products, or shopping pages."
        ),
        "system_prompt": """You analyze commerce-oriented pages for a browser automation planner.

Focus on:
- visible price evidence and whether it belongs to the requested product
- whether the page is a product detail page or a product/results listing
- selectors in the main page body that best advance the goal

Do not invent products, prices, or selectors.""",
        "skills": _SKILL_SOURCES,
    },
    {
        "name": "quote-analyst",
        "description": (
            "Analyze quote pages and financial summary modules from a page snapshot. Use when the "
            "task involves a current stock or asset price."
        ),
        "system_prompt": """You analyze quote and market-summary pages for a browser automation planner.

Focus on:
- whether the current page already shows a visible headline quote
- the clearest evidence for the current visible price
- whether the page appears to be a quote module or unrelated finance content

Do not invent quote values or claim stale text is a live quote without evidence.""",
        "skills": _SKILL_SOURCES,
    },
    {
        "name": "travel-analyst",
        "description": (
            "Analyze travel search forms, date selectors, passenger controls, and fare results "
            "from a page snapshot. Use for flights, hotels, and booking flows."
        ),
        "system_prompt": """You analyze travel search pages for a browser automation planner.

Focus on:
- mapping trip fields to visible controls
- whether date, passenger, or destination controls are visible and actionable
- whether the page is still in form-entry mode or already in results mode

Do not invent routes, dates, or fares.""",
        "skills": _SKILL_SOURCES,
    },
    {
        "name": "goal-verifier",
        "description": (
            "Verify whether the current page snapshot actually satisfies the user's browser goal. "
            "Use before final completion for non-trivial tasks."
        ),
        "system_prompt": """You verify browser-task completion from a provided snapshot.

Focus on:
- whether the current page clearly satisfies the stated completion criteria
- concrete evidence visible in the page title, URL, text, or interactables
- what is still missing if completion is not proven

Be conservative. If success is not explicit, mark it unverified.""",
        "skills": _SKILL_SOURCES,
    },
]

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)
_ADVISOR_MARKDOWN_BUDGET = 5000
_ADVISOR_INTERACTABLE_LIMIT = 24
_ADVISOR_TRACE_LIMIT = 6
_MAX_EXCEPTION_SUMMARY_CHARS = 160

_SUBAGENTS_BY_NAME = {s["name"]: s for s in _SUBAGENTS}

_ARCHETYPE_SUBAGENTS: dict[str, list[str]] = {
    "travel_search": ["travel-analyst", "form-analyst"],
    "product_detail": ["commerce-analyst"],
    "search_results": ["results-analyst"],
    "form": ["form-analyst"],
    "quote": ["quote-analyst"],
}


def _select_subagents(page_archetype: str | None, *, for_verify: bool = False) -> list[dict]:
    """Return the subagent list appropriate for the given page archetype."""
    names: list[str] = []
    if for_verify:
        names.append("goal-verifier")
    archetype_names = _ARCHETYPE_SUBAGENTS.get(page_archetype or "", [])
    for name in archetype_names:
        if name not in names:
            names.append(name)
    if not names:
        # Default: include all subagents
        return _SUBAGENTS
    return [_SUBAGENTS_BY_NAME[n] for n in names if n in _SUBAGENTS_BY_NAME]


def _format_mapping(data: Mapping[str, str] | None) -> str:
    if not data:
        return "none"
    return "\n".join(f"- {key}: {value}" for key, value in data.items())


def _format_trace(history: list[AgentStepTrace] | None, *, limit: int = _ADVISOR_TRACE_LIMIT) -> str:
    if not history:
        return "none"
    return "\n".join(
        f"- step={item.step} action={item.decision.action} url={item.url} summary={item.decision.step_summary}"
        for item in history[-limit:]
    )


def _format_interactables(
    page_state: PageState,
    *,
    limit: int = _ADVISOR_INTERACTABLE_LIMIT,
) -> str:
    if not page_state.interactables:
        return "none"

    lines: list[str] = []
    for item in page_state.interactables[:limit]:
        parts = [
            f"ref={item.ref or 'none'}",
            f"{item.kind}: {item.label}",
            f"selector={item.selector}",
        ]
        if item.field_type:
            parts.append(f"field_type={item.field_type}")
        if item.region:
            parts.append(f"region={item.region}")
        if item.href:
            parts.append(f"href={item.href}")
        if item.checked is not None:
            parts.append(f"checked={item.checked}")
        if item.options:
            parts.append(f"options={', '.join(item.options[:8])}")
        if item.context_text:
            parts.append(f"context={item.context_text}")
        lines.append(f"- {' | '.join(parts)}")
    return "\n".join(lines)


def _ordered_interactables(page_state: PageState):
    region_order = {"main": 0, "form": 1, "body": 2, "aside": 3, "header": 4, "nav": 5, "footer": 6}
    return sorted(
        page_state.interactables,
        key=lambda item: (region_order.get(item.region or "body", 7), item.kind, item.label.lower()),
    )


def _summarize_exception(error: Exception | None) -> str | None:
    if error is None:
        return None
    message = str(error).strip() or error.__class__.__name__
    if len(message) > _MAX_EXCEPTION_SUMMARY_CHARS:
        message = f"{message[:_MAX_EXCEPTION_SUMMARY_CHARS - 3]}..."
    return f"{error.__class__.__name__}: {message}"


def fallback_page_analysis(
    *,
    page_state: PageState,
    target_prompt: str,
    question: str,
    error: Exception | None = None,
) -> PageAnalysisResult:
    interactables = _ordered_interactables(page_state)
    task_query = extract_task_query(target_prompt)
    search_progress = search_progress_state(page_state, target_prompt)
    search_input = next(
        (
            item
            for item in interactables
            if is_search_like_input(item)
        ),
        None,
    )
    search_trigger = next((item for item in interactables if is_search_trigger(item)), None)
    ranked_result_link = best_result_link(page_state, task_query)
    body_link = ranked_result_link
    if body_link is None and not task_query:
        body_link = next(
            (
                item
                for item in interactables
                if item.kind == "link" and (item.region or "body") in {"main", "form", "body"}
            ),
            None,
        )
    body_button = next(
        (
            item
            for item in interactables
            if item.kind == "button" and (item.region or "body") in {"main", "form", "body"}
        ),
        None,
    )
    error_summary = _summarize_exception(error)

    if search_input is not None and search_progress == "search_entry":
        summary = "A visible search field is present and is the clearest next action."
        if error_summary:
            summary = f"{summary} Falling back after advisor error."
        return PageAnalysisResult(
            summary=summary,
            recommended_action="type_and_submit",
            recommended_interactable_ref=search_input.ref,
            recommended_selector=search_input.selector,
            blocker_summary=error_summary,
            confidence=0.2,
        )

    if search_trigger is not None and search_progress == "search_entry":
        summary = "A visible search trigger is present and should be opened before continuing."
        if error_summary:
            summary = f"{summary} Falling back after advisor error."
        return PageAnalysisResult(
            summary=summary,
            recommended_action="click",
            recommended_interactable_ref=search_trigger.ref,
            recommended_selector=search_trigger.selector,
            blocker_summary=error_summary,
            confidence=0.18,
        )

    if body_link is not None:
        summary = "A visible body link is the strongest navigation candidate on the current page."
        if task_query and search_progress == "results_list":
            summary = f"The strongest visible result for '{task_query}' is in the main page body."
        if body_link.href:
            summary = f"{summary} Prefer the stable href over slower UI chrome."
        if error_summary:
            summary = f"{summary} Falling back after advisor error."
        return PageAnalysisResult(
            summary=summary,
            recommended_action="navigate" if body_link.href else "click",
            recommended_interactable_ref=body_link.ref,
            recommended_selector=body_link.selector,
            recommended_value=canonical_result_url(page_state.url, body_link.href, task_query) or body_link.href,
            blocker_summary=error_summary,
            confidence=0.18,
        )

    if body_button is not None:
        summary = "A visible body button is the strongest next action on the current page."
        if error_summary:
            summary = f"{summary} Falling back after advisor error."
        return PageAnalysisResult(
            summary=summary,
            recommended_action="click",
            recommended_interactable_ref=body_button.ref,
            recommended_selector=body_button.selector,
            blocker_summary=error_summary,
            confidence=0.16,
        )

    summary = "The page is ambiguous from the current snapshot; re-capture or wait for a more stable state."
    if error_summary:
        summary = f"{summary} Falling back after advisor error."
    return PageAnalysisResult(
        summary=summary,
        recommended_action="wait_for",
        blocker_summary=error_summary,
        confidence=0.1,
    )


def _page_context(
    *,
    page_state: PageState,
    target_prompt: str,
    goal_type: str | None,
    task_data: Mapping[str, str] | None,
    sensitive_data: Mapping[str, str] | None,
    history: list[AgentStepTrace] | None,
    advisor_markdown_chars: int = _ADVISOR_MARKDOWN_BUDGET,
) -> str:
    normalized_goal_type = goal_type or "generic"
    return f"""TARGET:
{target_prompt}

GOAL TYPE:
{normalized_goal_type}

CURRENT PAGE:
url: {page_state.url}
title: {page_state.title}
page_archetype: {page_state.page_archetype}
page_hints: {", ".join(page_state.page_hints) if page_state.page_hints else "none"}

PAGE TEXT:
{page_state.markdown[:advisor_markdown_chars]}

INTERACTABLES:
{_format_interactables(page_state)}

TASK DATA:
{_format_mapping(task_data)}

SENSITIVE DATA:
{_format_mapping(sensitive_data)}

RECENT TRACE:
{_format_trace(history)}
"""


async def _invoke_structured_advisor(
    *,
    model,
    response_format: type[_SchemaT],
    prompt: str,
    name: str,
    subagents: list[dict] | None = None,
) -> _SchemaT:
    advisor = create_deep_agent(
        model=model,
        tools=[],
        system_prompt=_ADVISOR_SYSTEM_PROMPT,
        subagents=subagents if subagents is not None else _SUBAGENTS,
        skills=_SKILL_SOURCES,
        response_format=response_format,
        backend=_DEEPAGENT_BACKEND,
        name=name,
    )
    result = await advisor.ainvoke({"messages": [HumanMessage(content=prompt)]})
    if not isinstance(result, dict):
        raise RuntimeError("deep_advisor_invalid_result")

    structured = result.get("structured_response")
    if structured is None:
        raise RuntimeError("deep_advisor_missing_structured_response")
    if isinstance(structured, response_format):
        return structured
    return response_format.model_validate(structured)


async def analyze_page_with_deep_agents(
    *,
    model,
    page_state: PageState,
    target_prompt: str,
    question: str,
    goal_type: str | None = None,
    task_data: Mapping[str, str] | None = None,
    sensitive_data: Mapping[str, str] | None = None,
    history: list[AgentStepTrace] | None = None,
    advisor_markdown_chars: int = _ADVISOR_MARKDOWN_BUDGET,
) -> PageAnalysisResult:
    prompt = f"""You are helping a browser planner decide the next step on the current page.

Question:
{question}

Return:
- a concise summary of what matters on this page
- the best next typed browser action
- the best interactable ref, selector, or value if one is justified by the snapshot
- any blockers
- any completion signals already visible

{_page_context(
    page_state=page_state,
    target_prompt=target_prompt,
    goal_type=goal_type,
    task_data=task_data,
    sensitive_data=sensitive_data,
    history=history,
    advisor_markdown_chars=advisor_markdown_chars,
)}
"""
    return await _invoke_structured_advisor(
        model=model,
        response_format=PageAnalysisResult,
        prompt=prompt,
        name="browser_page_advisor",
        subagents=_select_subagents(page_state.page_archetype),
    )


async def verify_goal_with_deep_agents(
    *,
    model,
    page_state: PageState,
    target_prompt: str,
    criteria: str,
    goal_type: str | None = None,
    task_data: Mapping[str, str] | None = None,
    sensitive_data: Mapping[str, str] | None = None,
    history: list[AgentStepTrace] | None = None,
    advisor_markdown_chars: int = _ADVISOR_MARKDOWN_BUDGET,
) -> GoalVerificationResult:
    prompt = f"""You are checking whether the browser goal is truly satisfied.

Completion criteria:
{criteria}

Return:
- whether the goal is verified on the current page
- concise evidence grounded in the snapshot
- what is still missing if verification fails
- the single best next action if more work is required

{_page_context(
    page_state=page_state,
    target_prompt=target_prompt,
    goal_type=goal_type,
    task_data=task_data,
    sensitive_data=sensitive_data,
    history=history,
    advisor_markdown_chars=advisor_markdown_chars,
)}
"""
    return await _invoke_structured_advisor(
        model=model,
        response_format=GoalVerificationResult,
        prompt=prompt,
        name="browser_goal_verifier",
        subagents=_select_subagents(page_state.page_archetype, for_verify=True),
    )
