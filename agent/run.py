from __future__ import annotations

import asyncio
from contextlib import suppress
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, TypedDict
from urllib.parse import urljoin, urlsplit

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from playwright.async_api import Page

from agent.browser import capture_state, goto_with_fallback, run_browser
from agent.browser_actions import (
    FALLBACK_SELECTOR_WAIT_MS,
    PRIMARY_SELECTOR_WAIT_MS,
    check_fallback_selectors,
    click_fallback_selectors,
    click_single_visible_link,
    click_via_css_fallback,
    click_via_text_heuristic,
    extract_selector_hint,
    selector_has_selected_value,
    selector_is_checked,
    selector_value_matches,
    select_fallback_selectors,
    try_check_selector,
    try_click_selector,
    try_fill_selector,
    try_select_option_selector,
    try_submit_selector,
    try_type_and_submit_selector,
    try_wait_for_selector,
    type_and_submit_via_text_heuristic,
    type_fallback_selectors,
    wait_for_action_effect,
    wait_short,
)
from agent.deep_advisor import (
    analyze_page_with_deep_agents,
    fallback_page_analysis,
    verify_goal_with_deep_agents,
)
from agent.extract import page_to_markdown
from agent.models import AgentDecision, AgentResult, AgentStepTrace, Interactable, PageState
from agent.observability import export_current_span_parent, flush, span_log, start_span
from agent.openrouter_client import OpenRouterClient
from agent.context_budget import compute_context_budget
from agent.memory import AgentScratchpad, format_scratchpad, update_scratchpad
from agent.planner import build_llm_messages
from agent.snapshot import PageSnapshotService, budget_page_state, capture_dom_signature
from agent.task_intent import (
    best_result_link,
    extract_task_query,
    is_search_like_input,
    page_identity_matches_query,
    page_matches_query,
    query_tokens,
    search_progress_state,
    task_requires_destination_page,
)
from agent.tool_args import (
    AnalyzePageArgs,
    CheckArgs,
    ClickArgs,
    CompleteGoalArgs,
    ExtractAnswerArgs,
    FailArgs,
    FillArgs,
    NavigateArgs,
    SelectOptionArgs,
    SubmitArgs,
    TypeAndSubmitArgs,
    VerifyGoalArgs,
    WaitForArgs,
)

StepCallback = Callable[[AgentStepTrace], None]
_SNAPSHOT_CAPTURE_TIMEOUT_SECONDS = 15.0
_DEEP_ADVISOR_TIMEOUT_SECONDS = 12.0
_LLM_CALL_TIMEOUT_SECONDS = 20.0
_SMALL_TARGET_TOKEN_SET_SIZE = 2

@dataclass
class _Runtime:
    openrouter_client: OpenRouterClient
    page: Page
    snapshot_service: PageSnapshotService
    start_url: str
    target_prompt: str
    goal_type: str | None
    task_data: Mapping[str, str] | None
    sensitive_data: Mapping[str, str] | None
    max_steps: int
    max_actions_per_step: int
    extraction_schema: dict[str, str] | None
    extraction_selector: str | None
    on_step: StepCallback | None
    trace_id: str
    current_trace: list[AgentStepTrace] = field(default_factory=list)
    current_page_state: PageState | None = None
    last_verification_passed: bool | None = None
    last_verification_url: str | None = None
    scratchpad: AgentScratchpad = field(default_factory=AgentScratchpad)


class ActionObservation(TypedDict):
    url: str
    title: str


class AgentGraphState(TypedDict):
    step: int
    trace: list[AgentStepTrace]
    page_state: PageState | None
    result: AgentResult | None
    messages: list[AIMessage | ToolMessage]
    action_observations: list[ActionObservation]


def _set_error(state: AgentGraphState, error: str) -> AgentGraphState:
    state["result"] = AgentResult(status="failed", error=error, trace=state["trace"])
    return state


def _advance(state: AgentGraphState) -> AgentGraphState:
    state["step"] += 1
    state["page_state"] = None
    state["messages"] = []
    state["action_observations"] = []
    return state


async def _wait_domcontentloaded(page: Page, timeout: int = 10000) -> None:
    with suppress(Exception):
        await page.wait_for_load_state("domcontentloaded", timeout=timeout)


async def _capture_page_observation(
    page: Page,
    *,
    fallback_url: str = "",
    fallback_title: str = "",
) -> ActionObservation:
    current_url = getattr(page, "url", None) or fallback_url
    current_title = fallback_title
    title_fn = getattr(page, "title", None)
    if callable(title_fn):
        try:
            current_title = await title_fn()
        except Exception:
            current_title = fallback_title
    return {"url": current_url, "title": current_title}


def _strip_markdown_artifacts(value: str) -> str:
    text = _normalize_spacing(value)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    if text.startswith("* "):
        text = text[2:].strip()
    return text


def _normalize_spacing(value: str) -> str:
    text = value.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2007", " ")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_tool_text(value: str) -> str:
    return _normalize_spacing(value)


def _error_detail(error: Exception, *, limit: int = 180) -> str:
    message = _normalize_tool_text(str(error)) if str(error).strip() else error.__class__.__name__
    detail = f"{error.__class__.__name__}: {message}"
    if len(detail) > limit:
        return f"{detail[: limit - 3]}..."
    return detail


def _normalized_page_identity(url: str) -> str:
    parsed = urlsplit(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _generic_url_tokens(url: str) -> set[str]:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").replace("www.", " ")
    parts = re.split(r"[^a-z0-9]+", f"{host} {parsed.path}".lower())
    return {
        token
        for token in parts
        if len(token) > 1
        and token
        not in {"com", "org", "net", "www", "http", "https", "en", "us", "new", "index"}
    }


def _url_target_overlap(current_url: str, current_title: str, target_url: str) -> bool:
    parsed_target = urlsplit(target_url)
    path_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", parsed_target.path.lower())
        if len(token) > 1 and token not in {"en", "us", "new", "index"}
    }
    target_tokens = path_tokens or _generic_url_tokens(target_url)
    if not target_tokens:
        return False
    current_tokens = _generic_url_tokens(current_url) | query_tokens(current_title)
    overlap = len(target_tokens & current_tokens)
    required = 1 if len(target_tokens) <= _SMALL_TARGET_TOKEN_SET_SIZE else 2
    return overlap >= required


def _prompt_requests_title(target_prompt: str, extraction_schema: dict[str, str] | None) -> bool:
    prompt = target_prompt.lower()
    if re.search(r"\b(title|heading|headline|page title)\b", prompt):
        return True

    if extraction_schema:
        for field_name, description in extraction_schema.items():
            combined = f"{field_name} {description}".lower()
            if re.search(r"\b(title|heading|headline|name)\b", combined):
                return True
    return False


def _tool_arg_preview(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        normalized = _normalize_tool_text(value)
        if len(normalized) > 160:
            return f"{normalized[:157]}..."
        return normalized
    if isinstance(value, list):
        return [_tool_arg_preview(item) for item in value[:4]]
    if isinstance(value, dict):
        return {str(key): _tool_arg_preview(item) for key, item in list(value.items())[:8]}
    return type(value).__name__


def _tool_call_summaries(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for call in tool_calls:
        tool_name = call.get("name")
        tool_args = call.get("args")
        summary: dict[str, Any] = {"name": str(tool_name) if tool_name is not None else ""}
        if isinstance(tool_args, dict):
            summary["args"] = _tool_arg_preview(tool_args)
        summaries.append(summary)
    return summaries


def _normalize_confidence_value(confidence: float | int | str | None) -> float | None:
    if confidence is None:
        return None
    if isinstance(confidence, bool):
        return 1.0 if confidence else 0.0
    if isinstance(confidence, (int, float)):
        value = float(confidence)
    else:
        normalized = _normalize_tool_text(confidence).lower()
        keyword_values = {"high": 0.85, "medium": 0.5, "low": 0.2}
        if normalized in keyword_values:
            value = keyword_values[normalized]
        else:
            percent = normalized.endswith("%")
            if percent:
                normalized = normalized[:-1].strip()
            try:
                value = float(normalized)
            except ValueError:
                return None
            if percent or value > 1:
                value = value / 100.0

    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _extract_markdown_table_rows(markdown: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        key = _strip_markdown_artifacts(cells[0])
        value = _strip_markdown_artifacts(" | ".join(cells[1:]))
        if not key or not value:
            continue
        if set(key) <= {"-"} or set(value) <= {"-"}:
            continue
        rows.append((key, value))
    return rows


def _field_aliases(field_name: str, field_description: str) -> list[str]:
    base = re.sub(r"[_-]+", " ", field_name).strip().lower()
    aliases: set[str] = {base}
    if base.endswith(" date"):
        aliases.add(base.replace(" date", ""))
    if base.endswith(" name"):
        aliases.add(base.replace(" name", ""))
    if base.endswith(" title"):
        aliases.add(base.replace(" title", ""))
    if base.endswith(" url"):
        aliases.add(base.replace(" url", ""))

    normalized_description = re.sub(r"[^a-z0-9]+", " ", field_description.lower()).strip()
    if normalized_description and len(normalized_description.split()) <= 4:
        aliases.add(normalized_description)

    cleaned_aliases = [
        alias.strip()
        for alias in aliases
        if alias.strip()
    ]
    cleaned_aliases.sort(key=len, reverse=True)
    return cleaned_aliases


def _match_tokens(value: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if not normalized:
        return set()

    stopwords = {"a", "an", "and", "by", "for", "in", "of", "on", "the", "to"}
    tokens: set[str] = set()
    for word in normalized.split():
        if len(word) <= 1 or word in stopwords:
            continue
        tokens.add(word)
        if len(word) > 4 and word.endswith("ed"):
            tokens.add(word[:-2])
        if len(word) > 5 and word.endswith("ing"):
            tokens.add(word[:-3])
        if len(word) > 4 and word.endswith(("er", "or")):
            tokens.add(word[:-2])
        if len(word) > 4 and word.endswith("es"):
            tokens.add(word[:-2])
        if len(word) > 3 and word.endswith("s"):
            tokens.add(word[:-1])
    return tokens


def _match_table_value(rows: list[tuple[str, str]], aliases: list[str]) -> tuple[str | None, str | None]:
    best_score = 0
    best_value: str | None = None
    best_key: str | None = None
    for alias in aliases:
        alias_lower = alias.lower()
        alias_tokens = _match_tokens(alias_lower)
        for key, value in rows:
            key_lower = key.lower()
            score = 0
            if key_lower == alias_lower:
                score = 3
            elif key_lower.startswith(f"{alias_lower} "):
                score = 2
            elif alias_lower in key_lower:
                score = 1
            elif alias_tokens:
                overlap = alias_tokens & _match_tokens(key_lower)
                if overlap == alias_tokens:
                    score = 2
                elif overlap:
                    score = 1

            if score > best_score:
                best_score = score
                best_value = value
                best_key = key
                if score == 3:
                    return best_value, best_key
    return best_value, best_key


def _should_attempt_schema_fallback(reason: str | None) -> bool:
    if reason is None:
        return True
    lowered = reason.lower()
    blocked_terms = (
        "captcha",
        "blocked",
        "access denied",
        "forbidden",
        "robot",
        "navigate_failed",
        "click_failed",
        "type_and_submit_failed",
        "tool_execution_failed",
    )
    return not any(term in lowered for term in blocked_terms)


def _schema_fallback_decision(
    *,
    extraction_schema: dict[str, str] | None,
    markdown: str,
    fail_reason: str | None,
) -> AgentDecision | None:
    if not extraction_schema:
        return None
    if not _should_attempt_schema_fallback(fail_reason):
        return None

    rows = _extract_markdown_table_rows(markdown)
    if not rows:
        return None

    structured_data: dict[str, str | None] = {}
    evidence_parts: list[str] = []
    non_null_count = 0
    for field_name, field_description in extraction_schema.items():
        value, matched_key = _match_table_value(
            rows,
            _field_aliases(field_name, field_description),
        )
        normalized_value = _strip_markdown_artifacts(value or "")
        structured_data[field_name] = normalized_value if normalized_value else None
        if normalized_value:
            non_null_count += 1
            label = matched_key or field_name
            evidence_parts.append(f"{label}: {normalized_value}")

    if non_null_count == 0:
        return None

    evidence = "; ".join(evidence_parts)[:1000] if evidence_parts else "Extracted from page text."
    return AgentDecision(
        action="extract",
        answer=None,
        structured_data=structured_data,
        evidence=evidence,
        confidence=0.45,
        step_summary="Fallback schema extraction from page text after model failure.",
        next_step="Return extracted answer now.",
    )



def _decision_from_tool_message(message: ToolMessage) -> AgentDecision:
    if not isinstance(message.content, str):
        raise ValueError("tool_output_not_string")
    return AgentDecision.model_validate_json(message.content)


def _fail_decision_json(reason: str, step_summary: str, next_step: str) -> str:
    return AgentDecision(
        action="fail",
        reason=reason,
        step_summary=step_summary,
        next_step=next_step,
    ).model_dump_json()


def _new_trace_id() -> str:
    uuid7_factory = getattr(uuid, "uuid7", None)
    if callable(uuid7_factory):
        return str(uuid7_factory())
    return str(uuid.uuid4())


def _openrouter_invoke_kwargs(runtime: _Runtime, step: int) -> dict[str, object]:
    generation_name = f"planner.{step + 1}"
    extra_body: dict[str, object] = {
        "session_id": runtime.trace_id,
        "trace": {
            "trace_id": runtime.trace_id,
            "trace_name": "auto_browse_agent_run",
            "generation_name": generation_name,
        }
    }
    return {"extra_body": extra_body}


def _normalize_extraction_schema(
    extraction_schema: dict[str, str] | None,
) -> dict[str, str] | None:
    if extraction_schema is None:
        return None
    if not extraction_schema:
        raise ValueError("extract_schema_empty")

    normalized: dict[str, str] = {}
    for key, value in extraction_schema.items():
        if not isinstance(key, str):
            raise ValueError("extract_schema_invalid_field_name")
        if not isinstance(value, str):
            raise ValueError("extract_schema_invalid_field_description")
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_key:
            raise ValueError("extract_schema_invalid_field_name")
        if not normalized_value:
            raise ValueError("extract_schema_invalid_field_description")
        normalized[normalized_key] = normalized_value

    return normalized


def _normalize_task_value_map(
    value: Mapping[str, str] | None,
    *,
    field_name: str,
) -> dict[str, str] | None:
    if value is None:
        return None
    if not value:
        raise ValueError(f"{field_name}_empty")

    normalized: dict[str, str] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name}_invalid_key")
        if not isinstance(raw_value, str):
            raise ValueError(f"{field_name}_invalid_value")
        normalized_key = key.strip()
        normalized_value = raw_value.strip()
        if not normalized_key:
            raise ValueError(f"{field_name}_invalid_key")
        if not normalized_value:
            raise ValueError(f"{field_name}_invalid_value")
        normalized[normalized_key] = normalized_value
    return normalized


def _normalize_completion_status(status: str) -> str:
    normalized = status.strip().lower()
    allowed_statuses = {"completed", "blocked", "requires_user_input", "requires_approval"}
    if normalized not in allowed_statuses:
        raise ValueError("complete_goal_invalid_status")
    return normalized


def _analysis_summary_text(
    summary: str,
    *,
    recommended_action: str,
    recommended_interactable_ref: str | None,
    recommended_selector: str | None,
    recommended_value: str | None,
    blocker_summary: str | None,
    completion_signals: list[str],
) -> str:
    parts = [summary.strip()]
    parts.append(f"best_next_action={recommended_action.strip()}")
    if recommended_interactable_ref:
        parts.append(f"interactable_ref={recommended_interactable_ref.strip()}")
    if recommended_selector:
        parts.append(f"selector={recommended_selector.strip()}")
    if recommended_value:
        parts.append(f"value={recommended_value.strip()}")
    if blocker_summary:
        parts.append(f"blockers={blocker_summary.strip()}")
    if completion_signals:
        parts.append(f"completion_signals={'; '.join(signal.strip() for signal in completion_signals[:3])}")
    return " | ".join(part for part in parts if part)


def _verification_summary_text(
    summary: str,
    *,
    verified: bool,
    missing_requirements: list[str],
    recommended_next_action: str | None,
) -> str:
    parts = [summary.strip(), f"verified={verified}"]
    if missing_requirements:
        parts.append(
            f"missing={'; '.join(item.strip() for item in missing_requirements[:3])}"
        )
    if recommended_next_action:
        parts.append(f"next_action={recommended_next_action.strip()}")
    return " | ".join(part for part in parts if part)


def _build_tools(runtime: _Runtime):
    def _find_interactable_by_ref(interactable_ref: str | None) -> Interactable | None:
        if interactable_ref is None:
            return None
        normalized_ref = interactable_ref.strip()
        if not normalized_ref:
            return None
        page_state = runtime.current_page_state
        if page_state is None:
            return None
        return next(
            (item for item in page_state.interactables if (item.ref or "") == normalized_ref),
            None,
        )

    def _find_interactable_by_selector(selector: str | None) -> Interactable | None:
        if selector is None:
            return None
        normalized_selector = selector.strip()
        if not normalized_selector:
            return None
        page_state = runtime.current_page_state
        if page_state is None:
            return None
        return next(
            (item for item in page_state.interactables if item.selector == normalized_selector),
            None,
        )

    def _resolve_action_target(
        *,
        selector: str | None,
        interactable_ref: str | None,
        allowed_kinds: set[str] | None = None,
    ) -> tuple[str | None, Interactable | None, str | None]:
        normalized_ref = interactable_ref.strip() if interactable_ref is not None else None
        if normalized_ref == "":
            normalized_ref = None
        interactable = _find_interactable_by_ref(normalized_ref)
        if normalized_ref is not None and interactable is None:
            return None, None, "unknown_interactable_ref"
        if interactable is not None and allowed_kinds is not None and interactable.kind not in allowed_kinds:
            return None, interactable, "interactable_kind_mismatch"
        if interactable is not None:
            return interactable.selector, interactable, None

        normalized_selector = _normalize_tool_text(selector) if selector and selector.strip() else None
        if normalized_selector:
            return normalized_selector, _find_interactable_by_selector(normalized_selector), None
        return None, None, "missing_selector"

    def _resolve_navigation_url(url: str) -> str | None:
        normalized_url = _normalize_tool_text(url) if url.strip() else ""
        if not normalized_url:
            return None

        parsed = urlsplit(normalized_url)
        if parsed.scheme:
            if parsed.scheme not in {"http", "https"}:
                return None
            if not parsed.netloc:
                return None
            return normalized_url

        current_url = getattr(runtime.page, "url", "") or ""
        resolved_url = urljoin(current_url, normalized_url)
        resolved = urlsplit(resolved_url)
        if resolved.scheme not in {"http", "https"} or not resolved.netloc:
            return None
        return resolved_url

    def _resolve_navigation_target(
        *,
        url: str | None,
        interactable_ref: str | None,
    ) -> tuple[str | None, Interactable | None, str | None]:
        if url is not None and url.strip():
            resolved_url = _resolve_navigation_url(url)
            if resolved_url is None:
                return None, None, "navigate_invalid_url"
            interactable = _find_interactable_by_ref(interactable_ref.strip()) if interactable_ref else None
            return resolved_url, interactable, None

        normalized_ref = interactable_ref.strip() if interactable_ref is not None else None
        if normalized_ref == "":
            normalized_ref = None
        interactable = _find_interactable_by_ref(normalized_ref)
        if normalized_ref is not None and interactable is None:
            return None, None, "unknown_interactable_ref"
        if interactable is not None:
            if interactable.kind != "link" or not interactable.href:
                return None, interactable, "interactable_missing_href"
            resolved_url = _resolve_navigation_url(interactable.href)
            if resolved_url is None:
                return None, interactable, "navigate_invalid_url"
            return resolved_url, interactable, None

        if url is None or not url.strip():
            return None, None, "navigate_missing_target"
        resolved_url = _resolve_navigation_url(url)
        if resolved_url is None:
            return None, None, "navigate_invalid_url"
        return resolved_url, None, None

    def _target_is_present(selector: str | None, interactable_ref: str | None) -> bool:
        if interactable_ref is not None:
            return _find_interactable_by_ref(interactable_ref) is not None
        if selector is None:
            return False
        normalized_selector = selector.strip()
        if not normalized_selector:
            return False
        page_state = runtime.current_page_state
        if page_state is None:
            return True
        return any(item.selector == normalized_selector for item in page_state.interactables)

    async def _runtime_page_state() -> PageState:
        page_state = await runtime.snapshot_service.capture()
        runtime.current_page_state = page_state
        return page_state

    async def _verification_is_current() -> bool:
        current_url = getattr(runtime.page, "url", "") or ""
        return bool(
            runtime.last_verification_passed is True
            and runtime.last_verification_url
            and runtime.last_verification_url == current_url
        )

    def _decision_json(
        action: str,
        step_summary: str,
        next_step: str,
        **decision_kwargs: object,
    ) -> str:
        return AgentDecision(
            action=action,
            step_summary=step_summary,
            next_step=next_step,
            **decision_kwargs,
        ).model_dump_json()

    async def _execute_tool_action(
        *,
        action: str,
        fail_reason: str,
        step_summary: str,
        next_step: str,
        operation: Callable[[], Awaitable[None]],
        wait_dom: bool = True,
        verify_effect: bool = False,
        verify_state: Callable[[], Awaitable[bool]] | None = None,
        invalidate_verification: bool = False,
        **decision_kwargs: object,
    ) -> str:
        try:
            before_snapshot: tuple[str, str, str] | None = None
            if verify_effect and verify_state is None:
                before_snapshot = await capture_dom_signature(runtime.page)

            await operation()
            if wait_dom:
                await _wait_domcontentloaded(runtime.page, timeout=10000)

            if verify_state is not None:
                if not await verify_state():
                    raise RuntimeError("action_had_no_effect")
            elif verify_effect and before_snapshot is not None:
                after_snapshot = await capture_dom_signature(runtime.page)
                if before_snapshot == after_snapshot:
                    has_effect = await wait_for_action_effect(runtime.page, before_snapshot)
                    if not has_effect:
                        raise RuntimeError("action_had_no_effect")

            if invalidate_verification:
                runtime.last_verification_passed = None
                runtime.last_verification_url = None
                runtime.snapshot_service.invalidate()
                # Do NOT clear current_page_state here: subsequent tool calls
                # within the same multi-action step still need it to resolve
                # interactable refs via _resolve_action_target.

            return _decision_json(action, step_summary, next_step, **decision_kwargs)
        except Exception:
            return _fail_decision_json(fail_reason, step_summary, next_step)

    @tool("analyze_page", args_schema=AnalyzePageArgs)
    async def analyze_page(
        question: str,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Analyze the current page with Deep Agents and return guidance for the next step."""
        normalized_question = _normalize_tool_text(question) if question.strip() else ""
        if not normalized_question:
            return _fail_decision_json("analyze_page_missing_question", step_summary, next_step)

        try:
            page_state = await _runtime_page_state()
        except Exception as exc:
            return _fail_decision_json(
                f"analyze_page_capture_failed:{_error_detail(exc)}",
                step_summary,
                next_step,
            )

        try:
            advisor_budget = compute_context_budget(
                page_state,
                goal_type=runtime.goal_type,
                extraction_schema=runtime.extraction_schema,
            )
            async with asyncio.timeout(_DEEP_ADVISOR_TIMEOUT_SECONDS):
                analysis = await analyze_page_with_deep_agents(
                    model=runtime.openrouter_client.chat_model(),
                    page_state=page_state,
                    target_prompt=runtime.target_prompt,
                    question=normalized_question,
                    goal_type=runtime.goal_type,
                    task_data=runtime.task_data,
                    sensitive_data=runtime.sensitive_data,
                    history=runtime.current_trace,
                    advisor_markdown_chars=advisor_budget.advisor_markdown_chars,
                )
        except Exception as exc:
            analysis = fallback_page_analysis(
                page_state=page_state,
                target_prompt=runtime.target_prompt,
                question=normalized_question,
                error=exc,
            )

        return _decision_json(
            action="analyze",
            analysis=_analysis_summary_text(
                analysis.summary,
                recommended_action=analysis.recommended_action,
                recommended_interactable_ref=analysis.recommended_interactable_ref,
                recommended_selector=analysis.recommended_selector,
                recommended_value=analysis.recommended_value,
                blocker_summary=analysis.blocker_summary,
                completion_signals=analysis.completion_signals,
            ),
            result_data=analysis.model_dump(mode="json"),
            confidence=analysis.confidence,
            step_summary=step_summary,
            next_step=next_step,
        )

    @tool("verify_goal", args_schema=VerifyGoalArgs)
    async def verify_goal(
        criteria: str,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Verify the current page really satisfies the browser goal."""
        normalized_criteria = _normalize_tool_text(criteria) if criteria.strip() else ""
        if not normalized_criteria:
            return _fail_decision_json("verify_goal_missing_criteria", step_summary, next_step)

        try:
            page_state = await _runtime_page_state()
            verify_budget = compute_context_budget(
                page_state,
                goal_type=runtime.goal_type,
                extraction_schema=runtime.extraction_schema,
            )
            async with asyncio.timeout(_DEEP_ADVISOR_TIMEOUT_SECONDS):
                verification = await verify_goal_with_deep_agents(
                    model=runtime.openrouter_client.chat_model(),
                    page_state=page_state,
                    target_prompt=runtime.target_prompt,
                    criteria=normalized_criteria,
                    goal_type=runtime.goal_type,
                    task_data=runtime.task_data,
                    sensitive_data=runtime.sensitive_data,
                    history=runtime.current_trace,
                    advisor_markdown_chars=verify_budget.advisor_markdown_chars,
                )
        except Exception as exc:
            return _fail_decision_json(
                f"verify_goal_failed:{_error_detail(exc)}",
                step_summary,
                next_step,
            )

        runtime.last_verification_passed = verification.verified
        runtime.last_verification_url = getattr(runtime.page, "url", "") or page_state.url

        return _decision_json(
            action="verify",
            analysis=_verification_summary_text(
                verification.summary,
                verified=verification.verified,
                missing_requirements=verification.missing_requirements,
                recommended_next_action=verification.recommended_next_action,
            ),
            verified=verification.verified,
            result_data=verification.model_dump(mode="json"),
            evidence=verification.evidence,
            confidence=verification.confidence,
            step_summary=step_summary,
            next_step=next_step,
        )

    @tool("type_and_submit", args_schema=TypeAndSubmitArgs)
    async def type_and_submit(
        text: str,
        step_summary: str,
        next_step: str,
        selector: str | None = None,
        interactable_ref: str | None = None,
    ) -> str:
        """Type into an input and submit with Enter."""
        resolved_selector, resolved_interactable, target_error = _resolve_action_target(
            selector=selector,
            interactable_ref=interactable_ref,
            allowed_kinds={"input"},
        )
        if target_error is not None or resolved_selector is None:
            return _fail_decision_json(
                target_error or "type_and_submit_missing_selector",
                step_summary,
                next_step,
            )

        async def _operation() -> None:
            hint = extract_selector_hint(resolved_selector)
            fallback_selectors = type_fallback_selectors(resolved_selector)
            for attempt in range(2):
                await _wait_domcontentloaded(runtime.page, timeout=3000)
                if await try_type_and_submit_selector(
                    runtime.page,
                    resolved_selector,
                    text,
                    wait_timeout_ms=PRIMARY_SELECTOR_WAIT_MS,
                ):
                    return

                for fallback_selector in fallback_selectors:
                    if await try_type_and_submit_selector(
                        runtime.page,
                        fallback_selector,
                        text,
                        wait_timeout_ms=FALLBACK_SELECTOR_WAIT_MS,
                    ):
                        return

                if await type_and_submit_via_text_heuristic(runtime.page, text, hint):
                    return

                if attempt == 0:
                    await wait_short(runtime.page, 400)

            raise RuntimeError("type_and_submit_failed")

        return await _execute_tool_action(
            action="type_and_submit",
            fail_reason="type_and_submit_failed",
            interactable_ref=resolved_interactable.ref if resolved_interactable is not None else interactable_ref,
            selector=resolved_selector,
            text=text,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            verify_effect=True,
            invalidate_verification=True,
        )

    @tool("fill", args_schema=FillArgs)
    async def fill(
        text: str,
        step_summary: str,
        next_step: str,
        selector: str | None = None,
        interactable_ref: str | None = None,
    ) -> str:
        """Fill a field without submitting it."""
        resolved_selector, resolved_interactable, target_error = _resolve_action_target(
            selector=selector,
            interactable_ref=interactable_ref,
            allowed_kinds={"input"},
        )
        if target_error is not None or resolved_selector is None:
            return _fail_decision_json(target_error or "fill_missing_selector", step_summary, next_step)
        resolved_selector_state = {"value": resolved_selector}

        async def _operation() -> None:
            fallback_selectors = type_fallback_selectors(resolved_selector)
            for attempt in range(2):
                await _wait_domcontentloaded(runtime.page, timeout=3000)
                if await try_fill_selector(
                    runtime.page,
                    resolved_selector,
                    text,
                    wait_timeout_ms=PRIMARY_SELECTOR_WAIT_MS,
                ):
                    resolved_selector_state["value"] = resolved_selector
                    return

                for fallback_selector in fallback_selectors:
                    if await try_fill_selector(
                        runtime.page,
                        fallback_selector,
                        text,
                        wait_timeout_ms=FALLBACK_SELECTOR_WAIT_MS,
                    ):
                        resolved_selector_state["value"] = fallback_selector
                        return

                if attempt == 0:
                    await wait_short(runtime.page, 300)

            raise RuntimeError("fill_failed")

        return await _execute_tool_action(
            action="fill",
            fail_reason="fill_failed",
            interactable_ref=resolved_interactable.ref if resolved_interactable is not None else interactable_ref,
            selector=resolved_selector,
            text=text,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            wait_dom=False,
            verify_state=lambda: selector_value_matches(runtime.page, resolved_selector_state["value"], text),
            invalidate_verification=True,
        )

    @tool("submit", args_schema=SubmitArgs)
    async def submit(
        step_summary: str,
        next_step: str,
        selector: str | None = None,
        interactable_ref: str | None = None,
    ) -> str:
        """Submit a form or field."""
        resolved_selector, resolved_interactable, target_error = _resolve_action_target(
            selector=selector,
            interactable_ref=interactable_ref,
            allowed_kinds={"input", "button", "select", "checkbox", "radio"},
        )
        if target_error is not None or resolved_selector is None:
            return _fail_decision_json(target_error or "submit_missing_selector", step_summary, next_step)

        async def _operation() -> None:
            if await try_submit_selector(runtime.page, resolved_selector):
                return
            raise RuntimeError("submit_failed")

        return await _execute_tool_action(
            action="submit",
            fail_reason="submit_failed",
            interactable_ref=resolved_interactable.ref if resolved_interactable is not None else interactable_ref,
            selector=resolved_selector,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            verify_effect=True,
            invalidate_verification=True,
        )

    @tool("select_option", args_schema=SelectOptionArgs)
    async def select_option(
        value: str,
        step_summary: str,
        next_step: str,
        selector: str | None = None,
        interactable_ref: str | None = None,
    ) -> str:
        """Select an option from a dropdown."""
        resolved_selector, resolved_interactable, target_error = _resolve_action_target(
            selector=selector,
            interactable_ref=interactable_ref,
            allowed_kinds={"select"},
        )
        if target_error is not None or resolved_selector is None:
            return _fail_decision_json(
                target_error or "select_option_missing_selector",
                step_summary,
                next_step,
            )
        resolved_selector_state = {"value": resolved_selector}

        async def _operation() -> None:
            fallback_selectors = select_fallback_selectors(resolved_selector)
            if await try_select_option_selector(runtime.page, resolved_selector, value):
                resolved_selector_state["value"] = resolved_selector
                return
            for fallback_selector in fallback_selectors:
                if await try_select_option_selector(runtime.page, fallback_selector, value):
                    resolved_selector_state["value"] = fallback_selector
                    return
            raise RuntimeError("select_option_failed")

        return await _execute_tool_action(
            action="select_option",
            fail_reason="select_option_failed",
            interactable_ref=resolved_interactable.ref if resolved_interactable is not None else interactable_ref,
            selector=resolved_selector,
            value=value,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            wait_dom=False,
            verify_state=lambda: selector_has_selected_value(
                runtime.page,
                resolved_selector_state["value"],
                value,
            ),
            invalidate_verification=True,
        )

    @tool("check", args_schema=CheckArgs)
    async def check(
        step_summary: str,
        next_step: str,
        selector: str | None = None,
        interactable_ref: str | None = None,
    ) -> str:
        """Check a checkbox or radio option."""
        resolved_selector, resolved_interactable, target_error = _resolve_action_target(
            selector=selector,
            interactable_ref=interactable_ref,
            allowed_kinds={"checkbox", "radio"},
        )
        if target_error is not None or resolved_selector is None:
            return _fail_decision_json(target_error or "check_missing_selector", step_summary, next_step)
        resolved_selector_state = {"value": resolved_selector}

        async def _operation() -> None:
            fallback_selectors = check_fallback_selectors(resolved_selector)
            if await try_check_selector(runtime.page, resolved_selector):
                resolved_selector_state["value"] = resolved_selector
                return
            for fallback_selector in fallback_selectors:
                if await try_check_selector(runtime.page, fallback_selector):
                    resolved_selector_state["value"] = fallback_selector
                    return
            raise RuntimeError("check_failed")

        return await _execute_tool_action(
            action="check",
            fail_reason="check_failed",
            interactable_ref=resolved_interactable.ref if resolved_interactable is not None else interactable_ref,
            selector=resolved_selector,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            wait_dom=False,
            verify_state=lambda: selector_is_checked(runtime.page, resolved_selector_state["value"]),
            invalidate_verification=True,
        )

    @tool("wait_for", args_schema=WaitForArgs)
    async def wait_for(
        state: str,
        timeout_ms: int,
        step_summary: str,
        next_step: str,
        selector: str | None = None,
        interactable_ref: str | None = None,
    ) -> str:
        """Wait for a selector to reach a specific state."""
        normalized_state = state.strip().lower()
        if normalized_state not in {"attached", "visible", "hidden", "detached"}:
            return _fail_decision_json("wait_for_invalid_state", step_summary, next_step)
        resolved_selector, resolved_interactable, target_error = _resolve_action_target(
            selector=selector,
            interactable_ref=interactable_ref,
        )
        if target_error is not None or resolved_selector is None:
            return _fail_decision_json(target_error or "wait_for_missing_selector", step_summary, next_step)

        async def _operation() -> None:
            if await try_wait_for_selector(
                runtime.page,
                resolved_selector,
                state=normalized_state,
                timeout_ms=timeout_ms,
            ):
                return
            raise RuntimeError("wait_for_failed")

        return await _execute_tool_action(
            action="wait_for",
            fail_reason="wait_for_failed",
            interactable_ref=resolved_interactable.ref if resolved_interactable is not None else interactable_ref,
            selector=resolved_selector,
            wait_state=normalized_state,
            timeout_ms=timeout_ms,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            wait_dom=False,
            invalidate_verification=True,
        )

    @tool("click", args_schema=ClickArgs)
    async def click(
        step_summary: str,
        next_step: str,
        selector: str | None = None,
        interactable_ref: str | None = None,
    ) -> str:
        """Click a visible element."""
        resolved_selector, resolved_interactable, target_error = _resolve_action_target(
            selector=selector,
            interactable_ref=interactable_ref,
            allowed_kinds={"button", "link"},
        )
        if target_error is not None or resolved_selector is None or not _target_is_present(
            resolved_selector,
            resolved_interactable.ref if resolved_interactable is not None else interactable_ref,
        ):
            return _decision_json(
                action="analyze",
                analysis=(
                    "The requested click target was not present in the current snapshot. "
                    "Choose a visible interactable ref or href from the provided interactables instead of inventing one."
                ),
                result_data={
                    "unknown_selector": selector,
                    "unknown_interactable_ref": interactable_ref,
                    "target_error": target_error,
                },
                confidence=0.0,
                step_summary="The requested click target was not present in the current page snapshot.",
                next_step="Use a visible interactable ref or stable href from the current interactables.",
            )

        async def _operation() -> None:
            hint = extract_selector_hint(resolved_selector)
            fallback_selectors = click_fallback_selectors(resolved_selector)
            for attempt in range(2):
                await _wait_domcontentloaded(runtime.page, timeout=3000)
                if await try_click_selector(runtime.page, resolved_selector):
                    return

                for fallback_selector in fallback_selectors:
                    if await try_click_selector(runtime.page, fallback_selector):
                        return

                if await click_via_text_heuristic(runtime.page, hint):
                    return

                if await click_single_visible_link(runtime.page):
                    return

                if await click_via_css_fallback(runtime.page, resolved_selector):
                    return

                if attempt == 0:
                    await wait_short(runtime.page, 400)

            raise RuntimeError("click_failed")

        return await _execute_tool_action(
            action="click",
            fail_reason="click_failed",
            interactable_ref=resolved_interactable.ref if resolved_interactable is not None else interactable_ref,
            selector=resolved_selector,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            verify_effect=True,
            invalidate_verification=True,
        )

    @tool("navigate", args_schema=NavigateArgs)
    async def navigate(
        step_summary: str,
        next_step: str,
        url: str | None = None,
        interactable_ref: str | None = None,
        selector: str | None = None,
    ) -> str:
        """Navigate to a stable URL or href. Relative URLs are resolved against the current page."""
        _ = selector
        resolved_url, resolved_interactable, target_error = _resolve_navigation_target(
            url=url,
            interactable_ref=interactable_ref,
        )
        if target_error is not None or resolved_url is None:
            return _fail_decision_json(target_error or "navigate_invalid_url", step_summary, next_step)

        async def _operation() -> None:
            await goto_with_fallback(runtime.page, resolved_url, timeout_ms=15000)

        async def _navigation_reached_target() -> bool:
            current_url = getattr(runtime.page, "url", "") or ""
            if _normalized_page_identity(current_url) == _normalized_page_identity(resolved_url):
                return True
            title_fn = getattr(runtime.page, "title", None)
            current_title = ""
            if callable(title_fn):
                try:
                    current_title = _normalize_tool_text(await title_fn())
                except Exception:
                    current_title = ""
            return _url_target_overlap(current_url, current_title, resolved_url)

        return await _execute_tool_action(
            action="navigate",
            fail_reason="navigate_failed",
            interactable_ref=resolved_interactable.ref if resolved_interactable is not None else interactable_ref,
            url=resolved_url,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            wait_dom=False,
            verify_state=_navigation_reached_target,
            invalidate_verification=True,
        )

    @tool("extract_answer", args_schema=ExtractAnswerArgs)
    async def extract_answer(
        answer: str | None,
        structured_data: dict[str, str | None] | None,
        evidence: str,
        confidence: float | int | str | None,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Return a final extracted answer from current page evidence."""
        normalized_answer = _normalize_tool_text(answer) if answer and answer.strip() else None
        normalized_structured_data: dict[str, str | None] | None = None
        if structured_data is not None:
            normalized_structured_data = {
                key: (_normalize_tool_text(value) if isinstance(value, str) and value.strip() else None)
                for key, value in structured_data.items()
            }
        normalized_answer = _normalize_tool_text(answer) if answer and answer.strip() else None
        normalized_evidence = _normalize_tool_text(evidence) if evidence.strip() else evidence
        normalized_confidence = _normalize_confidence_value(confidence)
        page_state = runtime.current_page_state
        task_query = extract_task_query(runtime.target_prompt)
        destination_required = task_requires_destination_page(runtime.target_prompt)

        # Detect stuck loops: if the last 2 trace steps are both analyze on
        # the same URL, let the extraction through even if destination check
        # would normally block it.
        _stuck_in_loop = False
        if (
            page_state is not None
            and len(runtime.current_trace) >= 2
            and all(
                s.decision.action == "analyze" and s.url == page_state.url
                for s in runtime.current_trace[-2:]
            )
        ):
            _stuck_in_loop = True

        if (
            not _stuck_in_loop
            and runtime.extraction_schema is None
            and page_state is not None
            and task_query
            and destination_required
        ):
            progress = search_progress_state(page_state, runtime.target_prompt)
            on_start_page = _normalized_page_identity(page_state.url) == _normalized_page_identity(runtime.start_url)
            identity_match = page_identity_matches_query(page_state, task_query)

            if (
                _prompt_requests_title(runtime.target_prompt, runtime.extraction_schema)
                and (on_start_page or progress in {"search_entry", "results_list"})
                and not identity_match
            ):
                return _decision_json(
                    action="analyze",
                    analysis=(
                        "The destination page has not been reached yet. The agent must navigate further "
                        "before extracting. Current search stage: " + progress
                    ),
                    result_data={
                        "task_query": task_query,
                        "current_url": page_state.url,
                        "current_title": page_state.title,
                        "search_stage": progress,
                    },
                    confidence=0.0,
                    step_summary="Destination page not reached yet — keep navigating before extracting.",
                    next_step="Navigate to the target page before attempting extraction.",
                )

        if runtime.extraction_schema:
            if normalized_structured_data is None:
                return _fail_decision_json(
                    "extract_schema_missing_structured_data",
                    step_summary,
                    next_step,
                )
            expected_fields = set(runtime.extraction_schema.keys())
            actual_fields = set(normalized_structured_data.keys())
            if expected_fields != actual_fields:
                return _fail_decision_json(
                    "extract_schema_fields_mismatch",
                    step_summary,
                    next_step,
                )
        elif normalized_answer is None:
            return _fail_decision_json(
                "extract_answer_missing_answer",
                step_summary,
                next_step,
            )

        return _decision_json(
            action="extract",
            answer=normalized_answer,
            structured_data=normalized_structured_data,
            evidence=normalized_evidence,
            confidence=normalized_confidence,
            step_summary=step_summary,
            next_step=next_step,
            status="completed",
            goal_summary=normalized_answer or step_summary,
            result_data=normalized_structured_data or (
                {"answer": normalized_answer} if normalized_answer is not None else None
            ),
        )

    @tool("complete_goal", args_schema=CompleteGoalArgs)
    async def complete_goal(
        status: str,
        goal_summary: str,
        result_data: dict[str, Any] | None,
        evidence: str,
        confidence: float | int | str | None,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Return the final outcome of a generic browser goal."""
        try:
            normalized_status = _normalize_completion_status(status)
        except ValueError:
            return _fail_decision_json("complete_goal_invalid_status", step_summary, next_step)

        if (
            normalized_status == "completed"
            and runtime.goal_type != "extract"
            and not await _verification_is_current()
        ):
            return _decision_json(
                action="verify",
                analysis=(
                    "Completion was attempted without a current verification step. "
                    "Call verify_goal with explicit success criteria before complete_goal."
                ),
                verified=False,
                step_summary="Completion needs verification before finalizing the goal.",
                next_step="Call verify_goal before declaring the task complete.",
            )

        normalized_goal_summary = _normalize_tool_text(goal_summary) if goal_summary.strip() else None
        if normalized_goal_summary is None:
            return _fail_decision_json("complete_goal_missing_summary", step_summary, next_step)
        normalized_evidence = _normalize_tool_text(evidence) if evidence.strip() else evidence
        normalized_confidence = _normalize_confidence_value(confidence)

        return _decision_json(
            action="complete",
            status=normalized_status,
            goal_summary=normalized_goal_summary,
            result_data=result_data,
            evidence=normalized_evidence,
            confidence=normalized_confidence,
            step_summary=step_summary,
            next_step=next_step,
        )

    @tool("fail", args_schema=FailArgs)
    async def fail(
        reason: str,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Stop execution with a concrete failure reason."""
        return _fail_decision_json(reason, step_summary, next_step)

    return [
        analyze_page,
        verify_goal,
        type_and_submit,
        fill,
        submit,
        select_option,
        check,
        wait_for,
        click,
        navigate,
        extract_answer,
        complete_goal,
        fail,
    ]


def _build_graph(runtime: _Runtime):
    tools = _build_tools(runtime)
    tools_by_name = {tool_def.name: tool_def for tool_def in tools}
    llm = runtime.openrouter_client.chat_model()
    try:
        llm_with_tools = llm.bind_tools(tools, tool_choice="required")
    except TypeError:
        llm_with_tools = llm.bind_tools(tools)

    async def capture_node(state: AgentGraphState) -> AgentGraphState:
        if state["result"] is not None:
            return state

        with start_span(
            name=f"capture.{state['step'] + 1}",
            span_type="task",
            metadata={"trace_id": runtime.trace_id, "step": state["step"]},
        ) as capture_span:
            page_state: PageState | None = None
            last_error: Exception | None = None
            for attempt, timeout in enumerate(
                [_SNAPSHOT_CAPTURE_TIMEOUT_SECONDS, _SNAPSHOT_CAPTURE_TIMEOUT_SECONDS + 10.0]
            ):
                try:
                    async with asyncio.timeout(timeout):
                        page_state = await runtime.snapshot_service.capture(
                            force=(attempt > 0),
                        )
                    break
                except TimeoutError as exc:
                    last_error = exc
                    if attempt == 0:
                        span_log(capture_span, output={"warning": "snapshot_timeout_retry", "attempt": attempt})
                except Exception as exc:
                    last_error = exc
                    break
            if page_state is None:
                if isinstance(last_error, TimeoutError):
                    span_log(capture_span, output={"error": "snapshot_timed_out"})
                    return _set_error(state, "snapshot_timed_out")
                span_log(capture_span, output={"error": "snapshot_failed", "detail": _error_detail(last_error)})
                return _set_error(state, "snapshot_failed")

            actual_observation = await _capture_page_observation(
                runtime.page,
                fallback_url=page_state.url,
                fallback_title=page_state.title,
            )
            actual_url = actual_observation["url"]
            actual_title = actual_observation["title"]
            if actual_url and page_state.url != actual_url:
                runtime.snapshot_service.invalidate()
                try:
                    async with asyncio.timeout(_SNAPSHOT_CAPTURE_TIMEOUT_SECONDS):
                        page_state = await runtime.snapshot_service.capture(force=True)
                except Exception:
                    page_state = page_state.model_copy(
                        update={
                            "url": actual_url,
                            "title": actual_title or page_state.title,
                        }
                    )
            elif actual_title and not page_state.title.strip():
                page_state = page_state.model_copy(update={"title": actual_title})

                if actual_url and page_state.url != actual_url:
                    page_state = page_state.model_copy(
                        update={
                            "url": actual_url,
                            "title": actual_title or page_state.title,
                        }
                    )

            state["page_state"] = page_state
            runtime.current_page_state = page_state
            runtime.current_trace = state["trace"]
            span_log(
                capture_span,
                output={
                    "url": page_state.url,
                    "title": page_state.title,
                    "page_archetype": page_state.page_archetype,
                    "page_hints": page_state.page_hints,
                    "interactable_count": len(page_state.interactables),
                },
            )
            return state

    def _fallback_tool_call(name: str, args: dict[str, Any], *, step: int) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"fallback_{step}_{name}",
                    "type": "tool_call",
                }
            ],
        )

    def _latest_analysis_step(state: AgentGraphState) -> AgentStepTrace | None:
        page_state = state.get("page_state")
        current_url = page_state.url if page_state is not None else ""
        for item in reversed(state["trace"]):
            if item.decision.action != "analyze":
                continue
            if current_url and item.url != current_url:
                continue
            if isinstance(item.decision.result_data, dict):
                return item
        return None

    def _is_fallback_analysis_step(step: AgentStepTrace) -> bool:
        summary = step.decision.step_summary.strip().lower()
        return summary.startswith("planner fallback after ")

    def _reuse_recent_analysis_message(state: AgentGraphState) -> AIMessage | None:
        page_state = state.get("page_state")
        if page_state is None or not state["trace"]:
            return None

        last_step = state["trace"][-1]
        if last_step.decision.action != "analyze":
            return None
        if last_step.url != page_state.url:
            return None
        if not _is_fallback_analysis_step(last_step):
            return None

        fallback_message = _planner_fallback_message(state, reason="apply_recent_analysis")
        if fallback_message is None:
            return None
        tool_calls = fallback_message.tool_calls or []
        if not tool_calls:
            return None
        tool_name = tool_calls[0].get("name")
        if tool_name == "analyze_page":
            return None
        return fallback_message

    def _planner_fallback_message(
        state: AgentGraphState,
        *,
        reason: str,
    ) -> AIMessage | None:
        page_state = state.get("page_state")
        if page_state is None:
            return None

        latest_analysis = _latest_analysis_step(state)
        if latest_analysis is not None:
            analysis_data = latest_analysis.decision.result_data
            if isinstance(analysis_data, dict):
                recommended_action = str(analysis_data.get("recommended_action") or "").strip()
                interactable_ref = analysis_data.get("recommended_interactable_ref")
                if not isinstance(interactable_ref, str) or not interactable_ref.strip():
                    interactable_ref = None
                recommended_selector = analysis_data.get("recommended_selector")
                if not isinstance(recommended_selector, str) or not recommended_selector.strip():
                    recommended_selector = None
                recommended_value = analysis_data.get("recommended_value")
                if not isinstance(recommended_value, str) or not recommended_value.strip():
                    recommended_value = None

                base_args: dict[str, Any] = {
                    "step_summary": f"Planner fallback after {reason}.",
                    "next_step": "Continue using the strongest grounded next action.",
                }
                if interactable_ref is not None:
                    base_args["interactable_ref"] = interactable_ref
                if recommended_selector is not None:
                    base_args["selector"] = recommended_selector

                if recommended_action == "navigate" and (recommended_value or interactable_ref):
                    args = dict(base_args)
                    if recommended_value is not None:
                        args["url"] = recommended_value
                    return _fallback_tool_call("navigate", args, step=state["step"])

                if recommended_action == "click" and (interactable_ref or recommended_selector):
                    return _fallback_tool_call("click", base_args, step=state["step"])

                if recommended_action == "wait_for" and (interactable_ref or recommended_selector):
                    args = dict(base_args)
                    args["state"] = "visible"
                    args["timeout_ms"] = 1500
                    return _fallback_tool_call("wait_for", args, step=state["step"])

                if recommended_action == "type_and_submit" and (interactable_ref or recommended_selector):
                    query = extract_task_query(runtime.target_prompt)
                    if query:
                        args = dict(base_args)
                        args["text"] = query
                        return _fallback_tool_call("type_and_submit", args, step=state["step"])

        return _fallback_tool_call(
            "analyze_page",
            {
                "question": "What is the strongest next action on this page based on visible interactables, hrefs, and body content?",
                "step_summary": f"Planner fallback after {reason}: inspect the page before acting.",
                "next_step": "Use the analysis result to choose the next grounded browser action.",
            },
            step=state["step"],
        )

    def _grounded_shortcut_message(state: AgentGraphState) -> AIMessage | None:
        page_state = state.get("page_state")
        if page_state is None:
            return None

        task_query = extract_task_query(runtime.target_prompt)
        if not task_query:
            return None

        search_progress = search_progress_state(page_state, runtime.target_prompt)
        destination_required = task_requires_destination_page(runtime.target_prompt)
        query_matches_page = page_matches_query(page_state, task_query)
        analysis = fallback_page_analysis(
            page_state=page_state,
            target_prompt=runtime.target_prompt,
            question="What is the strongest next grounded action from the current page state?",
            error=None,
        )
        data = analysis.model_dump(mode="json")
        recommended_action = str(data.get("recommended_action") or "").strip()
        interactable_ref = data.get("recommended_interactable_ref")
        if not isinstance(interactable_ref, str) or not interactable_ref.strip():
            interactable_ref = None
        recommended_selector = data.get("recommended_selector")
        if not isinstance(recommended_selector, str) or not recommended_selector.strip():
            recommended_selector = None
        recommended_value = data.get("recommended_value")
        if not isinstance(recommended_value, str) or not recommended_value.strip():
            recommended_value = None

        recent_step = state["trace"][-1] if state["trace"] else None
        if (
            recent_step is not None
            and recent_step.url == page_state.url
            and recent_step.decision.action == recommended_action
        ):
            if interactable_ref and recent_step.decision.interactable_ref == interactable_ref:
                return None
            if recommended_value and recent_step.decision.url == recommended_value:
                return None

        needs_result_progress = search_progress == "results_list" or (
            destination_required
            and not query_matches_page
            and search_progress != "search_entry"
        )

        # Search entry: find the best search input and type the query
        if search_progress == "search_entry":
            search_input = next(
                (item for item in page_state.interactables if is_search_like_input(item)),
                None,
            )
            if search_input is not None and search_input.ref:
                return _fallback_tool_call(
                    "type_and_submit",
                    {
                        "text": task_query,
                        "interactable_ref": search_input.ref,
                        "selector": search_input.selector,
                        "step_summary": "Grounded shortcut: submit the visible search field with the task query.",
                        "next_step": "Inspect the resulting page after the query is submitted.",
                    },
                    step=state["step"],
                )
            if recommended_action == "type_and_submit" and (interactable_ref or recommended_selector):
                return _fallback_tool_call(
                    "type_and_submit",
                    {
                        "text": task_query,
                        "interactable_ref": interactable_ref,
                        "selector": recommended_selector,
                        "step_summary": "Grounded shortcut: submit the visible search field with the task query.",
                        "next_step": "Inspect the resulting page after the query is submitted.",
                    },
                    step=state["step"],
                )
            if recommended_action == "click" and (interactable_ref or recommended_selector):
                return _fallback_tool_call(
                    "click",
                    {
                        "interactable_ref": interactable_ref,
                        "selector": recommended_selector,
                        "step_summary": "Grounded shortcut: open the visible search control before entering the query.",
                        "next_step": "Use the search field once it is visible.",
                    },
                    step=state["step"],
                )

        # Results list: use best_result_link directly instead of advisor recommendation
        if needs_result_progress:
            best_link = best_result_link(page_state, task_query)
            if best_link is not None and best_link.href:
                nav_args: dict[str, Any] = {
                    "step_summary": "Grounded shortcut: navigate to the best matching result link.",
                    "next_step": "Inspect the destination page after navigation.",
                }
                if best_link.ref:
                    nav_args["interactable_ref"] = best_link.ref
                nav_args["url"] = best_link.href
                return _fallback_tool_call("navigate", nav_args, step=state["step"])

            if best_link is not None and best_link.ref:
                return _fallback_tool_call(
                    "click",
                    {
                        "interactable_ref": best_link.ref,
                        "selector": best_link.selector,
                        "step_summary": "Grounded shortcut: click the best matching result link.",
                        "next_step": "Inspect the destination page after the click.",
                    },
                    step=state["step"],
                )

            # Fall back to advisor recommendation
            if recommended_action in {"navigate", "click"} and (interactable_ref or recommended_selector or recommended_value):
                if recommended_action == "navigate" and (interactable_ref or recommended_value):
                    args: dict[str, Any] = {
                        "step_summary": "Grounded shortcut: follow the strongest result link from the current page.",
                        "next_step": "Inspect the destination page after navigation.",
                    }
                    if interactable_ref is not None:
                        args["interactable_ref"] = interactable_ref
                    if recommended_value is not None:
                        args["url"] = recommended_value
                    return _fallback_tool_call("navigate", args, step=state["step"])

                return _fallback_tool_call(
                    "click",
                    {
                        "interactable_ref": interactable_ref,
                        "selector": recommended_selector,
                        "step_summary": "Grounded shortcut: open the strongest result from the current page.",
                        "next_step": "Inspect the destination page after the click.",
                    },
                    step=state["step"],
                )

        return None

    async def _last_ditch_extract(state: AgentGraphState) -> AgentGraphState:
        """When max_steps is hit, try to extract an answer from the current page
        instead of returning a bare 'max_steps_exceeded' error."""
        page_state = state.get("page_state")
        if page_state is None:
            return _set_error(state, "max_steps_exceeded")

        current_url = page_state.url or ""
        current_title = page_state.title or ""

        # Schema fallback: try structured extraction from page markdown
        if runtime.extraction_schema:
            try:
                md = await page_to_markdown(
                    runtime.page, selector=runtime.extraction_selector
                )
            except Exception:
                md = page_state.markdown
            fallback = _schema_fallback_decision(
                extraction_schema=runtime.extraction_schema,
                markdown=md,
                fail_reason=None,
            )
            if fallback is not None:
                trace_entry = AgentStepTrace(
                    step=len(state["trace"]),
                    url=current_url,
                    title=current_title,
                    decision=fallback,
                )
                state["trace"] = state["trace"] + [trace_entry]
                if runtime.on_step:
                    runtime.on_step(trace_entry)
                state["result"] = AgentResult(
                    status="completed",
                    answer=fallback.answer,
                    structured_data=fallback.structured_data,
                    source_url=current_url,
                    final_url=current_url,
                    final_title=current_title,
                    evidence=fallback.evidence,
                    confidence=fallback.confidence,
                    trace=state["trace"],
                )
                return state

        # Generic fallback: use page title + first chunk of markdown as answer
        answer_parts: list[str] = []
        if current_title:
            answer_parts.append(current_title)
        if page_state.markdown:
            snippet = page_state.markdown[:500].strip()
            if snippet:
                answer_parts.append(snippet)
        if answer_parts:
            answer = "\n".join(answer_parts)
            decision = AgentDecision(
                action="extract",
                answer=answer,
                confidence=0.3,
                step_summary="Last-ditch extraction at max_steps: returning page title and content snippet.",
                next_step="Return the best available answer.",
            )
            trace_entry = AgentStepTrace(
                step=len(state["trace"]),
                url=current_url,
                title=current_title,
                decision=decision,
            )
            state["trace"] = state["trace"] + [trace_entry]
            if runtime.on_step:
                runtime.on_step(trace_entry)
            state["result"] = AgentResult(
                status="completed",
                answer=answer,
                source_url=current_url,
                final_url=current_url,
                final_title=current_title,
                evidence="Extracted from page at max_steps limit.",
                confidence=0.3,
                trace=state["trace"],
            )
            return state

        return _set_error(state, "max_steps_exceeded")

    async def llm_node(state: AgentGraphState) -> AgentGraphState:
        if state["result"] is not None or state["page_state"] is None:
            return state
        if state["step"] >= runtime.max_steps:
            return await _last_ditch_extract(state)

        with start_span(
            name=f"llm.{state['step'] + 1}",
            span_type="task",
            metadata={"trace_id": runtime.trace_id, "step": state["step"]},
        ) as llm_span:
            reused_analysis_message = _reuse_recent_analysis_message(state)
            if reused_analysis_message is not None:
                tool_calls = reused_analysis_message.tool_calls or []
                span_log(
                    llm_span,
                    output={
                        "warning": "skipping_llm_using_recent_analysis",
                        "tool_names": [str(call.get("name")) for call in tool_calls],
                        "tool_calls": _tool_call_summaries(tool_calls),
                    },
                )
                state["messages"] = [reused_analysis_message]
                state["action_observations"] = []
                return state

            grounded_shortcut = _grounded_shortcut_message(state)
            if grounded_shortcut is not None:
                tool_calls = grounded_shortcut.tool_calls or []
                span_log(
                    llm_span,
                    output={
                        "warning": "skipping_llm_using_grounded_shortcut",
                        "tool_names": [str(call.get("name")) for call in tool_calls],
                        "tool_calls": _tool_call_summaries(tool_calls),
                    },
                )
                state["messages"] = [grounded_shortcut]
                state["action_observations"] = []
                return state

            ctx_budget = compute_context_budget(
                state["page_state"],
                goal_type=runtime.goal_type,
                extraction_schema=runtime.extraction_schema,
            )
            planner_page_state = budget_page_state(
                state["page_state"],
                markdown_chars=ctx_budget.markdown_chars,
                interactable_limit=ctx_budget.interactable_limit,
            )
            base_messages = build_llm_messages(
                planner_page_state,
                runtime.target_prompt,
                history=state["trace"],
                goal_type=runtime.goal_type,
                task_data=runtime.task_data,
                sensitive_data=runtime.sensitive_data,
                extraction_schema=runtime.extraction_schema,
                extraction_selector=runtime.extraction_selector,
                max_actions_per_step=runtime.max_actions_per_step,
                budget=ctx_budget,
                working_memory=format_scratchpad(runtime.scratchpad),
            )
            try:
                async with asyncio.timeout(_LLM_CALL_TIMEOUT_SECONDS):
                    message = await llm_with_tools.ainvoke(
                        base_messages,
                        **_openrouter_invoke_kwargs(runtime, state["step"]),
                    )
            except TimeoutError:
                fallback_message = _planner_fallback_message(state, reason="llm_timed_out")
                if fallback_message is None:
                    span_log(llm_span, output={"error": "llm_timed_out"})
                    return _set_error(state, "llm_timed_out")
                message = fallback_message
                tool_calls = message.tool_calls or []
                span_log(
                    llm_span,
                    output={
                        "warning": "llm_timed_out_using_fallback_planner",
                        "tool_names": [str(call.get("name")) for call in tool_calls],
                        "tool_calls": _tool_call_summaries(tool_calls),
                    },
                )
                state["messages"] = [message]
                state["action_observations"] = []
                return state
            except Exception as exc:
                fallback_message = _planner_fallback_message(
                    state,
                    reason=f"llm_error_{exc.__class__.__name__}",
                )
                if fallback_message is None:
                    span_log(llm_span, output={"error": "llm_invoke_failed", "detail": _error_detail(exc)})
                    return _set_error(state, "llm_invoke_failed")
                message = fallback_message
                tool_calls = message.tool_calls or []
                span_log(
                    llm_span,
                    output={
                        "warning": "llm_invoke_failed_using_fallback_planner",
                        "detail": _error_detail(exc),
                        "tool_names": [str(call.get("name")) for call in tool_calls],
                        "tool_calls": _tool_call_summaries(tool_calls),
                    },
                )
                state["messages"] = [message]
                state["action_observations"] = []
                return state
            if not isinstance(message, AIMessage):
                span_log(llm_span, output={"error": "llm_response_not_ai_message"})
                return _set_error(state, "llm_response_not_ai_message")

            tool_calls = message.tool_calls or []
            if not tool_calls:
                retry_message: AIMessage | None = None
                try:
                    async with asyncio.timeout(_LLM_CALL_TIMEOUT_SECONDS):
                        retry_message = await llm_with_tools.ainvoke(
                            base_messages
                            + [
                                HumanMessage(
                                    content=(
                                        "You returned no tool call. "
                                        "Call at least one tool now. "
                                        "Do not repeat a blocked or identical previous action."
                                    )
                                )
                            ],
                            **_openrouter_invoke_kwargs(runtime, state["step"]),
                        )
                except TimeoutError:
                    fallback_message = _planner_fallback_message(state, reason="llm_retry_timed_out")
                    if fallback_message is None:
                        span_log(llm_span, output={"error": "llm_retry_timed_out"})
                        return _set_error(state, "llm_timed_out")
                    message = fallback_message
                    tool_calls = message.tool_calls or []
                except Exception as exc:
                    fallback_message = _planner_fallback_message(
                        state,
                        reason=f"llm_retry_error_{exc.__class__.__name__}",
                    )
                    if fallback_message is None:
                        span_log(
                            llm_span,
                            output={"error": "llm_retry_failed", "detail": _error_detail(exc)},
                        )
                        return _set_error(state, "llm_invoke_failed")
                    message = fallback_message
                    tool_calls = message.tool_calls or []
                if isinstance(retry_message, AIMessage):
                    message = retry_message
                    tool_calls = message.tool_calls or []

            if not tool_calls:
                fallback_message = _planner_fallback_message(state, reason="llm_returned_no_tool_call")
                if fallback_message is None:
                    span_log(llm_span, output={"error": "llm_returned_no_tool_call"})
                    return _set_error(state, "llm_returned_no_tool_call")
                message = fallback_message
                tool_calls = message.tool_calls or []
            if len(tool_calls) > runtime.max_actions_per_step:
                span_log(llm_span, output={"error": "llm_returned_multiple_tool_calls"})
                return _set_error(state, "llm_returned_multiple_tool_calls")
            if len(tool_calls) > 1 and any(
                call.get("name") in {"extract_answer", "complete_goal", "fail"} for call in tool_calls[:-1]
            ):
                span_log(llm_span, output={"error": "llm_returned_invalid_terminal_tool_order"})
                return _set_error(state, "llm_returned_invalid_terminal_tool_order")
            state["messages"] = [message]
            state["action_observations"] = []
            span_log(
                llm_span,
                output={
                    "tool_names": [str(call.get("name")) for call in tool_calls],
                    "tool_calls": _tool_call_summaries(tool_calls),
                },
            )
            return state

    async def execute_tools_node(state: AgentGraphState) -> AgentGraphState:
        if state["result"] is not None:
            return state
        if not state["messages"]:
            return _set_error(state, "missing_ai_message")

        with start_span(
            name=f"execute_tools.{state['step'] + 1}",
            span_type="task",
            metadata={"trace_id": runtime.trace_id, "step": state["step"]},
        ) as tools_span:
            ai_message = state["messages"][0]
            if not isinstance(ai_message, AIMessage):
                span_log(tools_span, output={"error": "invalid_ai_message"})
                return _set_error(state, "invalid_ai_message")

            tool_calls = ai_message.tool_calls or []
            if not tool_calls:
                span_log(tools_span, output={"error": "llm_returned_no_tool_call"})
                return _set_error(state, "llm_returned_no_tool_call")
            if len(tool_calls) > runtime.max_actions_per_step:
                span_log(tools_span, output={"error": "llm_returned_multiple_tool_calls"})
                return _set_error(state, "llm_returned_multiple_tool_calls")

            tool_messages: list[ToolMessage] = []
            observations: list[ActionObservation] = []
            fallback_page_state = state.get("page_state")
            fallback_url = fallback_page_state.url if fallback_page_state is not None else ""
            fallback_title = fallback_page_state.title if fallback_page_state is not None else ""
            executed_tools: list[str] = []
            executed_tool_calls: list[dict[str, Any]] = []

            _TERMINAL_TOOL_NAMES = frozenset({"extract_answer", "complete_goal", "fail"})

            for call_idx, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    span_log(tools_span, output={"error": "tool_call_missing_name"})
                    return _set_error(state, "tool_call_missing_name")

                tool_def = tools_by_name.get(tool_name)
                if tool_def is None:
                    span_log(tools_span, output={"error": "tool_not_found", "tool_name": tool_name})
                    return _set_error(state, "tool_not_found")

                tool_args = tool_call.get("args", {})
                if not isinstance(tool_args, dict):
                    span_log(tools_span, output={"error": "tool_call_args_not_object", "tool_name": tool_name})
                    return _set_error(state, "tool_call_args_not_object")

                try:
                    tool_output = await tool_def.ainvoke(tool_args)
                except Exception as exc:
                    span_log(
                        tools_span,
                        output={
                            "error": "tool_execution_failed",
                            "tool_name": tool_name,
                            "tool_args": _tool_arg_preview(tool_args),
                            "detail": _error_detail(exc),
                        },
                    )
                    return _set_error(state, "tool_execution_failed")

                if not isinstance(tool_output, str):
                    span_log(
                        tools_span,
                        output={
                            "error": "tool_output_not_string",
                            "tool_name": tool_name,
                            "tool_args": _tool_arg_preview(tool_args),
                        },
                    )
                    return _set_error(state, "tool_output_not_string")

                tool_call_id = tool_call.get("id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    tool_call_id = f"call_{state['step']}_{call_idx}"

                tool_messages.append(
                    ToolMessage(
                        content=tool_output,
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )
                executed_tools.append(tool_name)
                executed_tool_calls.append({"name": tool_name, "args": tool_args})

                try:
                    _ = AgentDecision.model_validate_json(tool_output)
                except Exception:
                    span_log(
                        tools_span,
                        output={
                            "error": "invalid_tool_output",
                            "tool_name": tool_name,
                            "tool_args": _tool_arg_preview(tool_args),
                        },
                    )
                    return _set_error(state, "invalid_tool_output")

                observations.append(
                    await _capture_page_observation(
                        runtime.page,
                        fallback_url=fallback_url,
                        fallback_title=fallback_title,
                    )
                )

                # Stop executing further tool calls after a terminal tool
                if tool_name in _TERMINAL_TOOL_NAMES:
                    break

            state["messages"] = tool_messages
            state["action_observations"] = observations
            span_log(
                tools_span,
                output={
                    "tool_names": executed_tools,
                    "tool_calls": _tool_call_summaries(executed_tool_calls),
                },
            )
            return state

    async def post_tool_node(state: AgentGraphState) -> AgentGraphState:
        if state["result"] is not None:
            return state
        with start_span(
            name=f"post_tool.{state['step'] + 1}",
            span_type="task",
            metadata={"trace_id": runtime.trace_id, "step": state["step"]},
        ) as post_span:
            try:
                page_state = state["page_state"]
                if page_state is None:
                    raise ValueError("missing_page_state")
                if not state["messages"]:
                    raise ValueError("missing_tool_messages")
                observations = state.get("action_observations", [])
                decisions = [
                    _decision_from_tool_message(message)
                    for message in state["messages"]
                    if isinstance(message, ToolMessage)
                ]
                if not decisions:
                    raise ValueError("missing_decisions")
                if observations and len(observations) < len(decisions):
                    raise ValueError("missing_action_observations")
            except Exception:
                span_log(post_span, output={"error": "invalid_tool_output"})
                return _set_error(state, "invalid_tool_output")

            for index, decision in enumerate(decisions):
                current_observation = (
                    observations[index]
                    if index < len(observations)
                    else await _capture_page_observation(
                        runtime.page,
                        fallback_url=page_state.url,
                        fallback_title=page_state.title,
                    )
                )
                current_url = current_observation["url"]
                current_title = current_observation["title"]

                step_trace = AgentStepTrace(
                    step=len(state["trace"]),
                    url=current_url,
                    title=current_title,
                    decision=decision,
                )
                state["trace"] = state["trace"] + [step_trace]
                runtime.scratchpad = update_scratchpad(runtime.scratchpad, step_trace)
                if runtime.on_step:
                    runtime.on_step(step_trace)

                span_log(
                    post_span,
                    output={
                        "action": decision.action,
                        "url": current_url,
                        "title": current_title,
                        "step_summary": decision.step_summary,
                        "reason": decision.reason,
                    },
                )

                if decision.action == "extract":
                    state["result"] = AgentResult(
                        status=decision.status or "completed",
                        goal_summary=decision.goal_summary or decision.answer or decision.step_summary,
                        result_data=decision.result_data
                        or decision.structured_data
                        or ({"answer": decision.answer} if decision.answer is not None else None),
                        answer=decision.answer,
                        structured_data=decision.structured_data,
                        source_url=current_url,
                        final_url=current_url,
                        final_title=current_title,
                        evidence=decision.evidence,
                        confidence=decision.confidence,
                        trace=state["trace"],
                    )
                    return state

                if decision.action == "complete":
                    state["result"] = AgentResult(
                        status=decision.status or "completed",
                        goal_summary=decision.goal_summary,
                        result_data=decision.result_data,
                        source_url=current_url,
                        final_url=current_url,
                        final_title=current_title,
                        evidence=decision.evidence,
                        confidence=decision.confidence,
                        trace=state["trace"],
                    )
                    return state

                if decision.action == "fail":
                    # Attempt schema fallback extraction from page markdown
                    fallback_markdown = page_state.markdown
                    if runtime.extraction_schema:
                        try:
                            fallback_markdown = await page_to_markdown(
                                runtime.page,
                                selector=runtime.extraction_selector,
                            )
                        except Exception:
                            fallback_markdown = page_state.markdown
                    fallback_decision = _schema_fallback_decision(
                        extraction_schema=runtime.extraction_schema,
                        markdown=fallback_markdown,
                        fail_reason=decision.reason,
                    )
                    if fallback_decision is not None:
                        fallback_trace = AgentStepTrace(
                            step=len(state["trace"]),
                            url=current_url,
                            title=current_title,
                            decision=fallback_decision,
                        )
                        state["trace"] = state["trace"] + [fallback_trace]
                        if runtime.on_step:
                            runtime.on_step(fallback_trace)
                        state["result"] = AgentResult(
                            status="completed",
                            answer=fallback_decision.answer,
                            structured_data=fallback_decision.structured_data,
                            source_url=current_url,
                            final_url=current_url,
                            final_title=current_title,
                            evidence=fallback_decision.evidence,
                            confidence=fallback_decision.confidence,
                            trace=state["trace"],
                        )
                        return state
                    return _set_error(state, decision.reason or decision.next_step)

            return _advance(state)

    def route_after_post_tool(state: AgentGraphState) -> str:
        if state["result"] is not None:
            return END
        return "capture"

    def route_after_llm(state: AgentGraphState) -> str:
        if state["result"] is not None:
            return END
        return "execute_tools"

    graph = StateGraph(AgentGraphState)
    graph.add_node("capture", capture_node)
    graph.add_node("llm", llm_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("post_tool", post_tool_node)

    graph.add_edge(START, "capture")
    graph.add_edge("capture", "llm")
    graph.add_conditional_edges("llm", route_after_llm, {"execute_tools": "execute_tools", END: END})
    graph.add_edge("execute_tools", "post_tool")
    graph.add_conditional_edges(
        "post_tool",
        route_after_post_tool,
        {"capture": "capture", END: END},
    )

    return graph.compile()


async def run_agent(
    openrouter_client: OpenRouterClient,
    start_url: str,
    target_prompt: str,
    *,
    goal_type: str | None = None,
    task_data: Mapping[str, str] | None = None,
    sensitive_data: Mapping[str, str] | None = None,
    max_steps: int = 10,
    max_actions_per_step: int = 1,
    max_runtime_seconds: int = 90,
    extraction_schema: dict[str, str] | None = None,
    extraction_selector: str | None = None,
    headless: bool = True,
    on_step: StepCallback | None = None,
    trace_id: str | None = None,
    trace_parent: str | None = None,
) -> AgentResult:
    if not 1 <= max_actions_per_step <= 3:
        raise ValueError("max_actions_per_step must be between 1 and 3")
    if max_runtime_seconds < 1:
        raise ValueError("max_runtime_seconds must be at least 1")
    normalized_goal_type = goal_type.strip() if goal_type is not None else None
    if goal_type is not None and not normalized_goal_type:
        raise ValueError("goal_type_empty")
    normalized_task_data = _normalize_task_value_map(task_data, field_name="task_data")
    normalized_sensitive_data = _normalize_task_value_map(
        sensitive_data,
        field_name="sensitive_data",
    )
    normalized_extraction_schema = _normalize_extraction_schema(extraction_schema)
    normalized_extraction_selector = (
        extraction_selector.strip() if extraction_selector is not None else None
    )
    if extraction_selector is not None and not normalized_extraction_selector:
        raise ValueError("extraction_selector_empty")

    resolved_trace_id = trace_id or _new_trace_id()
    resolved_trace_parent = trace_parent or export_current_span_parent()
    span_input = {
        "start_url": start_url,
        "target_prompt": target_prompt,
        "goal_type": normalized_goal_type,
        "task_data": normalized_task_data,
        "has_sensitive_data": bool(normalized_sensitive_data),
        "max_steps": max_steps,
        "max_actions_per_step": max_actions_per_step,
        "max_runtime_seconds": max_runtime_seconds,
        "headless": headless,
    }
    pw = None
    browser = None
    page = None

    with start_span(
        "agent.run",
        span_type="task",
        parent=resolved_trace_parent,
        metadata={
            "trace_id": resolved_trace_id,
            "goal_type": normalized_goal_type,
            "max_steps": max_steps,
            "max_actions_per_step": max_actions_per_step,
            "max_runtime_seconds": max_runtime_seconds,
        },
    ) as run_span:
        span_log(run_span, input=span_input)
        try:
            with start_span(
                "startup.browser",
                span_type="task",
                metadata={"trace_id": resolved_trace_id},
            ) as startup_span:
                try:
                    pw, browser, page = await run_browser(start_url, headless=headless)
                except Exception as exc:
                    span_log(startup_span, output={"error": "browser_startup_failed", "detail": _error_detail(exc)})
                    return AgentResult(status="failed", error="browser_startup_failed", trace=[])
            runtime = _Runtime(
                openrouter_client=openrouter_client,
                page=page,
                snapshot_service=PageSnapshotService(
                    page=page,
                    extraction_selector=normalized_extraction_selector,
                    capture_state_fn=capture_state,
                    markdown_fn=page_to_markdown,
                ),
                start_url=start_url,
                target_prompt=target_prompt,
                goal_type=normalized_goal_type,
                task_data=normalized_task_data,
                sensitive_data=normalized_sensitive_data,
                max_steps=max_steps,
                max_actions_per_step=max_actions_per_step,
                extraction_schema=normalized_extraction_schema,
                extraction_selector=normalized_extraction_selector,
                on_step=on_step,
                trace_id=resolved_trace_id,
            )
            graph = _build_graph(runtime)

            initial_state: AgentGraphState = AgentGraphState(
                step=0,
                trace=[],
                page_state=None,
                result=None,
                messages=[],
                action_observations=[],
            )

            async with asyncio.timeout(max_runtime_seconds):
                final_state = await graph.ainvoke(initial_state)
            result = final_state.get("result")
            if result is None:
                result = AgentResult(
                    status="failed",
                    error="graph_finished_without_result",
                    trace=final_state.get("trace", []),
                )
            span_log(
                run_span,
                output={
                    "status": result.status,
                    "error": result.error,
                    "trace_steps": len(result.trace),
                    "goal_summary": result.goal_summary,
                    "answer": result.answer,
                    "source_url": result.source_url,
                    "final_url": result.final_url,
                    "final_title": result.final_title,
                },
            )
            return result
        except asyncio.TimeoutError:
            span_log(
                run_span,
                error="run_timed_out",
                output={
                    "trace_id": resolved_trace_id,
                    "final_url": getattr(page, "url", start_url),
                },
            )
            raise
        except Exception as exc:
            span_log(
                run_span,
                error=f"run_exception:{type(exc).__name__}",
                metadata={"trace_id": resolved_trace_id},
            )
            raise
        finally:
            if browser is not None:
                await browser.close()
            if pw is not None:
                await pw.stop()
            flush()
