from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, Field

from agent.browser import capture_state, run_browser
from agent.extract import page_to_markdown
from agent.models import AgentDecision, AgentResult, AgentStepTrace, PageState
from agent.observability import span_log, start_span
from agent.openrouter_client import OpenRouterClient
from agent.planner import build_llm_messages

StepCallback = Callable[[AgentStepTrace], None]


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _StepArgs(_StrictArgs):
    step_summary: str
    next_step: str


class _SelectorStepArgs(_StepArgs):
    selector: str


class TypeAndSubmitArgs(_SelectorStepArgs):
    text: str


class ClickArgs(_SelectorStepArgs):
    pass


class NavigateArgs(_StepArgs):
    url: str


class ExtractAnswerArgs(_StepArgs):
    answer: str | None = None
    structured_data: dict[str, str | None] | None = None
    evidence: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class FailArgs(_StepArgs):
    reason: str


@dataclass
class _Runtime:
    openrouter_client: OpenRouterClient
    page: Page
    target_prompt: str
    max_steps: int
    max_actions_per_step: int
    extraction_schema: dict[str, str] | None
    extraction_selector: str | None
    on_step: StepCallback | None
    trace_id: str


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
    state["result"] = AgentResult(error=error, trace=state["trace"])
    return state


def _advance(state: AgentGraphState) -> AgentGraphState:
    state["step"] += 1
    state["page_state"] = None
    state["messages"] = []
    state["action_observations"] = []
    return state


async def _wait_domcontentloaded(page: Page, timeout: int = 10000) -> None:
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except Exception:
        pass


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


async def _capture_action_snapshot(page: Page) -> tuple[str, str, str]:
    url = page.url or ""

    title = ""
    title_fn = getattr(page, "title", None)
    if callable(title_fn):
        try:
            title = await title_fn()
        except Exception:
            title = ""

    dom_signature = ""
    evaluate_fn = getattr(page, "evaluate", None)
    if callable(evaluate_fn):
        try:
            signature = await evaluate_fn(
                """() => {
                    const body = document.body;
                    if (!body) return "";
                    const text = (body.innerText || body.textContent || "")
                      .replace(/\\s+/g, " ")
                      .trim()
                      .slice(0, 1200);
                    const count = document.querySelectorAll("a,button,input,textarea,select,form").length;
                    return `${count}|${text}`;
                }"""
            )
            dom_signature = str(signature or "")
        except Exception:
            dom_signature = ""

    return (url, title, dom_signature)


def _snapshot_changed(before: tuple[str, str, str], after: tuple[str, str, str]) -> bool:
    return before != after


async def _wait_for_action_effect(
    page: Page,
    before_snapshot: tuple[str, str, str],
    *,
    retries: int = 3,
    delay_ms: int = 300,
) -> bool:
    wait_timeout_fn = getattr(page, "wait_for_timeout", None)
    if not callable(wait_timeout_fn):
        return False

    for _ in range(retries):
        try:
            await wait_timeout_fn(delay_ms)
        except Exception:
            return False

        after_snapshot = await _capture_action_snapshot(page)
        if _snapshot_changed(before_snapshot, after_snapshot):
            return True

    return False


def _strip_markdown_artifacts(value: str) -> str:
    text = value
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2007", " ")
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("* "):
        text = text[2:].strip()
    return text


def _normalize_tool_text(value: str) -> str:
    text = value.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2007", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate_for_span_log(value: str, *, limit: int = 1500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated>"


def _result_for_root_span_log(result: AgentResult) -> dict[str, Any]:
    return {
        "answer": _truncate_for_span_log(result.answer) if result.answer else None,
        "structured_data": result.structured_data,
        "source_url": result.source_url,
        "evidence": _truncate_for_span_log(result.evidence) if result.evidence else None,
        "confidence": result.confidence,
        "error": result.error,
        "trace_steps": len(result.trace),
    }


def _serialize_llm_message(message: Any) -> dict[str, Any]:
    if isinstance(message, BaseMessage):
        return message.model_dump()
    return {"type": type(message).__name__, "value": str(message)}


def _serialize_llm_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [_serialize_llm_message(message) for message in messages]


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

    source = f"{base} {field_description.lower()}"
    if "director" in source:
        aliases.update({"director", "directed by"})
    if "producer" in source:
        aliases.update({"producer", "produced by"})
    if "release" in source and "date" in source:
        aliases.update({"release date", "released", "release"})
    if "designer" in source:
        aliases.update({"designer", "designed by"})
    if "first appeared" in source:
        aliases.add("first appeared")

    cleaned_aliases = [
        alias.strip()
        for alias in aliases
        if alias.strip()
    ]
    cleaned_aliases.sort(key=len, reverse=True)
    return cleaned_aliases


def _match_table_value(rows: list[tuple[str, str]], aliases: list[str]) -> tuple[str | None, str | None]:
    best_score = 0
    best_value: str | None = None
    best_key: str | None = None
    for alias in aliases:
        alias_lower = alias.lower()
        for key, value in rows:
            key_lower = key.lower()
            score = 0
            if key_lower == alias_lower:
                score = 3
            elif key_lower.startswith(f"{alias_lower} "):
                score = 2
            elif alias_lower in key_lower:
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


def _normalize_hint_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _unquote_selector_value(raw: str) -> str:
    stripped = raw.strip()
    if len(stripped) >= 2 and (
        (stripped.startswith('"') and stripped.endswith('"'))
        or (stripped.startswith("'") and stripped.endswith("'"))
    ):
        stripped = stripped[1:-1]
    return stripped.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")


def _extract_selector_hint(selector: str) -> str | None:
    normalized_selector = selector.strip()
    if not normalized_selector:
        return None

    if normalized_selector.startswith("text="):
        return _normalize_hint_text(_unquote_selector_value(normalized_selector[len("text=") :]))

    role_name_match = re.search(
        r"""name\s*=\s*(?:"([^"]+)"|'([^']+)')""",
        normalized_selector,
    )
    if role_name_match:
        return _normalize_hint_text(role_name_match.group(1) or role_name_match.group(2))

    for attr in ("aria-label", "placeholder", "name", "id", "title", "alt", "value"):
        attr_match = re.search(
            rf"""\[{re.escape(attr)}\s*=\s*(?:"([^"]+)"|'([^']+)')\]""",
            normalized_selector,
        )
        if attr_match:
            return _normalize_hint_text(attr_match.group(1) or attr_match.group(2))

    return None


def _escape_selector_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _parse_css_selector(selector: str) -> tuple[str, int | None] | None:
    if not selector.startswith("css="):
        return None
    css_selector = selector[len("css=") :].strip()
    if not css_selector:
        return None

    base_selector = css_selector
    nth_index: int | None = None
    if " >> nth=" in css_selector:
        base_selector, nth_part = css_selector.rsplit(" >> nth=", 1)
        base_selector = base_selector.strip()
        if not base_selector:
            return None
        try:
            nth_index = int(nth_part.strip())
        except ValueError:
            return None
        if nth_index < 0:
            return None

    return (base_selector, nth_index)


def _dedupe_selectors(candidates: list[str], *, exclude: str | None = None) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized:
            continue
        if exclude is not None and normalized == exclude:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _click_fallback_selectors(selector: str) -> list[str]:
    fallback_selectors: list[str] = []
    parsed_css_selector = _parse_css_selector(selector)
    if parsed_css_selector is not None:
        base_selector, _ = parsed_css_selector
        fallback_selectors.append(f"css={base_selector}")

    hint = _extract_selector_hint(selector)
    if hint:
        escaped_hint = _escape_selector_value(hint)
        fallback_selectors.extend(
            [
                f'role=link[name="{escaped_hint}"]',
                f'role=button[name="{escaped_hint}"]',
                f'text="{escaped_hint}"',
            ]
        )

    return _dedupe_selectors(fallback_selectors, exclude=selector)


def _type_fallback_selectors(selector: str) -> list[str]:
    fallback_selectors: list[str] = []
    hint = _extract_selector_hint(selector)
    if hint:
        escaped_hint = _escape_selector_value(hint)
        fallback_selectors.extend(
            [
                f'role=textbox[name="{escaped_hint}"]',
                f'css=input[aria-label="{escaped_hint}"]',
                f'css=input[placeholder="{escaped_hint}"]',
                f'css=textarea[aria-label="{escaped_hint}"]',
                f'css=input[name="{escaped_hint}"]',
            ]
        )

    fallback_selectors.extend(
        [
            "css=input[type='search']",
            "css=input[type='text']",
            "css=textarea",
        ]
    )

    parsed_css_selector = _parse_css_selector(selector)
    if parsed_css_selector is not None:
        base_selector, _ = parsed_css_selector
        fallback_selectors.append(f"css={base_selector}")

    return _dedupe_selectors(fallback_selectors, exclude=selector)


async def _wait_short(page: Page, timeout_ms: int) -> None:
    wait_fn = getattr(page, "wait_for_timeout", None)
    if not callable(wait_fn):
        return
    try:
        await wait_fn(timeout_ms)
    except Exception:
        pass


async def _wait_for_selector_visible(page: Page, selector: str, *, timeout_ms: int = 3500) -> None:
    wait_for_selector_fn = getattr(page, "wait_for_selector", None)
    if not callable(wait_for_selector_fn):
        return
    try:
        await wait_for_selector_fn(selector, state="visible", timeout=timeout_ms)
    except TypeError:
        try:
            await wait_for_selector_fn(selector, timeout=timeout_ms)
        except Exception:
            pass
    except Exception:
        pass


async def _try_click_selector(page: Page, selector: str) -> bool:
    await _wait_for_selector_visible(page, selector)

    click_fn = getattr(page, "click", None)
    if not callable(click_fn):
        return False
    try:
        await click_fn(selector, timeout=5000)
        return True
    except TypeError:
        try:
            await click_fn(selector)
            return True
        except Exception:
            return False
    except Exception:
        return False


async def _try_type_and_submit_selector(page: Page, selector: str, text: str) -> bool:
    await _wait_for_selector_visible(page, selector)

    focus_fn = getattr(page, "focus", None)
    if callable(focus_fn):
        try:
            await focus_fn(selector)
        except Exception:
            pass

    fill_fn = getattr(page, "fill", None)
    press_fn = getattr(page, "press", None)
    if not callable(fill_fn) or not callable(press_fn):
        return False

    try:
        await fill_fn(selector, text)
        await press_fn(selector, "Enter")
        return True
    except Exception:
        return False


async def _click_via_css_fallback(page: Page, selector: str) -> bool:
    parsed_css_selector = _parse_css_selector(selector)
    if parsed_css_selector is None:
        return False
    base_selector, nth_index = parsed_css_selector

    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return False

    try:
        clicked = await evaluate_fn(
            """(payload) => {
                const { baseSelector, nthIndex } = payload;
                let el = null;
                if (typeof nthIndex === "number" && Number.isInteger(nthIndex) && nthIndex >= 0) {
                    const nodes = document.querySelectorAll(baseSelector);
                    el = nodes.length > nthIndex ? nodes[nthIndex] : null;
                } else {
                    el = document.querySelector(baseSelector);
                }
                if (!el) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                const isDisabled = Boolean(el.disabled);
                const isVisible =
                  style.display !== "none" &&
                  style.visibility !== "hidden" &&
                  rect.width > 0 &&
                  rect.height > 0 &&
                  !isDisabled;
                if (!isVisible) return false;
                el.click();
                return true;
            }""",
            {"baseSelector": base_selector, "nthIndex": nth_index},
        )
    except Exception:
        return False

    if clicked is True:
        await _wait_short(page, 300)
        return True
    return False


async def _click_via_text_heuristic(page: Page, hint: str | None) -> bool:
    if not hint:
        return False
    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return False

    try:
        clicked = await evaluate_fn(
            """(payload) => {
                const hint = String(payload?.hint || "").trim().toLowerCase();
                if (!hint) return false;
                const normalize = (value) => String(value || "").replace(/\\s+/g, " ").trim().toLowerCase();
                const isVisible = (el) => {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    const disabled = Boolean(el.disabled);
                    return (
                        style.display !== "none" &&
                        style.visibility !== "hidden" &&
                        rect.width > 0 &&
                        rect.height > 0 &&
                        !disabled
                    );
                };
                const candidates = Array.from(
                    document.querySelectorAll(
                        "a[href],button,input[type='submit'],input[type='button'],[role='button'],[role='link']"
                    )
                );
                let best = null;
                let bestScore = 0;
                for (const el of candidates) {
                    if (!isVisible(el)) continue;
                    const label = normalize(
                        el.getAttribute("aria-label") ||
                        el.innerText ||
                        el.textContent ||
                        el.value ||
                        el.getAttribute("title")
                    );
                    if (!label) continue;
                    let score = 0;
                    if (label === hint) score = 3;
                    else if (label.includes(hint)) score = 2;
                    else if (hint.includes(label) && label.length >= 4) score = 1;
                    if (score > bestScore) {
                        best = el;
                        bestScore = score;
                        if (score === 3) break;
                    }
                }
                if (!best) return false;
                best.click();
                return true;
            }""",
            {"hint": hint},
        )
    except Exception:
        return False

    if clicked is True:
        await _wait_short(page, 300)
        return True
    return False


async def _click_single_visible_link(page: Page) -> bool:
    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return False

    try:
        clicked = await evaluate_fn(
            """() => {
                // singleVisibleLinkFallback
                const isVisible = (el) => {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return (
                        style.display !== "none" &&
                        style.visibility !== "hidden" &&
                        rect.width > 0 &&
                        rect.height > 0
                    );
                };
                const links = Array.from(document.querySelectorAll("a[href]")).filter(isVisible);
                if (links.length !== 1) return false;
                links[0].click();
                return true;
            }"""
        )
    except Exception:
        return False

    if clicked is True:
        await _wait_short(page, 300)
        return True
    return False


async def _type_and_submit_via_text_heuristic(page: Page, text: str, hint: str | None) -> bool:
    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return False

    try:
        submitted = await evaluate_fn(
            """(payload) => {
                const text = String(payload?.text || "");
                if (!text) return false;
                const hint = String(payload?.hint || "").trim().toLowerCase();
                const normalize = (value) => String(value || "").replace(/\\s+/g, " ").trim().toLowerCase();
                const editableInput = (el) => {
                    if (!el) return false;
                    const tag = String(el.tagName || "").toLowerCase();
                    if (tag === "textarea") return true;
                    if (tag !== "input") return false;
                    const t = String(el.getAttribute("type") || "text").toLowerCase();
                    return ["", "text", "search", "email", "url", "tel", "password"].includes(t);
                };
                const isVisibleEnabled = (el) => {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return (
                        style.display !== "none" &&
                        style.visibility !== "hidden" &&
                        rect.width > 0 &&
                        rect.height > 0 &&
                        !el.disabled
                    );
                };
                const associatedLabel = (el) => {
                    if (!el) return "";
                    const direct = el.getAttribute("aria-label") || el.getAttribute("placeholder");
                    if (direct) return direct;
                    const id = el.getAttribute("id");
                    if (id) {
                        const label = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                        if (label && label.textContent) return label.textContent;
                    }
                    const wrapped = el.closest("label");
                    if (wrapped && wrapped.textContent) return wrapped.textContent;
                    return el.getAttribute("name") || el.getAttribute("id") || "";
                };

                const candidates = Array.from(document.querySelectorAll("input,textarea"));
                let best = null;
                let bestScore = -1;
                for (const el of candidates) {
                    if (!editableInput(el) || !isVisibleEnabled(el)) continue;
                    const label = normalize(associatedLabel(el));
                    let score = 0;
                    if (hint) {
                        if (label === hint) score = 4;
                        else if (label.includes(hint)) score = 3;
                        else if (hint.includes(label) && label.length >= 3) score = 2;
                    } else {
                        score = 1;
                    }
                    if (score > bestScore) {
                        best = el;
                        bestScore = score;
                        if (score === 4) break;
                    }
                }
                if (!best) return false;

                best.focus();
                best.value = text;
                best.dispatchEvent(new Event("input", { bubbles: true }));
                best.dispatchEvent(new Event("change", { bubbles: true }));

                if (best.form && typeof best.form.requestSubmit === "function") {
                    best.form.requestSubmit();
                    return true;
                }
                if (best.form) {
                    best.form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
                }
                best.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", bubbles: true }));
                best.dispatchEvent(new KeyboardEvent("keypress", { key: "Enter", code: "Enter", bubbles: true }));
                best.dispatchEvent(new KeyboardEvent("keyup", { key: "Enter", code: "Enter", bubbles: true }));
                return true;
            }""",
            {"text": text, "hint": hint},
        )
    except Exception:
        return False

    if submitted:
        await _wait_short(page, 300)
    return bool(submitted)


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


def _build_tools(runtime: _Runtime):
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
        **decision_kwargs: object,
    ) -> str:
        try:
            before_snapshot: tuple[str, str, str] | None = None
            if verify_effect:
                before_snapshot = await _capture_action_snapshot(runtime.page)

            await operation()
            if wait_dom:
                await _wait_domcontentloaded(runtime.page, timeout=10000)

            if verify_effect and before_snapshot is not None:
                after_snapshot = await _capture_action_snapshot(runtime.page)
                if not _snapshot_changed(before_snapshot, after_snapshot):
                    has_effect = await _wait_for_action_effect(runtime.page, before_snapshot)
                    if not has_effect:
                        raise RuntimeError("action_had_no_effect")

            return _decision_json(action, step_summary, next_step, **decision_kwargs)
        except Exception:
            return _fail_decision_json(fail_reason, step_summary, next_step)

    @tool("type_and_submit", args_schema=TypeAndSubmitArgs)
    async def type_and_submit(
        selector: str,
        text: str,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Type into an input and submit with Enter."""

        async def _operation() -> None:
            hint = _extract_selector_hint(selector)
            fallback_selectors = _type_fallback_selectors(selector)
            for attempt in range(2):
                await _wait_domcontentloaded(runtime.page, timeout=3000)
                if await _try_type_and_submit_selector(runtime.page, selector, text):
                    return

                for fallback_selector in fallback_selectors:
                    if await _try_type_and_submit_selector(runtime.page, fallback_selector, text):
                        return

                if await _type_and_submit_via_text_heuristic(runtime.page, text, hint):
                    return

                if attempt == 0:
                    await _wait_short(runtime.page, 400)

            raise RuntimeError("type_and_submit_failed")

        return await _execute_tool_action(
            action="type_and_submit",
            fail_reason="type_and_submit_failed",
            selector=selector,
            text=text,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            verify_effect=True,
        )

    @tool("click", args_schema=ClickArgs)
    async def click(
        selector: str,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Click a visible element."""

        async def _operation() -> None:
            hint = _extract_selector_hint(selector)
            fallback_selectors = _click_fallback_selectors(selector)
            for attempt in range(2):
                await _wait_domcontentloaded(runtime.page, timeout=3000)
                if await _try_click_selector(runtime.page, selector):
                    return

                for fallback_selector in fallback_selectors:
                    if await _try_click_selector(runtime.page, fallback_selector):
                        return

                if await _click_via_text_heuristic(runtime.page, hint):
                    return

                if await _click_single_visible_link(runtime.page):
                    return

                if await _click_via_css_fallback(runtime.page, selector):
                    return

                if attempt == 0:
                    await _wait_short(runtime.page, 400)

            raise RuntimeError("click_failed")

        return await _execute_tool_action(
            action="click",
            fail_reason="click_failed",
            selector=selector,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            verify_effect=True,
        )

    @tool("navigate", args_schema=NavigateArgs)
    async def navigate(
        url: str,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Navigate to an absolute URL."""
        if not url.startswith(("http://", "https://")):
            return _fail_decision_json("navigate_requires_absolute_url", step_summary, next_step)

        async def _operation() -> None:
            await runtime.page.goto(url, wait_until="domcontentloaded", timeout=15000)

        return await _execute_tool_action(
            action="navigate",
            fail_reason="navigate_failed",
            url=url,
            step_summary=step_summary,
            next_step=next_step,
            operation=_operation,
            wait_dom=False,
            verify_effect=True,
        )

    @tool("extract_answer", args_schema=ExtractAnswerArgs)
    async def extract_answer(
        answer: str | None,
        structured_data: dict[str, str | None] | None,
        evidence: str,
        confidence: float | None,
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
        normalized_evidence = _normalize_tool_text(evidence) if evidence.strip() else evidence

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
            confidence=confidence,
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

    return [type_and_submit, click, navigate, extract_answer, fail]


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
            metadata={"run_id": runtime.trace_id, "step": state["step"]},
        ) as capture_span:
            page_state = await capture_state(runtime.page)
            page_state.markdown = await page_to_markdown(
                runtime.page,
                selector=runtime.extraction_selector,
            )
            span_log(
                capture_span,
                output={
                    "url": page_state.url,
                    "title": page_state.title,
                },
            )

        state["page_state"] = page_state
        return state

    async def llm_node(state: AgentGraphState) -> AgentGraphState:
        if state["result"] is not None or state["page_state"] is None:
            return state
        if state["step"] >= runtime.max_steps:
            return _set_error(state, "max_steps_exceeded")

        base_messages = build_llm_messages(
            state["page_state"],
            runtime.target_prompt,
            history=state["trace"],
            extraction_schema=runtime.extraction_schema,
            extraction_selector=runtime.extraction_selector,
            max_actions_per_step=runtime.max_actions_per_step,
        )
        with start_span(
            name=f"planner.{state['step'] + 1}",
            span_type="llm",
            metadata={"run_id": runtime.trace_id, "step": state["step"]},
            input={
                "target_prompt": runtime.target_prompt,
                "url": state["page_state"].url,
                "history_steps": len(state["trace"]),
                "max_actions_per_step": runtime.max_actions_per_step,
            },
        ) as llm_span:
            llm_calls: list[dict[str, Any]] = []
            message = await llm_with_tools.ainvoke(
                base_messages,
                **_openrouter_invoke_kwargs(runtime, state["step"]),
            )
            llm_calls.append(
                {
                    "request_messages": _serialize_llm_messages(base_messages),
                    "response": _serialize_llm_message(message),
                }
            )
            if not isinstance(message, AIMessage):
                span_log(
                    llm_span,
                    output={"error": "llm_response_not_ai_message", "llm_calls": llm_calls},
                )
                return _set_error(state, "llm_response_not_ai_message")

            tool_calls = message.tool_calls or []
            retried_for_tool_call = False
            if not tool_calls:
                retried_for_tool_call = True
                retry_messages = base_messages + [
                    HumanMessage(
                        content=(
                            "You returned no tool call. "
                            "Call at least one tool now. "
                            "Do not repeat a blocked or identical previous action."
                        )
                    )
                ]
                retry_message = await llm_with_tools.ainvoke(
                    retry_messages,
                    **_openrouter_invoke_kwargs(runtime, state["step"]),
                )
                llm_calls.append(
                    {
                        "request_messages": _serialize_llm_messages(retry_messages),
                        "response": _serialize_llm_message(retry_message),
                    }
                )
                if isinstance(retry_message, AIMessage):
                    message = retry_message
                    tool_calls = message.tool_calls or []

            if not tool_calls:
                span_log(
                    llm_span,
                    output={"error": "llm_returned_no_tool_call", "llm_calls": llm_calls},
                )
                return _set_error(state, "llm_returned_no_tool_call")
            if len(tool_calls) > runtime.max_actions_per_step:
                if runtime.max_actions_per_step == 1:
                    span_log(
                        llm_span,
                        output={
                            "error": "llm_returned_multiple_tool_calls",
                            "llm_calls": llm_calls,
                        },
                    )
                    return _set_error(state, "llm_returned_multiple_tool_calls")
                span_log(
                    llm_span,
                    output={
                        "error": "llm_returned_too_many_tool_calls",
                        "llm_calls": llm_calls,
                    },
                )
                return _set_error(state, "llm_returned_too_many_tool_calls")
            if len(tool_calls) > 1 and any(
                call.get("name") in {"extract_answer", "fail"} for call in tool_calls[:-1]
            ):
                span_log(
                    llm_span,
                    output={
                        "error": "llm_returned_invalid_terminal_tool_order",
                        "llm_calls": llm_calls,
                    },
                )
                return _set_error(state, "llm_returned_invalid_terminal_tool_order")

            span_log(
                llm_span,
                output={
                    "tool_names": [str(call.get("name")) for call in tool_calls],
                    "retried_for_tool_call": retried_for_tool_call,
                    "llm_calls": llm_calls,
                },
            )
            state["messages"] = [message]
            state["action_observations"] = []
            return state

    async def execute_tools_node(state: AgentGraphState) -> AgentGraphState:
        if state["result"] is not None:
            return state
        if not state["messages"]:
            return _set_error(state, "missing_ai_message")

        with start_span(
            name=f"execute_tools.{state['step'] + 1}",
            span_type="task",
            metadata={"run_id": runtime.trace_id, "step": state["step"]},
        ) as tools_span:
            ai_message = state["messages"][0]
            if not isinstance(ai_message, AIMessage):
                span_log(tools_span, output={"error": "invalid_ai_message"})
                return _set_error(state, "invalid_ai_message")

            tool_calls = ai_message.tool_calls or []
            if not tool_calls:
                span_log(tools_span, output={"error": "llm_returned_no_tool_call"})
                return _set_error(state, "llm_returned_no_tool_call")

            tool_messages: list[ToolMessage] = []
            observations: list[ActionObservation] = []
            fallback_page_state = state.get("page_state")
            fallback_url = fallback_page_state.url if fallback_page_state is not None else ""
            fallback_title = fallback_page_state.title if fallback_page_state is not None else ""
            for i, tool_call in enumerate(tool_calls):
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

                with start_span(
                    name=f"tool.{tool_name}",
                    span_type="tool",
                    metadata={
                        "run_id": runtime.trace_id,
                        "step": state["step"],
                        "tool_index": i,
                    },
                    input={"args": tool_args},
                ) as tool_span:
                    try:
                        tool_output = await tool_def.ainvoke(tool_args)
                    except Exception as exc:
                        span_log(tool_span, error=str(exc))
                        span_log(tools_span, output={"error": "tool_execution_failed", "tool_name": tool_name})
                        return _set_error(state, "tool_execution_failed")

                    if not isinstance(tool_output, str):
                        span_log(tool_span, output={"error": "tool_output_not_string"})
                        span_log(tools_span, output={"error": "tool_output_not_string", "tool_name": tool_name})
                        return _set_error(state, "tool_output_not_string")

                    span_log(tool_span, output={"decision_json": _truncate_for_span_log(tool_output)})

                tool_call_id = tool_call.get("id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    tool_call_id = f"call_{state['step']}_{i}"

                tool_messages.append(
                    ToolMessage(
                        content=tool_output,
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )

                try:
                    decision = AgentDecision.model_validate_json(tool_output)
                except Exception:
                    span_log(tools_span, output={"error": "invalid_tool_output", "tool_name": tool_name})
                    return _set_error(state, "invalid_tool_output")

                observations.append(
                    await _capture_page_observation(
                        runtime.page,
                        fallback_url=fallback_url,
                        fallback_title=fallback_title,
                    )
                )
                span_log(tools_span, metadata={"last_action": decision.action, "tool_name": tool_name})

                if decision.action in {"extract", "fail"}:
                    break
                if decision.action in {"type_and_submit", "click", "navigate"} and i < (len(tool_calls) - 1):
                    # Replan after first state-changing action so subsequent decisions
                    # can use fresh page state instead of stale pre-action context.
                    break

            state["messages"] = tool_messages
            state["action_observations"] = observations
            span_log(tools_span, output={"tool_calls_executed": len(tool_messages)})
            return state

    async def post_tool_node(state: AgentGraphState) -> AgentGraphState:
        if state["result"] is not None:
            return state
        with start_span(
            name=f"post_tool.{state['step'] + 1}",
            span_type="task",
            metadata={"run_id": runtime.trace_id, "step": state["step"]},
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
                span_log(post_span, metadata={"decision_index": index, "action": decision.action})
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
                if runtime.on_step:
                    runtime.on_step(step_trace)

                if decision.action == "extract":
                    state["result"] = AgentResult(
                        answer=decision.answer,
                        structured_data=decision.structured_data,
                        source_url=current_url,
                        evidence=decision.evidence,
                        confidence=decision.confidence,
                        trace=state["trace"],
                    )
                    span_log(
                        post_span,
                        output={
                            "result": "extract",
                            "trace_steps": len(state["trace"]),
                        },
                    )
                    return state

                if decision.action == "fail":
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
                            answer=fallback_decision.answer,
                            structured_data=fallback_decision.structured_data,
                            source_url=current_url,
                            evidence=fallback_decision.evidence,
                            confidence=fallback_decision.confidence,
                            trace=state["trace"],
                        )
                        span_log(
                            post_span,
                            output={
                                "result": "fallback_extract",
                                "trace_steps": len(state["trace"]),
                            },
                        )
                        return state

                    span_log(
                        post_span,
                        output={
                            "result": "fail",
                            "reason": decision.reason or decision.next_step,
                        },
                    )
                    return _set_error(state, decision.reason or decision.next_step)

            span_log(post_span, output={"result": "advance"})
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
    max_steps: int = 10,
    max_actions_per_step: int = 1,
    extraction_schema: dict[str, str] | None = None,
    extraction_selector: str | None = None,
    headless: bool = True,
    on_step: StepCallback | None = None,
    trace_id: str | None = None,
    trace_parent: str | None = None,
) -> AgentResult:
    if max_actions_per_step < 1 or max_actions_per_step > 4:
        raise ValueError("max_actions_per_step must be between 1 and 4")
    normalized_extraction_schema = _normalize_extraction_schema(extraction_schema)
    normalized_extraction_selector = (
        extraction_selector.strip() if extraction_selector is not None else None
    )
    if extraction_selector is not None and not normalized_extraction_selector:
        raise ValueError("extraction_selector_empty")

    resolved_trace_id = trace_id or _new_trace_id()
    with start_span(
        name="auto_browse_agent_run",
        span_type="task",
        parent=trace_parent,
        metadata={
            "run_id": resolved_trace_id,
            "max_steps": max_steps,
            "max_actions_per_step": max_actions_per_step,
        },
        input={
            "start_url": start_url,
            "target_prompt": target_prompt,
            "headless": headless,
            "extraction_schema_fields": sorted(normalized_extraction_schema.keys())
            if normalized_extraction_schema
            else None,
            "extraction_selector": normalized_extraction_selector,
        },
        tags=[f"run_id:{resolved_trace_id}"],
    ) as run_span:
        pw, browser, page = await run_browser(start_url, headless=headless)
        runtime = _Runtime(
            openrouter_client=openrouter_client,
            page=page,
            target_prompt=target_prompt,
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

        try:
            final_state = await graph.ainvoke(initial_state)
            result = final_state.get("result")
            if result is None:
                fallback_result = AgentResult(
                    error="graph_finished_without_result",
                    trace=final_state.get("trace", []),
                )
                span_log(
                    run_span,
                    output={
                        "error": fallback_result.error,
                        "trace_steps": len(fallback_result.trace),
                    },
                )
                return fallback_result
            span_log(
                run_span,
                output={
                    "final_result": _result_for_root_span_log(result),
                },
            )
            return result
        except Exception as exc:
            span_log(run_span, error=str(exc))
            raise
        finally:
            await browser.close()
            await pw.stop()
