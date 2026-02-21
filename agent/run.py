from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, Field

from agent.browser import capture_state, run_browser
from agent.extract import page_to_markdown
from agent.models import AgentDecision, AgentResult, AgentStepTrace, PageState
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
    answer: str
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
    on_step: StepCallback | None


class AgentGraphState(TypedDict):
    step: int
    trace: list[AgentStepTrace]
    page_state: PageState | None
    result: AgentResult | None
    messages: list[AIMessage | ToolMessage]


def _set_error(state: AgentGraphState, error: str) -> AgentGraphState:
    state["result"] = AgentResult(error=error, trace=state["trace"])
    return state


def _advance(state: AgentGraphState) -> AgentGraphState:
    state["step"] += 1
    state["page_state"] = None
    state["messages"] = []
    return state


async def _wait_domcontentloaded(page: Page, timeout: int = 10000) -> None:
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except Exception:
        pass


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
            await runtime.page.focus(selector)
            await runtime.page.fill(selector, text)
            await runtime.page.press(selector, "Enter")

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
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    await runtime.page.wait_for_selector(selector, state="visible", timeout=4000)
                except Exception:
                    pass

                try:
                    await runtime.page.click(selector, timeout=5000)
                    return
                except Exception as exc:
                    last_error = exc

                if selector.startswith("css="):
                    css_selector = selector[len("css=") :].strip()
                    base_selector = css_selector
                    nth_index: int | None = None
                    if " >> nth=" in css_selector:
                        base_selector, nth_part = css_selector.rsplit(" >> nth=", 1)
                        base_selector = base_selector.strip()
                        try:
                            nth_index = int(nth_part.strip())
                        except ValueError:
                            # Malformed nth selector should not fall back to first-match click.
                            continue
                        if nth_index < 0:
                            continue
                    if not base_selector:
                        continue
                    try:
                        clicked = await runtime.page.evaluate(
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
                        if clicked:
                            await runtime.page.wait_for_timeout(300)
                            return
                    except Exception:
                        pass

                if attempt == 0:
                    await runtime.page.wait_for_timeout(400)

            if last_error is not None:
                raise last_error
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
        answer: str,
        evidence: str,
        confidence: float | None,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Return a final extracted answer from current page evidence."""
        return _decision_json(
            action="extract",
            answer=answer,
            evidence=evidence,
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
    llm = runtime.openrouter_client.chat_model()
    try:
        llm_with_tools = llm.bind_tools(tools, tool_choice="required")
    except TypeError:
        llm_with_tools = llm.bind_tools(tools)

    async def capture_node(state: AgentGraphState) -> AgentGraphState:
        if state["result"] is not None:
            return state

        page_state = await capture_state(runtime.page)
        page_state.markdown = await page_to_markdown(runtime.page)

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
        )
        message = await llm_with_tools.ainvoke(
            base_messages
        )
        if not isinstance(message, AIMessage):
            return _set_error(state, "llm_response_not_ai_message")

        tool_calls = message.tool_calls or []
        if not tool_calls:
            retry_message = await llm_with_tools.ainvoke(
                base_messages
                + [
                    HumanMessage(
                        content=(
                            "You returned no tool call. "
                            "Call exactly one tool now. "
                            "Do not repeat a blocked or identical previous action."
                        )
                    )
                ]
            )
            if isinstance(retry_message, AIMessage):
                message = retry_message
                tool_calls = message.tool_calls or []

        if not tool_calls:
            return _set_error(state, "llm_returned_no_tool_call")
        if len(tool_calls) != 1:
            return _set_error(state, "llm_returned_multiple_tool_calls")
        state["messages"] = [message]
        return state

    async def post_tool_node(state: AgentGraphState) -> AgentGraphState:
        if state["result"] is not None:
            return state
        try:
            page_state = state["page_state"]
            if page_state is None:
                raise ValueError("missing_page_state")
            last_message = state["messages"][-1]
            if not isinstance(last_message, ToolMessage):
                raise TypeError("last_message_not_tool")
            decision = _decision_from_tool_message(last_message)
        except Exception:
            return _set_error(state, "invalid_tool_output")

        step_trace = AgentStepTrace(
            step=state["step"],
            url=page_state.url,
            title=page_state.title,
            decision=decision,
        )
        state["trace"] = state["trace"] + [step_trace]
        if runtime.on_step:
            runtime.on_step(step_trace)

        if decision.action == "extract":
            state["result"] = AgentResult(
                answer=decision.answer,
                source_url=page_state.url,
                evidence=decision.evidence,
                confidence=decision.confidence,
                trace=state["trace"],
            )
            return state

        if decision.action == "fail":
            return _set_error(state, decision.reason or decision.next_step)

        return _advance(state)

    def route_after_post_tool(state: AgentGraphState) -> str:
        if state["result"] is not None:
            return END
        return "capture"

    def route_after_llm(state: AgentGraphState) -> str:
        if state["result"] is not None:
            return END
        return "tool_node"

    graph = StateGraph(AgentGraphState)
    graph.add_node("capture", capture_node)
    graph.add_node("llm", llm_node)
    graph.add_node("tool_node", ToolNode(tools))
    graph.add_node("post_tool", post_tool_node)

    graph.add_edge(START, "capture")
    graph.add_edge("capture", "llm")
    graph.add_conditional_edges("llm", route_after_llm, {"tool_node": "tool_node", END: END})
    graph.add_edge("tool_node", "post_tool")
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
    headless: bool = True,
    on_step: StepCallback | None = None,
) -> AgentResult:
    pw, browser, page = await run_browser(start_url, headless=headless)
    runtime = _Runtime(
        openrouter_client=openrouter_client,
        page=page,
        target_prompt=target_prompt,
        max_steps=max_steps,
        on_step=on_step,
    )
    graph = _build_graph(runtime)

    initial_state: AgentGraphState = AgentGraphState(
        step=0,
        trace=[],
        page_state=None,
        result=None,
        messages=[],
    )

    try:
        final_state = await graph.ainvoke(initial_state)
        result = final_state.get("result")
        if result is None:
            return AgentResult(error="graph_finished_without_result", trace=final_state.get("trace", []))
        return result
    finally:
        await browser.close()
        await pw.stop()
