from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, Field

from agent.browser import capture_state, run_browser
from agent.extract import page_to_markdown
from agent.models import AgentDecision, AgentResult, AgentStepTrace, PageState
from agent.planner import OpenRouterClient, build_llm_messages

StepCallback = Callable[[AgentStepTrace], None]


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TypeAndSubmitArgs(_StrictArgs):
    selector: str
    text: str
    step_summary: str
    next_step: str


class ClickArgs(_StrictArgs):
    selector: str
    step_summary: str
    next_step: str


class NavigateArgs(_StrictArgs):
    url: str
    step_summary: str
    next_step: str


class ExtractAnswerArgs(_StrictArgs):
    answer: str
    evidence: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    step_summary: str
    next_step: str


class FailArgs(_StrictArgs):
    reason: str
    step_summary: str
    next_step: str


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
    @tool("type_and_submit", args_schema=TypeAndSubmitArgs)
    async def type_and_submit(
        selector: str,
        text: str,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Type into an input and submit with Enter."""
        decision = AgentDecision(
            action="type_and_submit",
            selector=selector,
            text=text,
            step_summary=step_summary,
            next_step=next_step,
        )
        try:
            await runtime.page.focus(selector)
            await runtime.page.fill(selector, text)
            await runtime.page.press(selector, "Enter")
            await _wait_domcontentloaded(runtime.page, timeout=10000)
            return decision.model_dump_json()
        except Exception:
            return _fail_decision_json("type_and_submit_failed", step_summary, next_step)

    @tool("click", args_schema=ClickArgs)
    async def click(
        selector: str,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Click a visible element."""
        decision = AgentDecision(
            action="click",
            selector=selector,
            step_summary=step_summary,
            next_step=next_step,
        )
        try:
            await runtime.page.click(selector)
            await _wait_domcontentloaded(runtime.page, timeout=10000)
            return decision.model_dump_json()
        except Exception:
            return _fail_decision_json("click_failed", step_summary, next_step)

    @tool("navigate", args_schema=NavigateArgs)
    async def navigate(
        url: str,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Navigate to an absolute URL."""
        if not url.startswith(("http://", "https://")):
            return _fail_decision_json("navigate_requires_absolute_url", step_summary, next_step)

        decision = AgentDecision(
            action="navigate",
            url=url,
            step_summary=step_summary,
            next_step=next_step,
        )
        try:
            await runtime.page.goto(url, wait_until="domcontentloaded", timeout=15000)
            return decision.model_dump_json()
        except Exception:
            return _fail_decision_json("navigate_failed", step_summary, next_step)

    @tool("extract_answer", args_schema=ExtractAnswerArgs)
    async def extract_answer(
        answer: str,
        evidence: str,
        confidence: float | None,
        step_summary: str,
        next_step: str,
    ) -> str:
        """Return a final extracted answer from current page evidence."""
        return AgentDecision(
            action="extract",
            answer=answer,
            evidence=evidence,
            confidence=confidence,
            step_summary=step_summary,
            next_step=next_step,
        ).model_dump_json()

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

        message = await llm_with_tools.ainvoke(
            build_llm_messages(
                state["page_state"],
                runtime.target_prompt,
                history=state["trace"],
            )
        )
        if not isinstance(message, AIMessage):
            return _set_error(state, "llm_response_not_ai_message")
        if not message.tool_calls:
            return _set_error(state, "llm_returned_no_tool_call")
        state["messages"] = [message]
        return state

    async def post_tool_node(state: AgentGraphState) -> AgentGraphState:
        if state["result"] is not None:
            return state
        if state["page_state"] is None:
            return _set_error(state, "missing_page_state")
        if not state["messages"]:
            return _set_error(state, "missing_tool_message")

        last_message = state["messages"][-1]
        if not isinstance(last_message, ToolMessage):
            return _set_error(state, "last_message_not_tool")

        try:
            decision = _decision_from_tool_message(last_message)
        except Exception:
            return _set_error(state, "invalid_tool_output")

        step_trace = AgentStepTrace(
            step=state["step"],
            url=state["page_state"].url,
            title=state["page_state"].title,
            decision=decision,
        )
        state["trace"] = state["trace"] + [step_trace]
        if runtime.on_step:
            runtime.on_step(step_trace)

        page_state = state["page_state"]
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
