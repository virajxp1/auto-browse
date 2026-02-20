from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.models import AgentStepTrace, PageState

SYSTEM_PROMPT = """You are a web extraction agent.
You MUST call exactly one tool on every turn.
Never respond with plain text.

Available tools:
- type_and_submit(selector, text, step_summary, next_step)
- click(selector, step_summary, next_step)
- navigate(url, step_summary, next_step)
- extract_answer(answer, evidence, confidence, step_summary, next_step)
- fail(reason, step_summary, next_step)

Rules:
- Prefer extract_answer only when evidence is present in PAGE TEXT.
- Prefer type_and_submit if a search input exists and answer is not present.
- Use click only with provided selectors.
- Use navigate only with absolute URLs.
- step_summary and next_step must be concise and specific.
- For extract_answer, set next_step to \"Return extracted answer now.\"
- For fail, provide a concrete reason.
"""

def build_prompt(state: PageState, target: str, history: list[AgentStepTrace] | None = None) -> str:
    interactables = "\n".join(
        (
            f"- {item.kind}: {item.label} | selector={item.selector}"
            + (f" | href={item.href}" if item.href else "")
        )
        for item in state.interactables[:25]
    )

    history_text = "none"
    if history:
        history_text = "\n".join(
            f"- step={item.step} action={item.decision.action} summary={item.decision.step_summary}"
            for item in history[-3:]
        )

    markdown = state.markdown[:8000]

    return f"""TARGET:
{target}

CURRENT PAGE:
url: {state.url}
title: {state.title}

PAGE TEXT (trimmed):
{markdown}

INTERACTABLES:
{interactables}

RECENT TRACE:
{history_text}

Call exactly one tool now.
"""

def build_llm_messages(
    state: PageState,
    target: str,
    history: list[AgentStepTrace] | None = None,
):
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(state, target, history=history)),
    ]
