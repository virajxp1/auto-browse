from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.models import AgentDecision, AgentStepTrace, PageState

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
- Use RECENT TRACE to avoid loops.
- Never repeat the same action with the same selector/text/url on the same page URL.
- If the last attempts are repeating without progress, choose a different action (navigate/click/extract/fail).
- If blocked by captcha/anti-bot or unable to progress, call fail with the concrete blocker.
"""


def _decision_params_text(decision: AgentDecision) -> str:
    parts: list[str] = []
    if decision.selector:
        parts.append(f"selector={decision.selector}")
    if decision.text:
        parts.append(f"text={decision.text}")
    if decision.url:
        parts.append(f"url={decision.url}")
    if decision.reason:
        parts.append(f"reason={decision.reason}")
    if decision.answer:
        parts.append(f"answer={decision.answer}")
    return " | ".join(parts) if parts else "none"


def _build_loop_alerts(history: list[AgentStepTrace]) -> str:
    signatures: dict[tuple[str, str, str, str, str], int] = {}
    for item in history[-8:]:
        decision = item.decision
        signature = (
            item.url,
            decision.action,
            decision.selector or "",
            decision.text or "",
            decision.url or "",
        )
        signatures[signature] = signatures.get(signature, 0) + 1

    repeated = [sig for sig, count in signatures.items() if count >= 2]
    if not repeated:
        return "none"

    lines = []
    for page_url, action, selector, text, nav_url in repeated[:3]:
        parts = [f"page_url={page_url}", f"action={action}"]
        if selector:
            parts.append(f"selector={selector}")
        if text:
            parts.append(f"text={text}")
        if nav_url:
            parts.append(f"url={nav_url}")
        lines.append(f"- repeated: {' | '.join(parts)}")

    lines.append("- do_not_repeat: choose a different action or call fail.")
    return "\n".join(lines)


def _build_blocker_alerts(state: PageState) -> str:
    content = f"{state.url}\n{state.title}\n{state.markdown[:4000]}".lower()
    signals = [
        ("captcha", "captcha"),
        ("google_sorry", "google.com/sorry"),
        ("forbidden", "403"),
        ("access_denied", "access denied"),
        ("enable_javascript", "enable javascript"),
        ("unusual_traffic", "unusual traffic"),
        ("robot_check", "robot check"),
    ]
    found = [name for name, needle in signals if needle in content]
    if not found:
        return "none"
    lines = [f"- detected: {name}" for name in found]
    lines.append("- blocked_guidance: avoid repeating same action; navigate elsewhere or call fail.")
    return "\n".join(lines)


def build_prompt(state: PageState, target: str, history: list[AgentStepTrace] | None = None) -> str:
    interactables = "\n".join(
        (
            f"- {item.kind}: {item.label} | selector={item.selector}"
            + (f" | href={item.href}" if item.href else "")
        )
        for item in state.interactables[:25]
    )
    if not interactables:
        interactables = "none"

    history_text = "none"
    loop_alerts_text = "none"
    blocker_alerts_text = _build_blocker_alerts(state)
    if history:
        history_text = "\n".join(
            (
                f"- step={item.step} page_url={item.url} action={item.decision.action} "
                f"params={_decision_params_text(item.decision)} "
                f"summary={item.decision.step_summary}"
            )
            for item in history[-8:]
        )
        loop_alerts_text = _build_loop_alerts(history)

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

LOOP ALERTS:
{loop_alerts_text}

BLOCKER ALERTS:
{blocker_alerts_text}

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
