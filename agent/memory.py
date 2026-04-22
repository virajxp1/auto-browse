from __future__ import annotations

from dataclasses import dataclass, field

from agent.models import AgentStepTrace

_MAX_SCRATCHPAD_CHARS = 2000
_MAX_URLS = 12
_MAX_FILLED_FIELDS = 10
_MAX_FAILED_ACTIONS = 8


@dataclass
class AgentScratchpad:
    visited_urls: list[str] = field(default_factory=list)
    filled_fields: list[str] = field(default_factory=list)  # "selector=value"
    failed_actions: list[str] = field(default_factory=list)  # "action:reason"
    current_plan_step: str = ""


def update_scratchpad(scratchpad: AgentScratchpad, step: AgentStepTrace) -> AgentScratchpad:
    """Return a new scratchpad updated with the latest step trace."""
    visited_urls = list(scratchpad.visited_urls)
    filled_fields = list(scratchpad.filled_fields)
    failed_actions = list(scratchpad.failed_actions)

    decision = step.decision
    action = decision.action

    # Track visited URL
    if step.url and step.url not in visited_urls:
        visited_urls.append(step.url)
    visited_urls = visited_urls[-_MAX_URLS:]

    # Track fill/type_and_submit fields
    if action in {"fill", "type_and_submit"}:
        selector = decision.selector or decision.interactable_ref or ""
        value = decision.text or decision.value or ""
        if selector and value:
            entry = f"{selector}={value[:40]}"
            if entry not in filled_fields:
                filled_fields.append(entry)
        filled_fields = filled_fields[-_MAX_FILLED_FIELDS:]

    # Track failures
    if action == "fail":
        reason = (decision.reason or "unknown")[:60]
        entry = f"fail:{reason}"
        if entry not in failed_actions:
            failed_actions.append(entry)
        failed_actions = failed_actions[-_MAX_FAILED_ACTIONS:]

    current_plan_step = decision.next_step or scratchpad.current_plan_step

    return AgentScratchpad(
        visited_urls=visited_urls,
        filled_fields=filled_fields,
        failed_actions=failed_actions,
        current_plan_step=current_plan_step,
    )


def format_scratchpad(scratchpad: AgentScratchpad) -> str:
    """Format scratchpad as a compact string for prompt injection."""
    parts: list[str] = []

    if scratchpad.visited_urls:
        parts.append("visited_urls: " + ", ".join(scratchpad.visited_urls))

    if scratchpad.filled_fields:
        parts.append("filled_fields: " + "; ".join(scratchpad.filled_fields))

    if scratchpad.failed_actions:
        parts.append("failed_actions: " + "; ".join(scratchpad.failed_actions))

    if scratchpad.current_plan_step:
        parts.append(f"current_plan_step: {scratchpad.current_plan_step[:120]}")

    text = "\n".join(parts)
    if len(text) > _MAX_SCRATCHPAD_CHARS:
        text = text[:_MAX_SCRATCHPAD_CHARS] + "...(truncated)"
    return text or "none"
