from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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


def _get_required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    joined = ", ".join(names)
    raise ValueError(f"Missing required environment variable. Expected one of: {joined}")


def _load_env_file_if_present(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


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


@dataclass
class OpenRouterClient:
    api_key: str
    model_name: str
    base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def from_env(cls) -> "OpenRouterClient":
        _load_env_file_if_present()
        api_key = _get_required_env("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY")
        model_name = _get_required_env("OPENROUTER_MODEL", "LLM_MODEL_NAME")
        return cls(api_key=api_key, model_name=model_name)

    def chat_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0,
        )


def build_llm_messages(
    state: PageState,
    target: str,
    history: list[AgentStepTrace] | None = None,
):
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(state, target, history=history)),
    ]
