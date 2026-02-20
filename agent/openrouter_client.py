from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI


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
