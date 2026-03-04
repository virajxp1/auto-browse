from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI

DEFAULT_OPENROUTER_CONFIG_PATH = "config/config.ini"


def _get_required_env_any(names: list[str]) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    joined = ", ".join(names)
    raise ValueError(f"Missing required environment variable: one of [{joined}]")


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


def _resolve_openrouter_config_path() -> Path:
    configured_path = os.getenv(
        "AUTO_BROWSE_OPENROUTER_CONFIG_PATH",
        DEFAULT_OPENROUTER_CONFIG_PATH,
    ).strip()
    if not configured_path:
        raise ValueError("AUTO_BROWSE_OPENROUTER_CONFIG_PATH cannot be empty")
    return Path(configured_path)


def _read_model_name_from_config(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Missing OpenRouter config file: {path}")

    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error as exc:
        raise ValueError(f"Failed to parse OpenRouter config file: {path}") from exc

    if "openrouter" not in parser:
        raise ValueError("OpenRouter config file must contain an [openrouter] section")

    model_name = parser.get("openrouter", "model", fallback="").strip()
    if not model_name:
        raise ValueError("OpenRouter config file must set [openrouter].model")

    return model_name


@dataclass
class OpenRouterClient:
    api_key: str
    model_name: str

    @classmethod
    def from_env(cls) -> "OpenRouterClient":
        _load_env_file_if_present()
        api_key = _get_required_env_any(["OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"])
        config_path = _resolve_openrouter_config_path()
        model_name = _read_model_name_from_config(config_path)
        return cls(api_key=api_key, model_name=model_name)

    def chat_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
