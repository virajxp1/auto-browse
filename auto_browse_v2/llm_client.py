from __future__ import annotations

import asyncio
import configparser
import json
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openai/gpt-4o-mini"
_ENV_VARS = ["OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"]
_CONFIG_PATH = Path("config/config.ini")


def _load_env_file(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def _resolve_api_key() -> str:
    _load_env_file()
    for name in _ENV_VARS:
        val = os.getenv(name, "").strip()
        if val:
            return val
    raise ValueError(f"Missing required env var: one of {_ENV_VARS}")


def _resolve_model() -> str:
    if _CONFIG_PATH.is_file():
        parser = configparser.ConfigParser()
        parser.read(_CONFIG_PATH, encoding="utf-8")
        model = parser.get("openrouter", "model", fallback="").strip()
        if model:
            return model
    return _DEFAULT_MODEL


class LLMClient:
    """Thin async wrapper around the OpenAI-compatible OpenRouter API."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or _resolve_api_key()
        self.model = model or _resolve_model()
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=_OPENROUTER_BASE_URL,
        )

    async def json_completion(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        retries: int = 2,
    ) -> dict:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except (json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                if attempt < retries:
                    logger.warning("json_completion parse error (attempt %d/%d): %s", attempt + 1, retries + 1, exc)
                    await asyncio.sleep(0.5)
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    logger.warning("json_completion error (attempt %d/%d): %s", attempt + 1, retries + 1, exc)
                    await asyncio.sleep(1.0)
        raise last_exc  # type: ignore[misc]
