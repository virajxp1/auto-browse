from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openai/gpt-4o-mini"
_ENV_VARS = ["OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"]

# SmartScraperGraph.run() calls asyncio.run() internally, so it must be executed
# in a thread where no event loop is running. max_workers=4 matches expected
# parallelism (most queries decompose into 2-4 subtasks).
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sgai-worker")


def _resolve_api_key() -> str:
    for name in _ENV_VARS:
        val = os.getenv(name, "").strip()
        if val:
            return val
    raise ValueError(f"Missing env var: one of {_ENV_VARS}")


def _build_config(api_key: str, model: str) -> dict[str, Any]:
    return {
        "llm": {
            "api_key": api_key,
            "model": model,
            "base_url": _OPENROUTER_BASE_URL,
        },
        "embeddings": None,   # suppress default OpenAI embeddings — no vector store needed
        "verbose": False,
        "headless": True,
    }


# 200k chars ≈ 50k tokens after markdown conversion — leaves ample room in gpt-4o's 128k context
_MAX_HTML_CHARS = 200_000


def _run_sync(
    prompt: str,
    html_source: str,
    source_url: str,
    *,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    """Synchronous ScrapeGraphAI call — runs inside ThreadPoolExecutor."""
    html_source = html_source[:_MAX_HTML_CHARS]  # guard against context overflow
    try:
        from scrapegraphai.graphs import SmartScraperGraph  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "scrapegraphai is not installed. Add 'scrapegraphai>=1.13.0' to pyproject.toml"
        ) from exc

    config = _build_config(api_key, model)

    # SGAI checks: if source starts with "http" → URL fetch (re-downloads page).
    # Any other string → handle_local_source (uses our pre-fetched HTML directly).
    # Since HTML content never starts with "http", we get direct HTML processing,
    # avoiding redundant network requests and bypassing bot-detection on re-fetch.
    graph = SmartScraperGraph(prompt=prompt, source=html_source, config=config)

    result = graph.run()
    return result if isinstance(result, dict) else {"raw": str(result)}


async def smart_scrape(
    prompt: str,
    html_source: str,
    source_url: str,
    *,
    model: str = _DEFAULT_MODEL,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run SmartScraperGraph in a thread executor to avoid blocking the event loop."""
    resolved_key = api_key or _resolve_api_key()
    loop = asyncio.get_running_loop()
    fn = partial(_run_sync, prompt, html_source, source_url, api_key=resolved_key, model=model)
    return await loop.run_in_executor(_executor, fn)
