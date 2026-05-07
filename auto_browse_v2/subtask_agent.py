from __future__ import annotations

import logging
from typing import Any

from auto_browse_v2.llm_client import LLMClient
from auto_browse_v2.models import SubTask, SubTaskResult
from auto_browse_v2.navigator import NavigationLoop, NavigationResult
from auto_browse_v2.site_router import route_subtask

logger = logging.getLogger(__name__)

_FORMAT_SYSTEM = """\
You are a research analyst. Given extracted structured data from a web page, write a
concise factual answer (2-5 sentences) that directly addresses the original question.
Focus on prices, availability, dates, flight numbers, and key facts.
If data is absent or the site blocked access, say so clearly.
Return a JSON object with exactly one key: "answer" (string).
"""

_FORMAT_USER_TEMPLATE = """\
Original question: {description}
Source URL: {url}
Extracted data: {data}
"""


async def run_subtask(subtask: SubTask, *, client: LLMClient) -> SubTaskResult:
    """Orchestrate a single SubTask end-to-end.

    Pipeline:
        site_router → NavigationLoop (LLM plans + Playwright executes + ScrapeGraphAI extracts)
        → LLM formats answer → SubTaskResult
    """
    task_id = subtask.task_id
    logger.info("[%s] Starting: %s", task_id, subtask.description)

    try:
        # 1. Route: determine starting URL
        start_url = await route_subtask(subtask, client=client)
        logger.info("[%s] Starting at %s", task_id, start_url)

        # 2. Navigate + extract via LLM-driven loop
        nav = NavigationLoop(client=client)
        result: NavigationResult = await nav.run(goal=subtask.description, start_url=start_url)

        logger.info(
            "[%s] Navigation done — success=%s steps=%d url=%s",
            task_id, result.success, result.steps_taken, result.url,
        )

        # 3. Format: synthesize a human-readable answer from extracted data
        raw = await client.json_completion(
            system=_FORMAT_SYSTEM,
            user=_FORMAT_USER_TEMPLATE.format(
                description=subtask.description,
                url=result.url,
                data=result.data,
            ),
        )
        answer = str(raw.get("answer", "")).strip() or str(result.data)

        return SubTaskResult(
            task_id=task_id,
            status="success" if result.success else "partial",
            answer=answer,
            structured_data=result.data,
            source_url=result.url,
            error=result.error,
        )

    except Exception as exc:
        logger.exception("[%s] Failed: %s", task_id, exc)
        return SubTaskResult(
            task_id=task_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
