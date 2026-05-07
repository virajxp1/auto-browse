from __future__ import annotations

import logging
import uuid

from auto_browse_v2.llm_client import LLMClient
from auto_browse_v2.models import SubTask

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research query decomposer.

Given a complex multi-part query, break it into independent, self-contained subtasks.
Each subtask targets a SINGLE website or data source.

Return a JSON object with key "subtasks" containing an array of objects, each with:
  - task_id: string (slug, no spaces, e.g. "hyatt-ziva-price")
  - description: string (plain-English, complete standalone question)
  - site_hint: string (target domain or brand, e.g. "hyatt.com", "aa.com")
  - params: object (key/value pairs: dates, origin, destination, cabin class, etc.)

Rules:
- Do NOT combine multiple sites into one subtask.
- Preserve date ranges verbatim in params (e.g. {"check_in": "2024-06-17", "check_out": "2024-06-20"}).
- Use ISO 8601 dates where the query provides them.
- If the query is already single-site, return exactly one subtask.
"""

_USER_TEMPLATE = 'Query: "{query}"'


async def decompose_query(query: str, *, client: LLMClient) -> list[SubTask]:
    raw = await client.json_completion(
        system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(query=query),
    )

    subtasks_raw = raw.get("subtasks", [])
    if not isinstance(subtasks_raw, list):
        raise ValueError(f"Decomposer returned unexpected shape: {raw!r}")

    subtasks: list[SubTask] = []
    for item in subtasks_raw:
        if not isinstance(item, dict):
            continue
        if not item.get("task_id", "").strip():
            item["task_id"] = f"task-{uuid.uuid4().hex[:8]}"
        try:
            subtasks.append(SubTask.model_validate(item))
        except Exception as exc:
            logger.warning("Skipping malformed subtask %r: %s", item, exc)

    if not subtasks:
        raise ValueError("Decomposer produced zero valid subtasks")

    logger.info("Decomposed into %d subtask(s): %s", len(subtasks), [s.task_id for s in subtasks])
    return subtasks
