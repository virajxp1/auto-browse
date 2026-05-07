from __future__ import annotations

import logging
from urllib.parse import urlsplit

from auto_browse_v2.llm_client import LLMClient
from auto_browse_v2.models import SubTask

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a URL routing specialist.

Given a subtask with a site_hint and optional params, produce the best starting URL
for a browser — ideally a deep-link results page that already has search parameters baked in.

Return a JSON object with exactly one key:
  - url: string (fully qualified URL starting with https://)

Guidelines:
- Prefer deep search result URLs over homepages when you know the URL schema.
  Examples:
    * Google Flights: https://www.google.com/travel/flights?q=...
    * Hyatt direct: https://www.hyatt.com/en-US/search?...
    * American Airlines: https://www.aa.com/booking/find-flights?...
- If you cannot construct a deep link with high confidence, return the site's homepage.
- NEVER return a URL that requires authentication.
- Only include URL parameters you are confident about.
"""

_USER_TEMPLATE = """\
Subtask:
  description: {description}
  site_hint: {site_hint}
  params: {params}
"""


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _build_fallback_url(site_hint: str) -> str:
    hint = site_hint.strip()
    if not hint:
        return ""

    if hint.startswith("//"):
        hint = hint[2:]
    elif "://" in hint:
        hint = urlsplit(hint).netloc

    hint = hint.split("/", 1)[0].strip()
    if not hint:
        return ""

    return f"https://{hint}"


async def route_subtask(subtask: SubTask, *, client: LLMClient) -> str:
    raw = await client.json_completion(
        system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(
            description=subtask.description,
            site_hint=subtask.site_hint,
            params=subtask.params or {},
        ),
    )

    url = str(raw.get("url", "")).strip()
    if _is_valid_url(url):
        logger.info("[%s] Routed to: %s", subtask.task_id, url)
        return url

    fallback = _build_fallback_url(subtask.site_hint)
    if not _is_valid_url(fallback):
        raise ValueError(f"Invalid site_hint for fallback URL: {subtask.site_hint!r}")

    logger.warning("[%s] LLM returned invalid URL %r; falling back to %s", subtask.task_id, url, fallback)
    return fallback
