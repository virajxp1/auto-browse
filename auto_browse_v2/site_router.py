from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
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

# Subdomain/path patterns that signal a technical docs or package-registry site.
# DDG finds exact deep-link pages on these far better than LLM URL guessing.
# Pattern-based so it covers any docs site, not just ones we've seen in eval.
_DOCS_SUBDOMAIN_PATTERNS = ("docs.", "developer.", "developers.", "pkg.", "doc.")
_DOCS_TLD_PATTERNS = (".dev", ".io/docs", ".io/api")
_PACKAGE_REGISTRY_HOSTS = ("npmjs.com", "pypi.org", "pkg.go.dev", "crates.io", "rubygems.org")

_ddg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ddg-worker")


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


def _ddg_search_sync(query: str, max_results: int = 5) -> list[str]:
    """Run DuckDuckGo search synchronously — called in executor."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        return [r["href"] for r in results if r.get("href")]
    except Exception as exc:
        logger.warning("[ddg] search failed for %r: %s", query, exc)
        return []


def _strip_www_prefix(host: str) -> str:
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def _url_matches_hint(url: str, site_hint: str) -> bool:
    """Check if a DDG result URL belongs to the expected domain.

    The result host must equal the hint host or be a subdomain of it —
    never the other way around, to prevent attacker-controlled hosts like
    'evildeveloper.mozilla.org.evil.com' from matching 'developer.mozilla.org'.
    """
    try:
        result_host = _strip_www_prefix(urlsplit(url).hostname or "")
        raw_hint = f"https://{site_hint}" if "://" not in site_hint else site_hint
        hint_host = _strip_www_prefix(urlsplit(raw_hint).hostname or "")
        if not result_host or not hint_host:
            return False
        # result must equal hint or be a subdomain: result ends with ".{hint}"
        return result_host == hint_host or result_host.endswith(f".{hint_host}")
    except Exception:
        return False


def _should_use_ddg(site_hint: str) -> bool:
    """Return True if site_hint looks like a docs/API/package-registry site.

    Uses structural URL patterns (subdomain prefix, TLD suffix, known registries)
    rather than a hardcoded domain allowlist, so it generalises to new sites.
    """
    h = site_hint.lower()
    # Strip scheme and www prefix properly (not lstrip which strips chars)
    for prefix in ("https://www.", "http://www.", "https://", "http://", "www."):
        if h.startswith(prefix):
            h = h[len(prefix):]
            break
    if any(h.startswith(p) for p in _DOCS_SUBDOMAIN_PATTERNS):
        return True
    if any(p in h for p in _DOCS_TLD_PATTERNS):
        return True
    if any(r in h for r in _PACKAGE_REGISTRY_HOSTS):
        return True
    return False


async def _ddg_lookup(query: str, site_hint: str) -> str | None:
    """Search DDG and return the first result URL matching site_hint, or None."""
    loop = asyncio.get_running_loop()
    urls = await loop.run_in_executor(_ddg_executor, _ddg_search_sync, query)
    for url in urls:
        if _url_matches_hint(url, site_hint):
            logger.info("[ddg] found matching URL for %r: %s", query, url)
            return url
    logger.debug("[ddg] no matching URL for hint %r among %d results", site_hint, len(urls))
    return None


async def route_subtask(subtask: SubTask, *, client: LLMClient) -> str:
    # Run DDG search and LLM URL generation in parallel
    use_ddg = _should_use_ddg(subtask.site_hint)
    ddg_query = f"{subtask.description} site:{subtask.site_hint}" if use_ddg else ""

    if use_ddg:
        ddg_task = asyncio.create_task(_ddg_lookup(ddg_query, subtask.site_hint))
    else:
        ddg_task = None

    llm_task = asyncio.create_task(
        client.json_completion(
            system=_SYSTEM_PROMPT,
            user=_USER_TEMPLATE.format(
                description=subtask.description,
                site_hint=subtask.site_hint,
                params=subtask.params or {},
            ),
        )
    )

    # Prefer DDG result for docs domains — it's more accurate than LLM for exact pages
    if ddg_task is not None:
        # return_exceptions=True so an LLM failure doesn't discard a valid DDG URL
        ddg_url, llm_result = await asyncio.gather(ddg_task, llm_task, return_exceptions=True)
        if ddg_url and not isinstance(ddg_url, BaseException) and _is_valid_url(ddg_url):
            logger.info("[%s] DDG URL selected: %s", subtask.task_id, ddg_url)
            return ddg_url
        # DDG missed — fall through to LLM result (or re-raise if LLM also failed)
        if isinstance(llm_result, BaseException):
            raise llm_result
        raw = llm_result
    else:
        raw = await llm_task
        ddg_url = None

    url = str(raw.get("url", "")).strip()
    if _is_valid_url(url):
        logger.info("[%s] LLM URL: %s", subtask.task_id, url)
        return url

    fallback = _build_fallback_url(subtask.site_hint)
    if not _is_valid_url(fallback):
        raise ValueError(f"Invalid site_hint for fallback URL: {subtask.site_hint!r}")

    logger.warning("[%s] LLM returned invalid URL %r; falling back to %s", subtask.task_id, url, fallback)
    return fallback
