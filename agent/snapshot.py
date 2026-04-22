from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from playwright.async_api import Page

from agent.browser import capture_state
from agent.extract import page_to_markdown
from agent.models import PageState
from agent.perception import enrich_page_state

_CAPTURE_STATE_TIMEOUT_SECONDS = 5.0
_MARKDOWN_CAPTURE_TIMEOUT_SECONDS = 5.0
_FALLBACK_TEXT_MAX_CHARS = 6000
_THIN_RESULTS_RETRY_DELAY_SECONDS = 0.6
_THIN_RESULTS_NETWORKIDLE_TIMEOUT_MS = 1500
_THIN_RESULTS_MAX_RETRIES = 3


async def capture_dom_signature(page: Page) -> tuple[str, str, str]:
    url = getattr(page, "url", "") or ""

    title = ""
    title_fn = getattr(page, "title", None)
    if callable(title_fn):
        try:
            title = await title_fn()
        except Exception:
            title = ""

    dom_signature = ""
    evaluate_fn = getattr(page, "evaluate", None)
    if callable(evaluate_fn):
        try:
            signature = await evaluate_fn(
                """() => {
                    const body = document.body;
                    if (!body) return "";
                    const text = (body.innerText || body.textContent || "")
                      .replace(/\\s+/g, " ")
                      .trim()
                      .slice(0, 1200);
                    const count = document.querySelectorAll("a,button,input,textarea,select,form").length;
                    return `${count}|${text}`;
                }"""
            )
            dom_signature = str(signature or "")
        except Exception:
            dom_signature = ""

    return (url, title, dom_signature)


async def _page_title(page: Page) -> str:
    title_fn = getattr(page, "title", None)
    if callable(title_fn):
        try:
            return str(await title_fn())
        except Exception:
            return ""
    return ""


async def fallback_visible_text(page: Page, *, max_chars: int = _FALLBACK_TEXT_MAX_CHARS) -> str:
    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return ""

    try:
        text = await evaluate_fn(
            """(maxChars) => {
                const body = document.body;
                if (!body) return "";
                return String(body.innerText || body.textContent || "")
                    .replace(/\\s+/g, " ")
                    .trim()
                    .slice(0, maxChars);
            }""",
            max_chars,
        )
    except Exception:
        return ""
    return str(text or "").strip()


def _body_link_count(page_state: PageState) -> int:
    return sum(
        item.kind == "link" and (item.region or "body") in {"main", "form", "body", "aside"}
        for item in page_state.interactables
    )


def _snapshot_signal_score(page_state: PageState) -> int:
    body_links = _body_link_count(page_state)
    body_buttons = sum(
        item.kind == "button" and (item.region or "body") in {"main", "form", "body", "aside"}
        for item in page_state.interactables
    )
    visible_text_score = min(len(page_state.markdown.strip()) // 250, 12)
    return (body_links * 10) + (body_buttons * 2) + visible_text_score


def _looks_like_thin_result_page(page_state: PageState) -> bool:
    text = f"{page_state.title}\n{page_state.markdown[:1200]}".lower()
    result_like = page_state.page_archetype == "search_results" or any(
        phrase in text for phrase in ("search results", "results for", "search result")
    )
    if not result_like:
        return False
    return _body_link_count(page_state) == 0


def budget_page_state(
    page_state: PageState,
    *,
    markdown_chars: int,
    interactable_limit: int,
) -> PageState:
    return page_state.model_copy(
        update={
            "markdown": page_state.markdown[:markdown_chars],
            "interactables": page_state.interactables[:interactable_limit],
        }
    )


@dataclass
class PageSnapshotService:
    page: Page
    extraction_selector: str | None = None
    capture_state_fn: Callable[[Page], Awaitable[PageState]] = capture_state
    markdown_fn: Callable[..., Awaitable[str]] = page_to_markdown
    _cached_state: PageState | None = None

    def invalidate(self) -> None:
        self._cached_state = None

    async def _minimal_page_state(self) -> PageState:
        return PageState(
            url=getattr(self.page, "url", "") or "",
            title=await _page_title(self.page),
            markdown="",
            interactables=[],
        )

    async def _capture_once(self) -> PageState:
        try:
            page_state = await asyncio.wait_for(
                self.capture_state_fn(self.page),
                timeout=_CAPTURE_STATE_TIMEOUT_SECONDS,
            )
        except Exception:
            page_state = await self._minimal_page_state()

        try:
            markdown = await asyncio.wait_for(
                self.markdown_fn(
                    self.page,
                    selector=self.extraction_selector,
                ),
                timeout=_MARKDOWN_CAPTURE_TIMEOUT_SECONDS,
            )
        except Exception:
            markdown = ""

        if not markdown:
            markdown = await fallback_visible_text(self.page)

        page_state.markdown = markdown
        return enrich_page_state(page_state)

    async def _wait_for_result_hydration(self) -> None:
        wait_for_load_state_fn = getattr(self.page, "wait_for_load_state", None)
        if callable(wait_for_load_state_fn):
            try:
                await wait_for_load_state_fn("networkidle", timeout=_THIN_RESULTS_NETWORKIDLE_TIMEOUT_MS)
            except Exception:
                pass
        await asyncio.sleep(_THIN_RESULTS_RETRY_DELAY_SECONDS)

    async def capture(self, *, force: bool = False) -> PageState:
        if not force and self._cached_state is not None:
            return self._cached_state

        enriched_page_state = await self._capture_once()
        if _looks_like_thin_result_page(enriched_page_state):
            best_page_state = enriched_page_state
            best_score = _snapshot_signal_score(best_page_state)
            for _ in range(_THIN_RESULTS_MAX_RETRIES):
                await self._wait_for_result_hydration()
                retried_page_state = await self._capture_once()
                retried_score = _snapshot_signal_score(retried_page_state)
                if retried_score > best_score:
                    best_page_state = retried_page_state
                    best_score = retried_score
                if not _looks_like_thin_result_page(best_page_state):
                    break
            enriched_page_state = best_page_state

        self._cached_state = enriched_page_state
        return enriched_page_state
