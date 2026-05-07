"""Navigation loop: LLM plans actions, Playwright executes, ScrapeGraphAI extracts."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, field_validator

from auto_browse_v2.browser import BrowserSession
from auto_browse_v2.llm_client import LLMClient
from auto_browse_v2.scraper import smart_scrape

logger = logging.getLogger(__name__)

_MAX_STEPS = 10
_SELECTOR_TIMEOUT_MS = 3_000  # per-action timeout — short so blocked clicks fail fast
_SETTLE_TIMEOUT_MS = 2_000
_MAX_PAGE_CHARS = 6_000


# ── Action model ─────────────────────────────────────────────────────────────

class Action(BaseModel):
    """A single browser action decided by the LLM."""

    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    type: str = "extract"             # fill | click | select | press_enter | extract | fail
    selector: str | None = None       # CSS selector for the target element
    value: str | None = None          # text to fill or option to select
    extract_prompt: str | None = None # specific extraction instruction (type=extract only)
    reasoning: str = ""               # brief explanation, for logging

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, v: Any) -> str:
        return str(v) if v is not None else ""

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: Any) -> str:
        return str(v) if v is not None else "extract"


@dataclass
class NavigationResult:
    success: bool
    data: dict[str, Any]
    url: str
    steps_taken: int
    history: list[str] = field(default_factory=list)
    error: str | None = None


# ── Page simplification ───────────────────────────────────────────────────────

def _simplify_page(html: str, url: str) -> str:
    """Convert raw HTML to a compact, LLM-readable page description.

    Extracts interactive elements (inputs, selects, buttons) and key visible
    text.  Stays well under the context budget.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        from markdownify import markdownify  # type: ignore[import]
        md = markdownify(html, strip=["script", "style", "noscript"])
        return f"URL: {url}\n\n{md[:_MAX_PAGE_CHARS]}"

    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.find("title")
    title = title_el.get_text(strip=True) if title_el else ""

    for tag in soup(["script", "style", "noscript", "svg", "head", "footer", "nav"]):
        tag.decompose()

    # Interactive elements: everything actionable
    interactive_lines: list[str] = []
    for el in soup.find_all(["input", "select", "button", "textarea", "a"]):
        relevant_attrs: dict[str, str] = {}
        for attr in ["name", "id", "type", "placeholder", "aria-label", "value", "href", "data-testid"]:
            val = el.get(attr)
            if val:
                relevant_attrs[attr] = str(val)[:60]
        text = el.get_text(strip=True)[:80]
        attr_str = ", ".join(f'{k}="{v}"' for k, v in relevant_attrs.items())
        interactive_lines.append(f"  <{el.name} {attr_str}>{text}</{el.name}>")

    # Key content: headings + first paragraphs
    content_lines: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "td", "span"])[:40]:
        text = el.get_text(strip=True)[:150]
        if len(text) > 20:
            content_lines.append(f"  {el.name}: {text}")

    parts = [
        f"URL: {url}",
        f"Title: {title}",
        f"\nINTERACTIVE ELEMENTS ({len(interactive_lines)}):",
        *interactive_lines[:60],
        "\nPAGE CONTENT (visible text):",
        *content_lines[:30],
    ]
    return "\n".join(parts)[:_MAX_PAGE_CHARS]


# ── LLM action planner ────────────────────────────────────────────────────────

_ACTION_SYSTEM = """\
You are a browser navigation agent. Your job is to reach the page that contains the
information requested, then trigger an extraction.

On every turn you receive:
- The goal (what data to find)
- The current page description (URL, title, interactive elements, visible text)
- The recent action history

Return a JSON object with these fields:
{
  "type": "fill|click|select|press_enter|extract|fail",
  "selector": "CSS selector for the target element (omit for press_enter/extract/fail)",
  "value": "text to fill or option value to select (omit if not applicable)",
  "extract_prompt": "specific instruction of what to extract (only when type=extract)",
  "reasoning": "one sentence explaining why"
}

ACTION TYPES:
- fill:        Populate a text input (date picker, search box, etc.)
- click:       Click a button, link, or interactive element
- select:      Choose a value from a <select> dropdown
- press_enter: Press Enter on the focused/last element (use after fill if no submit button)
- extract:     The CURRENT PAGE already has the data — trigger ScrapeGraphAI extraction
- fail:        No useful action is possible and no data is available

SELECTOR RULES (in order of preference):
1. [name="checkin"], [name="check_in"], [placeholder*="Check-in"] — attribute matchers
2. [data-testid="..."] — test IDs
3. [aria-label="..."] — accessibility labels
4. button:has-text("Search"), a:has-text("Book") — text-content matchers
5. Avoid class-based selectors (they change often)

NAVIGATION STRATEGY:
- If you see a search form with empty date/destination fields → fill them then click submit
- If you see a date picker UI → fill the text input with the date (e.g. "06/17/2026")
- If the page shows actual results (prices, flight list, hotel rates) → extract immediately
- If a page has a cookie/consent banner blocking content → click the accept button first
- If stuck after 3+ failed attempts on the same page → use type=extract to try anyway
- Never navigate away from a results page to try a different approach

WHEN TO EXTRACT:
- You see hotel room prices with the requested dates
- You see a flight results list with prices and times
- You see the specific data requested in the goal
"""


async def _decide_action(
    goal: str,
    page_desc: str,
    step: int,
    history: list[str],
    client: LLMClient,
) -> Action:
    history_text = "\n".join(history[-5:]) if history else "None yet."
    raw = await client.json_completion(
        system=_ACTION_SYSTEM,
        user=(
            f"Goal: {goal}\n"
            f"Step {step + 1} of {_MAX_STEPS}\n\n"
            f"Recent history:\n{history_text}\n\n"
            f"Current page:\n{page_desc}"
        ),
    )
    # LLM sometimes returns a list of actions instead of a single dict — take the first.
    if isinstance(raw, list) and raw:
        raw = raw[0]
    try:
        return Action.model_validate(raw)
    except Exception as exc:
        logger.warning("[nav] Action validation failed (step %d): %s — defaulting to extract", step + 1, exc)
        return Action(type="extract", reasoning=f"Validation fallback: {exc}")


# ── Playwright action executor ────────────────────────────────────────────────

async def _execute(page: Page, action: Action) -> str:
    """Execute an Action on the Playwright page. Returns a description."""
    t = action.type
    sel = action.selector
    val = action.value or ""

    if t == "fill":
        if not sel:
            raise ValueError("fill requires a selector")
        await page.fill(sel, val, timeout=_SELECTOR_TIMEOUT_MS)
        return f"filled {sel!r} with {val!r}"

    elif t == "click":
        if not sel:
            raise ValueError("click requires a selector")
        try:
            await page.click(sel, timeout=_SELECTOR_TIMEOUT_MS)
        except Exception:
            # Fallback 1: visible text match
            if val:
                try:
                    await page.get_by_text(val, exact=False).first.click(timeout=_SELECTOR_TIMEOUT_MS)
                    return f"clicked by text {val!r} (fallback)"
                except Exception:
                    pass
            # Fallback 2: force=True bypasses overlay interception (cookie banners, etc.)
            await page.click(sel, timeout=_SELECTOR_TIMEOUT_MS, force=True)
        return f"clicked {sel!r}"

    elif t == "select":
        if not sel:
            raise ValueError("select requires a selector")
        await page.select_option(sel, value=val, timeout=_SELECTOR_TIMEOUT_MS)
        return f"selected {val!r} in {sel!r}"

    elif t == "press_enter":
        target = sel or "body"
        await page.press(target, "Enter", timeout=_SELECTOR_TIMEOUT_MS)
        return f"pressed Enter on {target!r}"

    else:
        return f"no-op for type={t!r}"


# ── Navigation loop ───────────────────────────────────────────────────────────

class NavigationLoop:
    """LLM-driven navigation loop: plan → act → settle → repeat → extract.

    The loop runs until:
    - The LLM decides the current page has the data (type=extract) → ScrapeGraphAI
    - The LLM gives up (type=fail)
    - Max steps is reached → forced extraction from current page
    """

    def __init__(self, *, client: LLMClient, max_steps: int = _MAX_STEPS) -> None:
        self._client = client
        self._max_steps = max_steps

    async def run(self, goal: str, start_url: str) -> NavigationResult:
        history: list[str] = []

        async with BrowserSession(headless=True, timeout_ms=30_000) as session:
            await session.navigate(start_url)
            logger.info("[nav] started at %s", start_url)

            for step in range(self._max_steps):
                # Wait for page to be stable before reading — prevents "page is navigating" errors
                try:
                    await session.page.wait_for_load_state("domcontentloaded", timeout=5_000)
                except Exception:
                    await asyncio.sleep(0.5)
                html = await session.get_html()
                url = session.get_url()
                page_desc = _simplify_page(html, url)

                action = await _decide_action(goal, page_desc, step, history, self._client)
                logger.info(
                    "[nav step %d/%d] type=%s selector=%r  — %s",
                    step + 1, self._max_steps,
                    action.type, action.selector,
                    action.reasoning[:100],
                )

                # ── Terminal: extract from current page ──
                if action.type == "extract":
                    extract_prompt = action.extract_prompt or goal
                    logger.info("[nav] extracting at step %d: %s", step + 1, url)
                    data = await smart_scrape(extract_prompt, html, url)
                    return NavigationResult(
                        success=True, data=data, url=url,
                        steps_taken=step + 1, history=history,
                    )

                # ── Terminal: agent gave up ──
                if action.type == "fail":
                    logger.warning("[nav] agent gave up at step %d: %s", step + 1, action.reasoning)
                    data = await smart_scrape(goal, html, url)
                    return NavigationResult(
                        success=False, data=data, url=url,
                        steps_taken=step + 1, history=history,
                        error=f"Agent gave up: {action.reasoning}",
                    )

                # ── Execute action ──
                try:
                    desc = await _execute(session.page, action)
                    history.append(f"step {step + 1}: {desc}")
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    history.append(f"step {step + 1}: FAILED {action.type} on {action.selector!r} — {err}")
                    logger.warning("[nav step %d] action failed: %s", step + 1, err)

                # Wait for page to settle — use domcontentloaded to avoid blocking
                # on JS-heavy sites that never reach networkidle
                try:
                    await session.page.wait_for_load_state("domcontentloaded", timeout=_SETTLE_TIMEOUT_MS)
                except Exception:
                    await asyncio.sleep(0.8)

            # Max steps reached — extract from wherever we landed
            html = await session.get_html()
            url = session.get_url()
            logger.warning("[nav] max steps reached; forcing extraction at %s", url)
            data = await smart_scrape(goal, html, url)
            return NavigationResult(
                success=False, data=data, url=url,
                steps_taken=self._max_steps, history=history,
                error="Max navigation steps reached",
            )
