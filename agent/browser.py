from __future__ import annotations

import re

from playwright.async_api import Browser, Page, Playwright, async_playwright

from agent.models import Interactable, PageState

_INPUT_QUERY = "input[type='text'], input[type='search'], textarea"
_BUTTON_QUERY = "button, input[type='submit'], input[type='button']"
_LINK_QUERY = "a[href]"

_VISIBLE_SCRIPT = """(el, requireEnabled) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== "none" &&
           style.visibility !== "hidden" &&
           rect.width > 0 &&
           rect.height > 0 &&
           (!requireEnabled || !el.disabled);
}"""


def _clean_label(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    collapsed = re.sub(r"\s+", " ", value).strip()
    return collapsed[:80] if collapsed else fallback


async def _is_visible(handle, *, require_enabled: bool = False) -> bool:
    try:
        return bool(await handle.evaluate(_VISIBLE_SCRIPT, require_enabled))
    except Exception:
        return False


async def _build_input_interactables(page: Page, limit: int = 10) -> list[Interactable]:
    interactables: list[Interactable] = []
    handles = await page.query_selector_all(_INPUT_QUERY)
    for i, handle in enumerate(handles[:limit]):
        if not await _is_visible(handle, require_enabled=True):
            continue
        try:
            input_name = await handle.get_attribute("name")
            input_id = await handle.get_attribute("id")
            aria = await handle.get_attribute("aria-label")
            placeholder = await handle.get_attribute("placeholder")
        except Exception:
            continue

        label = _clean_label(
            aria or placeholder or input_name or input_id,
            fallback=f"input_{i}",
        )
        interactables.append(
            Interactable(
                kind="input",
                label=label,
                selector=f"css={_INPUT_QUERY} >> nth={i}",
                href=None,
            )
        )
    return interactables


async def _build_button_interactables(page: Page, limit: int = 10) -> list[Interactable]:
    interactables: list[Interactable] = []
    handles = await page.query_selector_all(_BUTTON_QUERY)

    for i, handle in enumerate(handles[:limit]):
        if not await _is_visible(handle, require_enabled=True):
            continue
        try:
            button_name = await handle.get_attribute("name")
            button_id = await handle.get_attribute("id")
            aria = await handle.get_attribute("aria-label")
            text = await handle.inner_text()
        except Exception:
            continue

        label = _clean_label(
            text or aria or button_name or button_id,
            fallback=f"button_{i}",
        )
        interactables.append(
            Interactable(
                kind="button",
                label=label,
                selector=f"css={_BUTTON_QUERY} >> nth={i}",
                href=None,
            )
        )

    return interactables


async def _build_link_interactables(page: Page, limit: int = 15) -> list[Interactable]:
    interactables: list[Interactable] = []
    handles = await page.query_selector_all(_LINK_QUERY)
    for i, handle in enumerate(handles[:limit]):
        if not await _is_visible(handle):
            continue
        try:
            href = await handle.get_attribute("href")
            text = await handle.inner_text()
        except Exception:
            continue

        label = _clean_label(text, fallback=f"link_{i}")

        interactables.append(
            Interactable(
                kind="link",
                label=label,
                selector=f"css={_LINK_QUERY} >> nth={i}",
                href=href,
            )
        )

    return interactables


async def capture_state(page: Page) -> PageState:
    title = await page.title()
    url = page.url

    interactables: list[Interactable] = []
    interactables.extend(await _build_input_interactables(page))
    interactables.extend(await _build_button_interactables(page))
    interactables.extend(await _build_link_interactables(page))

    return PageState(url=url, title=title, markdown="", interactables=interactables)


async def run_browser(start_url: str, *, headless: bool = True) -> tuple[Playwright, Browser, Page]:
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=headless)
    page = await browser.new_page()
    await page.goto(start_url, wait_until="domcontentloaded")
    return pw, browser, page
