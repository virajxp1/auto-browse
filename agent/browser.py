from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

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


def _escape_css_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _escape_role_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _scoped_attr_selector(scope_query: str, attr: str, value: str | None) -> str | None:
    if not value:
        return None
    clean = value.strip()
    if not clean:
        return None
    return f'css=:is({scope_query})[{attr}="{_escape_css_value(clean)}"]'


async def _first_unique_selector(page: Page, candidates: list[str | None], fallback: str) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        try:
            matches = await page.query_selector_all(candidate)
        except Exception:
            continue
        if len(matches) == 1:
            return candidate
    return fallback


def _normalize_start_url(start_url: str) -> str:
    normalized = start_url.strip()
    if not normalized:
        raise ValueError("start_url_empty")
    if any(char.isspace() for char in normalized):
        raise ValueError("start_url_contains_whitespace")

    if normalized.startswith("//"):
        normalized = f"https:{normalized}"
    elif "://" in normalized:
        scheme = urlsplit(normalized).scheme.lower()
        if scheme not in {"http", "https"}:
            raise ValueError("start_url_invalid_scheme")
    else:
        scheme_like = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", normalized)
        if scheme_like:
            shorthand_host_port = re.match(
                r"^(?P<host>\[[0-9A-Fa-f:]+\]|[A-Za-z0-9.-]+):(?P<port>\d+)(?:/.*)?$",
                normalized,
            )
            if not shorthand_host_port:
                raise ValueError("start_url_invalid_scheme")
        normalized = f"https://{normalized}"

    parsed = urlsplit(normalized)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("start_url_invalid_port") from exc

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("start_url_missing_host")
    if any(char.isspace() for char in hostname):
        raise ValueError("start_url_invalid_host")

    if hostname != "localhost":
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            if len(hostname) > 253:
                raise ValueError("start_url_invalid_host")
            labels = hostname.split(".")
            if not labels:
                raise ValueError("start_url_invalid_host")
            for label in labels:
                if not label:
                    raise ValueError("start_url_invalid_host")
                if len(label) > 63:
                    raise ValueError("start_url_invalid_host")
                if label.startswith("-") or label.endswith("-"):
                    raise ValueError("start_url_invalid_host")
                if not re.fullmatch(r"[A-Za-z0-9-]+", label):
                    raise ValueError("start_url_invalid_host")

    return normalized


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
    for i, handle in enumerate(handles):
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
        selector = await _first_unique_selector(
            page,
            [
                _scoped_attr_selector(_INPUT_QUERY, "id", input_id),
                _scoped_attr_selector(_INPUT_QUERY, "name", input_name),
                _scoped_attr_selector(_INPUT_QUERY, "aria-label", aria),
                _scoped_attr_selector(_INPUT_QUERY, "placeholder", placeholder),
            ],
            fallback=f"css={_INPUT_QUERY} >> nth={i}",
        )
        interactables.append(
            Interactable(
                kind="input",
                label=label,
                selector=selector,
                href=None,
            )
        )
        if len(interactables) >= limit:
            break
    return interactables


async def _build_button_interactables(page: Page, limit: int = 10) -> list[Interactable]:
    interactables: list[Interactable] = []
    handles = await page.query_selector_all(_BUTTON_QUERY)

    for i, handle in enumerate(handles):
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
        selector = await _first_unique_selector(
            page,
            [
                _scoped_attr_selector(_BUTTON_QUERY, "id", button_id),
                _scoped_attr_selector(_BUTTON_QUERY, "name", button_name),
                _scoped_attr_selector(_BUTTON_QUERY, "aria-label", aria),
                (
                    f'role=button[name="{_escape_role_value(label)}"]'
                    if label and not label.startswith("button_")
                    else None
                ),
            ],
            fallback=f"css={_BUTTON_QUERY} >> nth={i}",
        )
        interactables.append(
            Interactable(
                kind="button",
                label=label,
                selector=selector,
                href=None,
            )
        )
        if len(interactables) >= limit:
            break

    return interactables


async def _build_link_interactables(page: Page, limit: int = 15) -> list[Interactable]:
    interactables: list[Interactable] = []
    handles = await page.query_selector_all(_LINK_QUERY)
    for i, handle in enumerate(handles):
        if not await _is_visible(handle):
            continue
        try:
            href = await handle.get_attribute("href")
            text = await handle.inner_text()
        except Exception:
            continue

        label = _clean_label(text, fallback=f"link_{i}")
        selector = await _first_unique_selector(
            page,
            [
                _scoped_attr_selector(_LINK_QUERY, "href", href),
                (
                    f'role=link[name="{_escape_role_value(label)}"]'
                    if label and not label.startswith("link_")
                    else None
                ),
            ],
            fallback=f"css={_LINK_QUERY} >> nth={i}",
        )

        interactables.append(
            Interactable(
                kind="link",
                label=label,
                selector=selector,
                href=href,
            )
        )
        if len(interactables) >= limit:
            break

    return interactables


async def capture_state(page: Page) -> PageState:
    url = page.url
    try:
        title = await page.title()
    except Exception:
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=2000)
            title = await page.title()
        except Exception:
            title = url or "Untitled"

    interactables: list[Interactable] = []
    for builder in (_build_input_interactables, _build_button_interactables, _build_link_interactables):
        try:
            interactables.extend(await builder(page))
        except Exception:
            # During mid-navigation transitions, selector collection can fail.
            # Returning partial state is better than crashing the run.
            continue

    return PageState(url=url, title=title, markdown="", interactables=interactables)


async def run_browser(start_url: str, *, headless: bool = True) -> tuple[Playwright, Browser, Page]:
    normalized_start_url = _normalize_start_url(start_url)
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ],
    )
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
        viewport={"width": 1366, "height": 768},
    )
    await context.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
    page = await context.new_page()
    await page.add_init_script(
        """
        Object.defineProperty(navigator, "webdriver", { get: () => undefined });
        """
    )
    await page.goto(normalized_start_url, wait_until="domcontentloaded")
    return pw, browser, page
