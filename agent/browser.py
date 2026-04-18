from __future__ import annotations

from contextlib import suppress
import ipaddress
import re
from urllib.parse import urlsplit

from playwright.async_api import Browser, Page, Playwright, async_playwright

from agent.models import Interactable, PageState

_INPUT_QUERY = (
    "input:not([type]), input[type='text'], input[type='search'], input[type='email'], "
    "input[type='password'], input[type='url'], input[type='tel'], textarea"
)
_BUTTON_QUERY = "button, input[type='submit'], input[type='button']"
_SELECT_QUERY = "select"
_CHECKBOX_QUERY = "input[type='checkbox']"
_RADIO_QUERY = "input[type='radio']"
_OPTION_QUERY = "[role='option']"
_LINK_QUERY = "a[href]"
_SUPPORTED_URL_SCHEMES = {"http", "https"}
_HOST_PORT_SHORTHAND_PATTERN = re.compile(
    r"^(?P<host>\[[0-9A-Fa-f:]+\]|[A-Za-z0-9.-]+):(?P<port>\d+)(?:/.*)?$"
)
_HOST_LABEL_PATTERN = re.compile(r"[A-Za-z0-9-]+")
_MAX_HOSTNAME_LENGTH = 253
_MAX_HOST_LABEL_LENGTH = 63
_COMMIT_FALLBACK_TIMEOUT_MS = 12000
_COMMIT_READY_TIMEOUT_MS = 3000
_PRIORITY_LABEL_MIN_LENGTH = 3
_PRIORITY_LABEL_MAX_LENGTH = 60

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
        if len(matches) > 1:
            visible_matches = 0
            for handle in matches:
                try:
                    if await _is_visible(handle):
                        visible_matches += 1
                except Exception:
                    continue
                if visible_matches > 1:
                    break
            if visible_matches == 1:
                return candidate
    return fallback


def _normalize_start_url_scheme(start_url: str) -> str:
    if start_url.startswith("//"):
        return f"https:{start_url}"
    if "://" in start_url:
        scheme = urlsplit(start_url).scheme.lower()
        if scheme not in _SUPPORTED_URL_SCHEMES:
            raise ValueError("start_url_invalid_scheme")
        return start_url
    scheme_like = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", start_url)
    if scheme_like and not _HOST_PORT_SHORTHAND_PATTERN.match(start_url):
        raise ValueError("start_url_invalid_scheme")
    return f"https://{start_url}"


def _validate_hostname(hostname: str) -> None:
    if hostname == "localhost":
        return

    try:
        ipaddress.ip_address(hostname)
        return
    except ValueError:
        pass

    if len(hostname) > _MAX_HOSTNAME_LENGTH:
        raise ValueError("start_url_invalid_host")

    for label in hostname.split("."):
        if not label:
            raise ValueError("start_url_invalid_host")
        if len(label) > _MAX_HOST_LABEL_LENGTH:
            raise ValueError("start_url_invalid_host")
        if label.startswith("-") or label.endswith("-"):
            raise ValueError("start_url_invalid_host")
        if not _HOST_LABEL_PATTERN.fullmatch(label):
            raise ValueError("start_url_invalid_host")


def _normalize_start_url(start_url: str) -> str:
    normalized = start_url.strip()
    if not normalized:
        raise ValueError("start_url_empty")
    if any(char.isspace() for char in normalized):
        raise ValueError("start_url_contains_whitespace")
    normalized = _normalize_start_url_scheme(normalized)

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
    _validate_hostname(hostname)

    return normalized


def _clean_label(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    collapsed = re.sub(r"\s+", " ", value).strip()
    return collapsed[:80] if collapsed else fallback


async def _associated_label(handle) -> str | None:
    try:
        label = await handle.evaluate(
            """(el) => {
                const normalize = (value) => String(value || "").replace(/\\s+/g, " ").trim();
                const direct =
                    el.getAttribute("aria-label") ||
                    el.getAttribute("placeholder") ||
                    el.getAttribute("title");
                if (normalize(direct)) return normalize(direct);
                const id = el.getAttribute("id");
                if (id) {
                    const explicit = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                    if (explicit && normalize(explicit.textContent)) return normalize(explicit.textContent);
                }
                const wrapped = el.closest("label");
                if (wrapped && normalize(wrapped.textContent)) return normalize(wrapped.textContent);
                const ariaLabelledBy = el.getAttribute("aria-labelledby");
                if (ariaLabelledBy) {
                    const combined = ariaLabelledBy
                        .split(/\\s+/)
                        .map((labelId) => document.getElementById(labelId))
                        .filter(Boolean)
                        .map((node) => normalize(node.textContent))
                        .filter(Boolean)
                        .join(" ");
                    if (combined) return combined;
                }
                return "";
            }"""
        )
    except Exception:
        return None
    normalized = _clean_label(str(label), fallback="") if label is not None else ""
    return normalized or None


async def _select_options(handle, *, limit: int = 8) -> list[str] | None:
    try:
        options = await handle.evaluate(
            """(el, limit) => {
                if (!el || !el.options) return [];
                return Array.from(el.options)
                    .map((option) => {
                        const text = String(option.textContent || option.label || option.value || "")
                            .replace(/\\s+/g, " ")
                            .trim();
                        return text;
                    })
                    .filter(Boolean)
                    .slice(0, limit);
            }""",
            limit,
        )
    except Exception:
        return None

    if not isinstance(options, list):
        return None
    normalized = [
        _clean_label(str(option), fallback="")
        for option in options
        if str(option).strip()
    ]
    normalized = [option for option in normalized if option]
    return normalized or None


async def _checked_state(handle) -> bool | None:
    try:
        checked = await handle.evaluate(
            """(el) => {
                if (!el || !("checked" in el)) return null;
                return Boolean(el.checked);
            }"""
        )
    except Exception:
        return None

    if checked is None:
        return None
    return bool(checked)


async def goto_with_fallback(page: Page, url: str, *, timeout_ms: int = 30000) -> None:
    attempts = [
        ("domcontentloaded", min(timeout_ms, 15000)),
        ("commit", min(timeout_ms, _COMMIT_FALLBACK_TIMEOUT_MS)),
    ]
    last_error: Exception | None = None

    for wait_until, attempt_timeout in attempts:
        try:
            await page.goto(url, wait_until=wait_until, timeout=attempt_timeout)
            if wait_until == "commit":
                with suppress(Exception):
                    await page.wait_for_load_state("domcontentloaded", timeout=_COMMIT_READY_TIMEOUT_MS)
            return
        except Exception as exc:
            last_error = exc

    # Last resort: try navigating and just waiting for any response
    try:
        await page.goto(url, wait_until="commit", timeout=min(timeout_ms, 8000))
        return
    except Exception as exc:
        last_error = exc

    if last_error is not None:
        raise last_error


async def _is_visible(handle, *, require_enabled: bool = False) -> bool:
    try:
        return bool(await handle.evaluate(_VISIBLE_SCRIPT, require_enabled))
    except Exception:
        return False


async def _element_context(handle) -> tuple[str | None, str | None]:
    try:
        details = await handle.evaluate(
            """(el) => {
                const normalize = (value, maxChars = 160) => String(value || "")
                    .replace(/\\s+/g, " ")
                    .trim()
                    .slice(0, maxChars);

                let region = "body";
                if (el.closest("form")) {
                    region = "form";
                } else if (el.closest("main, [role='main']")) {
                    region = "main";
                } else if (el.closest("nav, [role='navigation']")) {
                    region = "nav";
                } else if (el.closest("header, [role='banner']")) {
                    region = "header";
                } else if (el.closest("footer, [role='contentinfo']")) {
                    region = "footer";
                } else if (el.closest("aside, [role='complementary']")) {
                    region = "aside";
                }

                const container = el.closest(
                    "li, article, section, form, main, [role='article'], [data-testid], [class*='card'], [class*='result'], [class*='product']"
                ) || el;
                const ownText = normalize(el.innerText || el.textContent || "");
                const containerText = normalize(container.innerText || container.textContent || "");
                const contextText = containerText && containerText !== ownText ? containerText : ownText;
                return { region, contextText };
            }"""
        )
    except Exception:
        return None, None

    if not isinstance(details, dict):
        return None, None

    region = details.get("region")
    context_text = details.get("contextText")
    normalized_region = str(region).strip() if isinstance(region, str) and region else None
    normalized_context = (
        _clean_label(str(context_text), fallback="")
        if isinstance(context_text, str) and context_text.strip()
        else None
    )
    return normalized_region, normalized_context or None


def _interactable_priority(
    *,
    kind: str,
    label: str,
    href: str | None,
    region: str | None,
    context_text: str | None,
) -> int:
    region_scores = {
        "form": 90,
        "main": 80,
        "body": 65,
        "aside": 45,
        "header": 20,
        "nav": 10,
        "footer": 0,
    }
    score = region_scores.get(region or "body", 50)

    if kind in {"input", "select", "checkbox", "radio"} and region == "form":
        score += 12
    if kind == "button" and region in {"form", "main"}:
        score += 8
    if kind == "link":
        if href and not href.startswith(("#", "javascript:")):
            score += 8
        elif href:
            score -= 12

    normalized = f"{label} {context_text or ''}".lower()
    if any(
        token in normalized
        for token in (
            "price",
            "buy",
            "shop",
            "cart",
            "result",
            "search",
            "quote",
            "flight",
            "depart",
            "return",
            "hotel",
        )
    ):
        score += 6

    if _PRIORITY_LABEL_MIN_LENGTH <= len(label) <= _PRIORITY_LABEL_MAX_LENGTH:
        score += 2
    return score


def _sort_interactables(interactables: list[Interactable]) -> list[Interactable]:
    deduped: dict[str, Interactable] = {}
    for item in interactables:
        existing = deduped.get(item.selector)
        if existing is None:
            deduped[item.selector] = item
            continue

        new_score = _interactable_priority(
            kind=item.kind,
            label=item.label,
            href=item.href,
            region=item.region,
            context_text=item.context_text,
        )
        existing_score = _interactable_priority(
            kind=existing.kind,
            label=existing.label,
            href=existing.href,
            region=existing.region,
            context_text=existing.context_text,
        )
        if new_score > existing_score:
            deduped[item.selector] = item

    return sorted(
        deduped.values(),
        key=lambda item: (
            -_interactable_priority(
                kind=item.kind,
                label=item.label,
                href=item.href,
                region=item.region,
                context_text=item.context_text,
            ),
            item.label.lower(),
        ),
    )


def _assign_interactable_refs(interactables: list[Interactable]) -> list[Interactable]:
    for index, item in enumerate(interactables, start=1):
        item.ref = f"el{index}"
    return interactables


async def _build_input_interactables(page: Page, limit: int = 10) -> list[Interactable]:
    candidates: list[tuple[int, Interactable]] = []
    handles = await page.query_selector_all(_INPUT_QUERY)
    for i, handle in enumerate(handles):
        if not await _is_visible(handle, require_enabled=True):
            continue
        try:
            input_type = await handle.get_attribute("type")
            input_name = await handle.get_attribute("name")
            input_id = await handle.get_attribute("id")
            aria = await handle.get_attribute("aria-label")
            placeholder = await handle.get_attribute("placeholder")
        except Exception:
            continue

        label = _clean_label(
            await _associated_label(handle) or aria or placeholder or input_name or input_id,
            fallback=f"input_{i}",
        )
        region, context_text = await _element_context(handle)
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
        interactable = Interactable(
            kind="input",
            label=label,
            selector=selector,
            href=None,
            field_type=_clean_label(input_type or "text", fallback="text"),
            region=region,  # type: ignore[arg-type]
            context_text=context_text,
        )
        candidates.append(
            (
                _interactable_priority(
                    kind="input",
                    label=label,
                    href=None,
                    region=region,
                    context_text=context_text,
                ),
                interactable,
            )
        )
    return [item for _, item in sorted(candidates, key=lambda pair: (-pair[0], pair[1].label.lower()))[:limit]]


async def _build_button_interactables(page: Page, limit: int = 10) -> list[Interactable]:
    candidates: list[tuple[int, Interactable]] = []
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
        region, context_text = await _element_context(handle)
        selector = await _first_unique_selector(
            page,
            [
                _scoped_attr_selector(_BUTTON_QUERY, "id", button_id),
                _scoped_attr_selector(_BUTTON_QUERY, "name", button_name),
                _scoped_attr_selector(_BUTTON_QUERY, "aria-label", aria),
                (
                    f'role=button[name="{_escape_css_value(label)}"]'
                    if label and not label.startswith("button_")
                    else None
                ),
            ],
            fallback=f"css={_BUTTON_QUERY} >> nth={i}",
        )
        interactable = Interactable(
            kind="button",
            label=label,
            selector=selector,
            href=None,
            field_type="button",
            region=region,  # type: ignore[arg-type]
            context_text=context_text,
        )
        candidates.append(
            (
                _interactable_priority(
                    kind="button",
                    label=label,
                    href=None,
                    region=region,
                    context_text=context_text,
                ),
                interactable,
            )
        )

    return [item for _, item in sorted(candidates, key=lambda pair: (-pair[0], pair[1].label.lower()))[:limit]]


async def _build_select_interactables(page: Page, limit: int = 10) -> list[Interactable]:
    candidates: list[tuple[int, Interactable]] = []
    handles = await page.query_selector_all(_SELECT_QUERY)
    for i, handle in enumerate(handles):
        if not await _is_visible(handle, require_enabled=True):
            continue
        try:
            select_name = await handle.get_attribute("name")
            select_id = await handle.get_attribute("id")
            aria = await handle.get_attribute("aria-label")
        except Exception:
            continue

        label = _clean_label(
            await _associated_label(handle) or aria or select_name or select_id,
            fallback=f"select_{i}",
        )
        region, context_text = await _element_context(handle)
        selector = await _first_unique_selector(
            page,
            [
                _scoped_attr_selector(_SELECT_QUERY, "id", select_id),
                _scoped_attr_selector(_SELECT_QUERY, "name", select_name),
                _scoped_attr_selector(_SELECT_QUERY, "aria-label", aria),
                (
                    f'role=combobox[name="{_escape_css_value(label)}"]'
                    if label and not label.startswith("select_")
                    else None
                ),
            ],
            fallback=f"css={_SELECT_QUERY} >> nth={i}",
        )
        interactable = Interactable(
            kind="select",
            label=label,
            selector=selector,
            href=None,
            field_type="select",
            options=await _select_options(handle),
            region=region,  # type: ignore[arg-type]
            context_text=context_text,
        )
        candidates.append(
            (
                _interactable_priority(
                    kind="select",
                    label=label,
                    href=None,
                    region=region,
                    context_text=context_text,
                ),
                interactable,
            )
        )
    return [item for _, item in sorted(candidates, key=lambda pair: (-pair[0], pair[1].label.lower()))[:limit]]


async def _build_checkable_interactables(
    page: Page,
    *,
    scope_query: str,
    kind: str,
    limit: int = 10,
) -> list[Interactable]:
    candidates: list[tuple[int, Interactable]] = []
    handles = await page.query_selector_all(scope_query)
    for i, handle in enumerate(handles):
        if not await _is_visible(handle, require_enabled=True):
            continue
        try:
            name = await handle.get_attribute("name")
            input_id = await handle.get_attribute("id")
            aria = await handle.get_attribute("aria-label")
            value = await handle.get_attribute("value")
        except Exception:
            continue

        label = _clean_label(
            await _associated_label(handle) or aria or value or name or input_id,
            fallback=f"{kind}_{i}",
        )
        region, context_text = await _element_context(handle)
        selector = await _first_unique_selector(
            page,
            [
                _scoped_attr_selector(scope_query, "id", input_id),
                _scoped_attr_selector(scope_query, "name", name),
                _scoped_attr_selector(scope_query, "aria-label", aria),
                (
                    f'role={kind}[name="{_escape_css_value(label)}"]'
                    if label and not label.startswith(f"{kind}_")
                    else None
                ),
            ],
            fallback=f"css={scope_query} >> nth={i}",
        )
        interactable = Interactable(
            kind=kind,  # type: ignore[arg-type]
            label=label,
            selector=selector,
            href=None,
            field_type=kind,
            checked=await _checked_state(handle),
            region=region,  # type: ignore[arg-type]
            context_text=context_text,
        )
        candidates.append(
            (
                _interactable_priority(
                    kind=kind,
                    label=label,
                    href=None,
                    region=region,
                    context_text=context_text,
                ),
                interactable,
            )
        )
    return [item for _, item in sorted(candidates, key=lambda pair: (-pair[0], pair[1].label.lower()))[:limit]]


async def _build_link_interactables(page: Page, limit: int = 40) -> list[Interactable]:
    candidates: list[tuple[int, Interactable]] = []
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
        region, context_text = await _element_context(handle)
        selector = await _first_unique_selector(
            page,
            [
                _scoped_attr_selector(_LINK_QUERY, "href", href),
                (
                    f'role=link[name="{_escape_css_value(label)}"]'
                    if label and not label.startswith("link_")
                    else None
                ),
            ],
            fallback=f"css={_LINK_QUERY} >> nth={i}",
        )
        interactable = Interactable(
            kind="link",
            label=label,
            selector=selector,
            href=href,
            field_type="link",
            region=region,  # type: ignore[arg-type]
            context_text=context_text,
        )
        candidates.append(
            (
                _interactable_priority(
                    kind="link",
                    label=label,
                    href=href,
                    region=region,
                    context_text=context_text,
                ),
                interactable,
            )
        )

    return [item for _, item in sorted(candidates, key=lambda pair: (-pair[0], pair[1].label.lower()))[:limit]]


async def _build_option_interactables(page: Page, limit: int = 20) -> list[Interactable]:
    candidates: list[tuple[int, Interactable]] = []
    handles = await page.query_selector_all(_OPTION_QUERY)
    for i, handle in enumerate(handles):
        if not await _is_visible(handle):
            continue
        try:
            details = await handle.evaluate(
                """(el) => {
                    const normalize = (value) => String(value || "").replace(/\\s+/g, " ").trim();
                    const anchor = el.matches("a[href]") ? el : el.querySelector("a[href]");
                    return {
                        label: normalize(
                            el.getAttribute("aria-label") ||
                            (anchor ? anchor.innerText || anchor.textContent : "") ||
                            el.innerText ||
                            el.textContent
                        ),
                        href: anchor ? anchor.getAttribute("href") : null,
                    };
                }"""
            )
        except Exception:
            continue

        if not isinstance(details, dict):
            continue
        label = _clean_label(str(details.get("label") or ""), fallback=f"option_{i}")
        href = str(details.get("href") or "").strip() or None
        region, context_text = await _element_context(handle)
        selector = await _first_unique_selector(
            page,
            [
                (
                    f'role=option[name="{_escape_css_value(label)}"]'
                    if label and not label.startswith("option_")
                    else None
                ),
                (f'css={_OPTION_QUERY} a[href="{_escape_css_value(href)}"]' if href else None),
            ],
            fallback=f"css={_OPTION_QUERY} >> nth={i}",
        )
        interactable = Interactable(
            kind="link" if href else "button",
            label=label,
            selector=selector,
            href=href,
            field_type="option",
            region=region,  # type: ignore[arg-type]
            context_text=context_text,
        )
        candidates.append(
            (
                _interactable_priority(
                    kind=interactable.kind,
                    label=label,
                    href=href,
                    region=region,
                    context_text=context_text,
                ),
                interactable,
            )
        )

    return [item for _, item in sorted(candidates, key=lambda pair: (-pair[0], pair[1].label.lower()))[:limit]]


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
    for builder in (
        _build_input_interactables,
        _build_button_interactables,
        _build_option_interactables,
        _build_select_interactables,
        lambda current_page: _build_checkable_interactables(
            current_page,
            scope_query=_CHECKBOX_QUERY,
            kind="checkbox",
        ),
        lambda current_page: _build_checkable_interactables(
            current_page,
            scope_query=_RADIO_QUERY,
            kind="radio",
        ),
        _build_link_interactables,
    ):
        try:
            interactables.extend(await builder(page))
        except Exception:
            # During mid-navigation transitions, selector collection can fail.
            # Returning partial state is better than crashing the run.
            continue

    return PageState(
        url=url,
        title=title,
        markdown="",
        interactables=_assign_interactable_refs(_sort_interactables(interactables)),
    )


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
    await goto_with_fallback(page, normalized_start_url)
    return pw, browser, page
