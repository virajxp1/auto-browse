"""Low-level browser interaction helpers and selector utilities.

Extracted from run.py to reduce module size. All functions operate on a
Playwright Page instance and handle timeout/fallback logic for clicks,
typing, form submission, and selector resolution.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import suppress



from playwright.async_api import Page

from agent.snapshot import capture_dom_signature

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIMARY_SELECTOR_WAIT_MS = 1500
FALLBACK_SELECTOR_WAIT_MS = 400
_MAX_TYPE_FALLBACK_SELECTORS = 4
_DIRECT_FILL_TIMEOUT_SECONDS = 1.2
_FILL_OVERALL_TIMEOUT_SECONDS = 3.0
_TYPE_AND_SUBMIT_EFFECT_RETRIES = 4
_TYPE_AND_SUBMIT_EFFECT_DELAY_MS = 250

# ---------------------------------------------------------------------------
# Selector parsing & hint extraction
# ---------------------------------------------------------------------------


def _normalize_hint_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _unquote_selector_value(raw: str) -> str:
    stripped = raw.strip()
    if len(stripped) >= 2 and (
        (stripped.startswith('"') and stripped.endswith('"'))
        or (stripped.startswith("'") and stripped.endswith("'"))
    ):
        stripped = stripped[1:-1]
    return stripped.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")


def extract_selector_hint(selector: str) -> str | None:
    normalized_selector = selector.strip()
    if not normalized_selector:
        return None

    if normalized_selector.startswith("text="):
        return _normalize_hint_text(_unquote_selector_value(normalized_selector[len("text=") :]))

    role_name_match = re.search(
        r"""name\s*=\s*(?:"([^"]+)"|'([^']+)')""",
        normalized_selector,
    )
    if role_name_match:
        return _normalize_hint_text(role_name_match.group(1) or role_name_match.group(2))

    for attr in ("aria-label", "placeholder", "name", "id", "title", "alt", "value"):
        attr_match = re.search(
            rf"""\[{re.escape(attr)}\s*=\s*(?:"([^"]+)"|'([^']+)')\]""",
            normalized_selector,
        )
        if attr_match:
            return _normalize_hint_text(attr_match.group(1) or attr_match.group(2))

    return None


def _escape_selector_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def parse_css_selector(selector: str) -> tuple[str, int | None] | None:
    if not selector.startswith("css="):
        return None
    css_selector = selector[len("css=") :].strip()
    if not css_selector:
        return None

    base_selector = css_selector
    nth_index: int | None = None
    if " >> nth=" in css_selector:
        base_selector, nth_part = css_selector.rsplit(" >> nth=", 1)
        base_selector = base_selector.strip()
        if not base_selector:
            return None
        try:
            nth_index = int(nth_part.strip())
        except ValueError:
            return None
        if nth_index < 0:
            return None

    return (base_selector, nth_index)


def _dedupe_selectors(candidates: list[str], *, exclude: str | None = None) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized:
            continue
        if exclude is not None and normalized == exclude:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


# ---------------------------------------------------------------------------
# Selector fallback generators
# ---------------------------------------------------------------------------


def click_fallback_selectors(selector: str) -> list[str]:
    fallback_selectors: list[str] = []
    parsed_css_selector = parse_css_selector(selector)
    if parsed_css_selector is not None:
        base_selector, _ = parsed_css_selector
        fallback_selectors.append(f"css={base_selector}")

    hint = extract_selector_hint(selector)
    if hint:
        escaped_hint = _escape_selector_value(hint)
        fallback_selectors.extend(
            [
                f'role=link[name="{escaped_hint}"]',
                f'role=button[name="{escaped_hint}"]',
                f'text="{escaped_hint}"',
            ]
        )

    return _dedupe_selectors(fallback_selectors, exclude=selector)


def type_fallback_selectors(selector: str) -> list[str]:
    fallback_selectors: list[str] = []
    parsed_css_selector = parse_css_selector(selector)
    if parsed_css_selector is not None:
        base_selector, _ = parsed_css_selector
        fallback_selectors.append(f"css={base_selector}")

    hint = extract_selector_hint(selector)
    if hint:
        escaped_hint = _escape_selector_value(hint)
        fallback_selectors.extend(
            [
                f'role=textbox[name="{escaped_hint}"]',
                f'css=input[aria-label="{escaped_hint}"]',
                f'css=input[placeholder="{escaped_hint}"]',
                f'css=textarea[aria-label="{escaped_hint}"]',
                f'css=input[name="{escaped_hint}"]',
            ]
        )

    fallback_selectors.extend(
        [
            "css=input[type='search']",
            "css=input[type='text']",
            "css=textarea",
        ]
    )

    return _dedupe_selectors(fallback_selectors, exclude=selector)[:_MAX_TYPE_FALLBACK_SELECTORS]


def select_fallback_selectors(selector: str) -> list[str]:
    fallback_selectors: list[str] = []
    hint = extract_selector_hint(selector)
    if hint:
        escaped_hint = _escape_selector_value(hint)
        fallback_selectors.extend(
            [
                f'role=combobox[name="{escaped_hint}"]',
                f'css=select[aria-label="{escaped_hint}"]',
                f'css=select[name="{escaped_hint}"]',
            ]
        )

    fallback_selectors.append("css=select")

    parsed_css_selector = parse_css_selector(selector)
    if parsed_css_selector is not None:
        base_selector, _ = parsed_css_selector
        fallback_selectors.append(f"css={base_selector}")

    return _dedupe_selectors(fallback_selectors, exclude=selector)


def check_fallback_selectors(selector: str) -> list[str]:
    fallback_selectors: list[str] = []
    hint = extract_selector_hint(selector)
    if hint:
        escaped_hint = _escape_selector_value(hint)
        fallback_selectors.extend(
            [
                f'role=checkbox[name="{escaped_hint}"]',
                f'role=radio[name="{escaped_hint}"]',
                f'css=input[aria-label="{escaped_hint}"]',
                f'css=input[name="{escaped_hint}"]',
            ]
        )

    fallback_selectors.extend(
        [
            "css=input[type='checkbox']",
            "css=input[type='radio']",
        ]
    )

    parsed_css_selector = parse_css_selector(selector)
    if parsed_css_selector is not None:
        base_selector, _ = parsed_css_selector
        fallback_selectors.append(f"css={base_selector}")

    return _dedupe_selectors(fallback_selectors, exclude=selector)


# ---------------------------------------------------------------------------
# Low-level async browser helpers
# ---------------------------------------------------------------------------


async def wait_short(page: Page, timeout_ms: int) -> None:
    wait_fn = getattr(page, "wait_for_timeout", None)
    if not callable(wait_fn):
        return
    with suppress(Exception):
        await wait_fn(timeout_ms)


async def wait_for_selector_visible(page: Page, selector: str, *, timeout_ms: int = 3500) -> bool:
    wait_for_selector_fn = getattr(page, "wait_for_selector", None)
    if not callable(wait_for_selector_fn):
        return True
    try:
        await wait_for_selector_fn(selector, state="visible", timeout=timeout_ms)
        return True
    except TypeError:
        try:
            await wait_for_selector_fn(selector, timeout=timeout_ms)
            return True
        except Exception:
            return False
    except Exception:
        return False


async def wait_for_action_effect(
    page: Page,
    before_snapshot: tuple[str, str, str],
    *,
    retries: int = 3,
    delay_ms: int = 300,
) -> bool:
    wait_timeout_fn = getattr(page, "wait_for_timeout", None)
    if not callable(wait_timeout_fn):
        return False

    for _ in range(retries):
        try:
            await wait_timeout_fn(delay_ms)
        except Exception:
            return False

        after_snapshot = await capture_dom_signature(page)
        if before_snapshot != after_snapshot:
            return True

    return False


async def try_click_selector(page: Page, selector: str) -> bool:
    if not await wait_for_selector_visible(page, selector):
        return False

    click_fn = getattr(page, "click", None)
    if not callable(click_fn):
        return False
    try:
        await click_fn(selector, timeout=5000)
        return True
    except TypeError:
        try:
            await click_fn(selector)
            return True
        except Exception:
            return False
    except Exception:
        return False


async def find_nearby_submit_button(page: Page, input_selector: str) -> str | None:
    """Find a submit/search button near the input field."""
    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return None
    try:
        result = await evaluate_fn(
            """(inputSelector) => {
                const input = document.querySelector(inputSelector);
                if (!input) return null;
                const form = input.closest("form");
                const container = form || input.parentElement?.parentElement?.parentElement || document.body;
                const candidates = container.querySelectorAll(
                    'button[type="submit"], input[type="submit"], button[aria-label*="search" i], button[aria-label*="Search" i], button[title*="search" i]'
                );
                for (const btn of candidates) {
                    if (btn.offsetWidth > 0 && btn.offsetHeight > 0) {
                        if (btn.id) return "#" + btn.id;
                        if (btn.name) return btn.tagName.toLowerCase() + '[name="' + btn.name + '"]';
                        if (btn.getAttribute("aria-label")) return btn.tagName.toLowerCase() + '[aria-label="' + btn.getAttribute("aria-label") + '"]';
                        return null;
                    }
                }
                return null;
            }""",
            input_selector,
        )
        return str(result) if result else None
    except Exception:
        return None


async def try_type_and_submit_selector(
    page: Page,
    selector: str,
    text: str,
    *,
    wait_timeout_ms: int = PRIMARY_SELECTOR_WAIT_MS,
) -> bool:
    if not await wait_for_selector_visible(page, selector, timeout_ms=wait_timeout_ms):
        return False
    before_snapshot = await capture_dom_signature(page)
    typed = await try_fill_text_selector(page, selector, text)
    if not typed:
        typed = await try_type_text_selector(page, selector, text)
    if not typed:
        return await try_submit_selector(page, selector)

    if await wait_for_action_effect(
        page,
        before_snapshot,
        retries=_TYPE_AND_SUBMIT_EFFECT_RETRIES,
        delay_ms=_TYPE_AND_SUBMIT_EFFECT_DELAY_MS,
    ):
        return True

    after_typing_snapshot = await capture_dom_signature(page)
    if before_snapshot != after_typing_snapshot:
        await wait_short(page, _TYPE_AND_SUBMIT_EFFECT_DELAY_MS)
        return True

    # Try Enter key submit
    if await try_submit_selector(page, selector):
        return True

    # Fallback: find and click a nearby submit/search button
    submit_selector = await find_nearby_submit_button(page, selector)
    if submit_selector:
        if await try_click_selector(page, submit_selector):
            if await wait_for_action_effect(page, before_snapshot, retries=3, delay_ms=300):
                return True
    return False


async def try_fill_selector(
    page: Page,
    selector: str,
    text: str,
    *,
    wait_timeout_ms: int = PRIMARY_SELECTOR_WAIT_MS,
) -> bool:
    if not await wait_for_selector_visible(page, selector, timeout_ms=wait_timeout_ms):
        return False
    if await try_fill_text_selector(page, selector, text):
        return True
    return await try_type_text_selector(page, selector, text)


async def try_fill_text_selector(page: Page, selector: str, text: str) -> bool:
    fill_fn = getattr(page, "fill", None)
    if not callable(fill_fn):
        return False

    async def _do_fill() -> bool:
        try:
            try:
                await fill_fn(selector, text, timeout=int(_DIRECT_FILL_TIMEOUT_SECONDS * 1000))
            except TypeError:
                await asyncio.wait_for(fill_fn(selector, text), timeout=_DIRECT_FILL_TIMEOUT_SECONDS)
            return True
        except Exception:
            return False

    # Outer timeout prevents fill from stalling the entire run (e.g. Best Buy)
    try:
        return await asyncio.wait_for(_do_fill(), timeout=_FILL_OVERALL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return False


async def try_type_text_selector(page: Page, selector: str, text: str) -> bool:
    type_fn = getattr(page, "type", None)
    if callable(type_fn):
        try:
            try:
                await type_fn(selector, text, timeout=int(_DIRECT_FILL_TIMEOUT_SECONDS * 1000))
            except TypeError:
                await asyncio.wait_for(type_fn(selector, text), timeout=_DIRECT_FILL_TIMEOUT_SECONDS)
            return True
        except Exception:
            pass

    click_fn = getattr(page, "click", None)
    focus_fn = getattr(page, "focus", None)
    keyboard = getattr(page, "keyboard", None)
    keyboard_press_fn = getattr(keyboard, "press", None)
    keyboard_type_fn = getattr(keyboard, "type", None)
    if not callable(keyboard_type_fn):
        return False

    try:
        if callable(click_fn):
            try:
                await click_fn(selector, timeout=1500)
            except TypeError:
                await click_fn(selector)
        elif callable(focus_fn):
            await focus_fn(selector)
        else:
            return False

        if callable(keyboard_press_fn):
            for combo in ("Meta+A", "Control+A"):
                try:
                    await keyboard_press_fn(combo)
                    break
                except Exception:
                    continue
            try:
                await keyboard_press_fn("Backspace")
            except Exception:
                pass

        await asyncio.wait_for(keyboard_type_fn(text), timeout=max(_DIRECT_FILL_TIMEOUT_SECONDS, 0.5))
        return True
    except Exception:
        return False


async def selector_value_matches(page: Page, selector: str, expected_text: str) -> bool:
    input_value_fn = getattr(page, "input_value", None)
    if callable(input_value_fn):
        try:
            actual_value = await input_value_fn(selector)
            return str(actual_value or "") == expected_text
        except Exception:
            pass

    evaluate_fn = getattr(page, "eval_on_selector", None)
    if callable(evaluate_fn):
        try:
            actual_value = await evaluate_fn(
                selector,
                "(el) => ('value' in el ? String(el.value || '') : '')",
            )
            return str(actual_value or "") == expected_text
        except Exception:
            return False

    return False


async def try_submit_selector(page: Page, selector: str) -> bool:
    if not await wait_for_selector_visible(page, selector):
        return False

    before_snapshot = await capture_dom_signature(page)
    press_fn = getattr(page, "press", None)
    if callable(press_fn):
        try:
            await press_fn(selector, "Enter")
            if before_snapshot != await capture_dom_signature(page):
                return True
            if await wait_for_action_effect(page, before_snapshot, retries=2, delay_ms=200):
                return True
        except Exception:
            pass

    keyboard = getattr(page, "keyboard", None)
    keyboard_press_fn = getattr(keyboard, "press", None)
    if callable(keyboard_press_fn):
        try:
            await keyboard_press_fn("Enter")
            if before_snapshot != await capture_dom_signature(page):
                return True
            if await wait_for_action_effect(page, before_snapshot, retries=2, delay_ms=200):
                return True
        except Exception:
            pass

    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return False

    try:
        submitted = await evaluate_fn(
            """(selector) => {
                const el = document.querySelector(selector);
                if (!el) return false;
                if (el.tagName && String(el.tagName).toLowerCase() === "form") {
                    if (typeof el.requestSubmit === "function") {
                        el.requestSubmit();
                        return true;
                    }
                    el.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
                    return true;
                }
                if (el.form && typeof el.form.requestSubmit === "function") {
                    el.form.requestSubmit();
                    return true;
                }
                if (el.form) {
                    el.form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
                    return true;
                }
                return false;
            }""",
            selector,
        )
    except Exception:
        return False
    return bool(submitted)


async def try_select_option_selector(page: Page, selector: str, value: str) -> bool:
    if not await wait_for_selector_visible(page, selector):
        return False

    select_option_fn = getattr(page, "select_option", None)
    if not callable(select_option_fn):
        return False

    try:
        await select_option_fn(selector, value=value)
        return True
    except TypeError:
        try:
            await select_option_fn(selector, value)
            return True
        except Exception:
            return False
    except Exception:
        return False


async def selector_has_selected_value(page: Page, selector: str, expected_value: str) -> bool:
    evaluate_fn = getattr(page, "eval_on_selector", None)
    if not callable(evaluate_fn):
        return False

    try:
        actual_value = await evaluate_fn(
            selector,
            "(el) => ('value' in el ? String(el.value || '') : '')",
        )
    except Exception:
        return False
    return str(actual_value or "") == expected_value


async def try_check_selector(page: Page, selector: str) -> bool:
    if not await wait_for_selector_visible(page, selector):
        return False

    check_fn = getattr(page, "check", None)
    if callable(check_fn):
        try:
            await check_fn(selector)
            return True
        except Exception:
            pass

    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return False

    try:
        checked = await evaluate_fn(
            """(selector) => {
                const el = document.querySelector(selector);
                if (!el) return false;
                if (!("checked" in el)) return false;
                el.checked = true;
                el.dispatchEvent(new Event("input", { bubbles: true }));
                el.dispatchEvent(new Event("change", { bubbles: true }));
                return Boolean(el.checked);
            }""",
            selector,
        )
    except Exception:
        return False
    return bool(checked)


async def selector_is_checked(page: Page, selector: str) -> bool:
    is_checked_fn = getattr(page, "is_checked", None)
    if callable(is_checked_fn):
        try:
            return bool(await is_checked_fn(selector))
        except Exception:
            pass

    evaluate_fn = getattr(page, "eval_on_selector", None)
    if not callable(evaluate_fn):
        return False

    try:
        checked = await evaluate_fn(
            selector,
            "(el) => ('checked' in el ? Boolean(el.checked) : false)",
        )
    except Exception:
        return False
    return bool(checked)


async def try_wait_for_selector(
    page: Page,
    selector: str,
    *,
    state: str,
    timeout_ms: int,
) -> bool:
    wait_for_selector_fn = getattr(page, "wait_for_selector", None)
    if not callable(wait_for_selector_fn):
        return False
    try:
        await wait_for_selector_fn(selector, state=state, timeout=timeout_ms)
        return True
    except TypeError:
        try:
            await wait_for_selector_fn(selector, timeout=timeout_ms)
            return True
        except Exception:
            return False
    except Exception:
        return False


async def click_via_css_fallback(page: Page, selector: str) -> bool:
    parsed = parse_css_selector(selector)
    if parsed is None:
        return False
    base_selector, nth_index = parsed

    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return False

    try:
        clicked = await evaluate_fn(
            """(payload) => {
                const { baseSelector, nthIndex } = payload;
                let el = null;
                if (typeof nthIndex === "number" && Number.isInteger(nthIndex) && nthIndex >= 0) {
                    const nodes = document.querySelectorAll(baseSelector);
                    el = nodes.length > nthIndex ? nodes[nthIndex] : null;
                } else {
                    el = document.querySelector(baseSelector);
                }
                if (!el) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                const isDisabled = Boolean(el.disabled);
                const isVisible =
                  style.display !== "none" &&
                  style.visibility !== "hidden" &&
                  rect.width > 0 &&
                  rect.height > 0 &&
                  !isDisabled;
                if (!isVisible) return false;
                el.click();
                return true;
            }""",
            {"baseSelector": base_selector, "nthIndex": nth_index},
        )
    except Exception:
        return False

    if clicked is True:
        await wait_short(page, 300)
        return True
    return False


async def click_via_text_heuristic(page: Page, hint: str | None) -> bool:
    if not hint:
        return False
    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return False

    try:
        clicked = await evaluate_fn(
            """(payload) => {
                const hint = String(payload?.hint || "").trim().toLowerCase();
                if (!hint) return false;
                const normalize = (value) => String(value || "").replace(/\\s+/g, " ").trim().toLowerCase();
                const isVisible = (el) => {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    const disabled = Boolean(el.disabled);
                    return (
                        style.display !== "none" &&
                        style.visibility !== "hidden" &&
                        rect.width > 0 &&
                        rect.height > 0 &&
                        !disabled
                    );
                };
                const candidates = Array.from(
                    document.querySelectorAll(
                        "a[href],button,input[type='submit'],input[type='button'],[role='button'],[role='link']"
                    )
                );
                let best = null;
                let bestScore = 0;
                for (const el of candidates) {
                    if (!isVisible(el)) continue;
                    const label = normalize(
                        el.getAttribute("aria-label") ||
                        el.innerText ||
                        el.textContent ||
                        el.value ||
                        el.getAttribute("title")
                    );
                    if (!label) continue;
                    let score = 0;
                    if (label === hint) score = 3;
                    else if (label.includes(hint)) score = 2;
                    else if (hint.includes(label) && label.length >= 4) score = 1;
                    if (score > bestScore) {
                        best = el;
                        bestScore = score;
                        if (score === 3) break;
                    }
                }
                if (!best) return false;
                best.click();
                return true;
            }""",
            {"hint": hint},
        )
    except Exception:
        return False

    if clicked is True:
        await wait_short(page, 300)
        return True
    return False


async def click_single_visible_link(page: Page) -> bool:
    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return False

    try:
        clicked = await evaluate_fn(
            """() => {
                // singleVisibleLinkFallback
                const isVisible = (el) => {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return (
                        style.display !== "none" &&
                        style.visibility !== "hidden" &&
                        rect.width > 0 &&
                        rect.height > 0
                    );
                };
                const links = Array.from(document.querySelectorAll("a[href]")).filter(isVisible);
                if (links.length !== 1) return false;
                links[0].click();
                return true;
            }"""
        )
    except Exception:
        return False

    if clicked is True:
        await wait_short(page, 300)
        return True
    return False


async def type_and_submit_via_text_heuristic(page: Page, text: str, hint: str | None) -> bool:
    evaluate_fn = getattr(page, "evaluate", None)
    if not callable(evaluate_fn):
        return False

    try:
        submitted = await evaluate_fn(
            """(payload) => {
                const text = String(payload?.text || "");
                if (!text) return false;
                const hint = String(payload?.hint || "").trim().toLowerCase();
                const normalize = (value) => String(value || "").replace(/\\s+/g, " ").trim().toLowerCase();
                const editableInput = (el) => {
                    if (!el) return false;
                    const tag = String(el.tagName || "").toLowerCase();
                    if (tag === "textarea") return true;
                    if (tag !== "input") return false;
                    const t = String(el.getAttribute("type") || "text").toLowerCase();
                    return ["", "text", "search", "email", "url", "tel", "password"].includes(t);
                };
                const isVisibleEnabled = (el) => {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return (
                        style.display !== "none" &&
                        style.visibility !== "hidden" &&
                        rect.width > 0 &&
                        rect.height > 0 &&
                        !el.disabled
                    );
                };
                const associatedLabel = (el) => {
                    if (!el) return "";
                    const direct = el.getAttribute("aria-label") || el.getAttribute("placeholder");
                    if (direct) return direct;
                    const id = el.getAttribute("id");
                    if (id) {
                        const label = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                        if (label && label.textContent) return label.textContent;
                    }
                    const wrapped = el.closest("label");
                    if (wrapped && wrapped.textContent) return wrapped.textContent;
                    return el.getAttribute("name") || el.getAttribute("id") || "";
                };

                const candidates = Array.from(document.querySelectorAll("input,textarea"));
                let best = null;
                let bestScore = -1;
                for (const el of candidates) {
                    if (!editableInput(el) || !isVisibleEnabled(el)) continue;
                    const label = normalize(associatedLabel(el));
                    let score = 0;
                    if (hint) {
                        if (label === hint) score = 4;
                        else if (label.includes(hint)) score = 3;
                        else if (hint.includes(label) && label.length >= 3) score = 2;
                    } else {
                        score = 1;
                    }
                    if (score > bestScore) {
                        best = el;
                        bestScore = score;
                        if (score === 4) break;
                    }
                }
                if (!best) return false;

                best.focus();
                best.value = text;
                best.dispatchEvent(new Event("input", { bubbles: true }));
                best.dispatchEvent(new Event("change", { bubbles: true }));

                if (best.form && typeof best.form.requestSubmit === "function") {
                    best.form.requestSubmit();
                    return true;
                }
                if (best.form) {
                    best.form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
                }
                best.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", bubbles: true }));
                best.dispatchEvent(new KeyboardEvent("keypress", { key: "Enter", code: "Enter", bubbles: true }));
                best.dispatchEvent(new KeyboardEvent("keyup", { key: "Enter", code: "Enter", bubbles: true }));
                return true;
            }""",
            {"text": text, "hint": hint},
        )
    except Exception:
        return False

    if submitted:
        await wait_short(page, 300)
    return bool(submitted)
