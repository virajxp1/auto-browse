from __future__ import annotations

import asyncio
import logging

try:
    from rebrowser_playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright
except ImportError:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 30_000
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# HTML signals present in bot-challenge pages (DataDome, Akamai, Cloudflare, etc.)
# Checked after the first navigation — if detected, session restarts with camoufox.
_BOT_CHALLENGE_SIGNALS = (
    "datadome",       # DataDome JS tag / cookie
    "dd_token",       # DataDome token field
    "__ddg",          # DataDome globals
    "cf-chl-",        # Cloudflare challenge
    "challenge-platform",  # Cloudflare turnstile
    "_Incapsula_Resource",  # Imperva/Incapsula
    "ak_bmsc",        # Akamai Bot Manager cookie hint
    "bmak.",          # Akamai script token
    "robot check",    # Amazon bot check page
    "access denied",  # Generic block page
)


def _is_bot_challenge(html: str) -> bool:
    """Return True if the page HTML looks like a bot-detection challenge page."""
    sample = html[:40_000].lower()
    return any(s in sample for s in _BOT_CHALLENGE_SIGNALS)


class BrowserSession:
    """Playwright (rebrowser) or camoufox session scoped to a single navigation task.

    Starts with rebrowser-playwright (Chromium). After the first navigation, if a
    bot-challenge page is detected, the session automatically restarts with camoufox
    (Firefox with C++-level fingerprint spoofing) and retries the URL.
    """

    def __init__(self, *, headless: bool = True, timeout_ms: int = _DEFAULT_TIMEOUT_MS, url: str = "") -> None:
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._use_camoufox = False
        self._bot_check_done = False
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._camoufox_browser: object | None = None

    async def __aenter__(self) -> "BrowserSession":
        await self._start_rebrowser()
        return self

    async def _start_rebrowser(self) -> None:
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        self._context = await self._browser.new_context(
            user_agent=_USER_AGENT,
            locale="en-US",
            timezone_id="America/New_York",
            viewport={"width": 1366, "height": 768},
        )
        await self._context.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
        self._page = await self._context.new_page()
        await self._page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

    async def _start_camoufox(self) -> bool:
        """Start camoufox. Returns True on success, False on any startup failure."""
        try:
            from camoufox.async_api import AsyncCamoufox
        except ImportError:
            logger.warning("camoufox not installed — staying with rebrowser")
            return False
        logger.info("[browser] switching to camoufox (Firefox) for bot-detection bypass")

        camoufox_browser: object | None = None
        try:
            camoufox_browser = AsyncCamoufox(headless=self._headless)
            browser = await camoufox_browser.__aenter__()
            self._camoufox_browser = camoufox_browser
            self._page = await browser.new_page()
            self._use_camoufox = True
            return True
        except Exception as exc:
            logger.warning("[browser] camoufox startup failed (%s) — falling back to rebrowser", exc)
            if camoufox_browser is not None:
                try:
                    await camoufox_browser.__aexit__(None, None, None)
                except Exception:
                    pass
            self._camoufox_browser = None
            self._use_camoufox = False
            self._page = None
            return False

    async def _close_rebrowser(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None
        self._context = None
        self._page = None

    async def __aexit__(self, *_: object) -> None:
        if self._camoufox_browser is not None:
            try:
                await self._camoufox_browser.__aexit__(None, None, None)
            except Exception:
                pass
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                pass

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserSession not started — use 'async with'")
        return self._page

    async def _goto(self, url: str) -> None:
        """Raw navigation with domcontentloaded → commit fallback."""
        page = self.page
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=min(self._timeout_ms, 15_000))
        except Exception as primary_exc:
            logger.warning("Primary navigation to %s failed (%s); trying commit fallback", url, primary_exc)
            try:
                await page.goto(url, wait_until="commit", timeout=min(self._timeout_ms, 12_000))
            except Exception as fallback_exc:
                raise RuntimeError(f"Navigation to {url!r} failed: {fallback_exc}") from fallback_exc

    async def navigate(self, url: str) -> None:
        """Navigate to url. On first load, auto-detects bot challenges and switches
        to camoufox (Firefox) if one is found, then retries the navigation."""
        await self._goto(url)

        # One-time bot-challenge check after first navigation
        if not self._bot_check_done and not self._use_camoufox:
            self._bot_check_done = True
            html = await self.get_html()
            if _is_bot_challenge(html):
                logger.info("[browser] bot challenge detected at %s — restarting with camoufox", url)
                await self._close_rebrowser()
                started = await self._start_camoufox()
                if started:
                    await self._goto(url)
                else:
                    # camoufox unavailable — restore rebrowser so _page is valid
                    await self._start_rebrowser()
                    await self._goto(url)

    async def get_html(self) -> str:
        for attempt in range(3):
            try:
                return await self.page.content()
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1.0)
        return await self.page.content()

    async def get_aria_snapshot(self) -> str | None:
        """Return a compact ARIA accessibility tree for the current page (rebrowser only)."""
        if self._use_camoufox:
            return None
        try:
            snap = await self.page.accessibility.snapshot()
            if not snap:
                return None
            return _format_aria(snap)
        except Exception:
            return None

    def get_url(self) -> str:
        return self.page.url


def _format_aria(node: dict, depth: int = 0) -> str:
    """Recursively format an ARIA snapshot dict into compact text for LLM context."""
    indent = "  " * depth
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    parts = [role]
    if name:
        parts.append(f'"{name}"')
    if value:
        parts.append(f'= "{value}"')
    line = indent + " ".join(parts)

    children = node.get("children", [])
    child_lines = [_format_aria(c, depth + 1) for c in children[:20]]
    return "\n".join([line] + child_lines)
