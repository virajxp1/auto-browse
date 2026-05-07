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


class BrowserSession:
    """Lightweight Playwright session scoped to a single subtask.

    Each session owns its own browser process for full isolation between
    parallel subtask branches running in the same asyncio event loop.
    """

    def __init__(self, *, headless: bool = True, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> None:
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def __aenter__(self) -> "BrowserSession":
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
        return self

    async def __aexit__(self, *_: object) -> None:
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

    async def navigate(self, url: str) -> None:
        """Navigate to url with a two-attempt strategy for slow/SPA pages."""
        page = self.page
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=min(self._timeout_ms, 15_000))
        except Exception as primary_exc:
            logger.warning("Primary navigation to %s failed (%s); trying commit fallback", url, primary_exc)
            try:
                await page.goto(url, wait_until="commit", timeout=min(self._timeout_ms, 12_000))
            except Exception as fallback_exc:
                raise RuntimeError(f"Navigation to {url!r} failed: {fallback_exc}") from fallback_exc

    async def get_html(self) -> str:
        for attempt in range(3):
            try:
                return await self.page.content()
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1.0)
        return await self.page.content()

    def get_url(self) -> str:
        return self.page.url
