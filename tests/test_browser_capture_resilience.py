from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from agent.browser import _first_unique_selector, capture_state, goto_with_fallback
from agent.models import Interactable, PageState
from agent.snapshot import PageSnapshotService


class _FakePage:
    url = "https://example.com"

    async def title(self) -> str:
        return "Example"


class _GotoFallbackPage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.waited_for_dom = False

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.calls.append((url, wait_until, timeout))
        if wait_until == "domcontentloaded":
            raise RuntimeError("domcontentloaded timeout")

    async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
        _ = timeout
        if state == "domcontentloaded":
            self.waited_for_dom = True


class _VisibilityHandle:
    def __init__(self, visible: bool) -> None:
        self.visible = visible

    async def evaluate(self, _script: str, _require_enabled: bool = False):
        return self.visible


class _SelectorDisambiguationPage:
    def __init__(self) -> None:
        self._matches = {
            'css=button[aria-label="Search"]': [_VisibilityHandle(False), _VisibilityHandle(True)],
            'css=button[id="search-button"]': [_VisibilityHandle(True)],
        }

    async def query_selector_all(self, selector: str):
        return list(self._matches.get(selector, []))


class _SnapshotFallbackPage:
    def __init__(self) -> None:
        self.url = "https://example.com/search"
        self.waited_for_networkidle = False

    async def title(self) -> str:
        return "Example Search"

    async def evaluate(self, script: str, *_args):
        if "body.innerText" in script:
            return "Visible fallback page text"
        return ""

    async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
        _ = timeout
        if state == "networkidle":
            self.waited_for_networkidle = True


class BrowserCaptureResilienceTest(unittest.IsolatedAsyncioTestCase):
    async def test_first_unique_selector_accepts_single_visible_match(self) -> None:
        page = _SelectorDisambiguationPage()

        selector = await _first_unique_selector(
            page,
            ['css=button[aria-label="Search"]', 'css=button[id="search-button"]'],
            fallback="css=button >> nth=0",
        )

        self.assertEqual(selector, 'css=button[aria-label="Search"]')

    async def test_builder_failure_does_not_skip_other_interactables(self) -> None:
        button = Interactable(kind="button", label="Submit", selector="css=button >> nth=0")
        link = Interactable(kind="link", label="Home", selector="css=a[href] >> nth=0", href="/")

        with (
            patch("agent.browser._build_input_interactables", new=AsyncMock(side_effect=RuntimeError("boom"))),
            patch("agent.browser._build_button_interactables", new=AsyncMock(return_value=[button])),
            patch("agent.browser._build_link_interactables", new=AsyncMock(return_value=[link])),
        ):
            state = await capture_state(_FakePage())

        self.assertEqual(state.url, "https://example.com")
        self.assertEqual(state.title, "Example")
        self.assertEqual({item.kind for item in state.interactables}, {"button", "link"})

    async def test_goto_with_fallback_uses_commit_when_domcontentloaded_times_out(self) -> None:
        page = _GotoFallbackPage()

        await goto_with_fallback(page, "https://www.bestbuy.com/", timeout_ms=15000)

        self.assertEqual(
            page.calls,
            [
                ("https://www.bestbuy.com/", "domcontentloaded", 15000),
                ("https://www.bestbuy.com/", "commit", 12000),
            ],
        )
        self.assertTrue(page.waited_for_dom)

    async def test_snapshot_service_falls_back_to_visible_text_when_markdown_fails(self) -> None:
        page = _SnapshotFallbackPage()
        snapshot_service = PageSnapshotService(
            page=page,
            capture_state_fn=AsyncMock(
                return_value=PageState(
                    url=page.url,
                    title="Example Search",
                    markdown="",
                    interactables=[],
                )
            ),
            markdown_fn=AsyncMock(side_effect=RuntimeError("markdown failed")),
        )

        state = await snapshot_service.capture()

        self.assertEqual(state.url, "https://example.com/search")
        self.assertIn("Visible fallback page text", state.markdown)

    async def test_snapshot_service_uses_minimal_state_when_capture_state_times_out(self) -> None:
        page = _SnapshotFallbackPage()
        snapshot_service = PageSnapshotService(
            page=page,
            capture_state_fn=AsyncMock(side_effect=asyncio.TimeoutError()),
            markdown_fn=AsyncMock(return_value="Visible search results"),
        )

        state = await snapshot_service.capture()

        self.assertEqual(state.url, "https://example.com/search")
        self.assertEqual(state.title, "Example Search")
        self.assertEqual(state.interactables, [])
        self.assertEqual(state.markdown, "Visible search results")

    async def test_snapshot_service_recaptures_thin_search_results_page(self) -> None:
        page = _SnapshotFallbackPage()
        initial_state = PageState(
            url=page.url,
            title="Search Results | Example",
            markdown="Search Results",
            interactables=[
                Interactable(
                    kind="input",
                    label="Search this site",
                    selector='css=input[type="search"]',
                    field_type="search",
                    region="main",
                ),
                Interactable(
                    kind="link",
                    label="Home",
                    selector='role=link[name="Home"]',
                    href="/",
                    region="nav",
                ),
            ],
            page_archetype="search_results",
            page_hints=["search_input", "result_links", "nav_heavy"],
        )
        hydrated_state = PageState(
            url=page.url,
            title="Search Results | Example",
            markdown="Search Results Example result details",
            interactables=[
                Interactable(
                    kind="input",
                    label="Search this site",
                    selector='css=input[type="search"]',
                    field_type="search",
                    region="main",
                ),
                Interactable(
                    kind="link",
                    label="Example result",
                    selector='role=link[name="Example result"]',
                    href="/docs/example-result",
                    region="main",
                    context_text="Example result details",
                ),
            ],
            page_archetype="search_results",
            page_hints=["search_input", "result_links", "body_links"],
        )
        snapshot_service = PageSnapshotService(
            page=page,
            capture_state_fn=AsyncMock(side_effect=[initial_state, hydrated_state]),
            markdown_fn=AsyncMock(side_effect=["Search Results", "Search Results Example result details"]),
        )

        state = await snapshot_service.capture()

        self.assertTrue(page.waited_for_networkidle)
        self.assertEqual(len(state.interactables), 2)
        self.assertEqual(state.interactables[1].label, "Example result")

    async def test_snapshot_service_polls_until_dynamic_results_appear(self) -> None:
        page = _SnapshotFallbackPage()
        thin_state = PageState(
            url=page.url,
            title="Search Results | Example",
            markdown="Search Results",
            interactables=[
                Interactable(
                    kind="input",
                    label="Search this site",
                    selector='css=input[type="search"]',
                    field_type="search",
                    region="main",
                ),
            ],
            page_archetype="search_results",
            page_hints=["search_input", "result_links"],
        )
        hydrated_state = PageState(
            url=page.url,
            title="Search Results | Example",
            markdown="Search Results Example result details",
            interactables=[
                Interactable(
                    kind="input",
                    label="Search this site",
                    selector='css=input[type="search"]',
                    field_type="search",
                    region="main",
                ),
                Interactable(
                    kind="link",
                    label="Example result",
                    selector='role=link[name="Example result"]',
                    href="/docs/example-result",
                    region="main",
                    context_text="Example result details",
                ),
            ],
            page_archetype="search_results",
            page_hints=["search_input", "result_links", "body_links"],
        )
        snapshot_service = PageSnapshotService(
            page=page,
            capture_state_fn=AsyncMock(side_effect=[thin_state, thin_state, hydrated_state]),
            markdown_fn=AsyncMock(
                side_effect=[
                    "Search Results",
                    "Search Results",
                    "Search Results Example result details",
                ]
            ),
        )

        state = await snapshot_service.capture()

        self.assertEqual(state.interactables[-1].label, "Example result")
