from __future__ import annotations

import unittest

from agent.browser import _build_link_interactables


class _FakeHandle:
    def __init__(self, *, visible: bool, href: str, text: str) -> None:
        self.visible = visible
        self.href = href
        self.text = text

    async def evaluate(self, _script, _require_enabled):
        return self.visible

    async def get_attribute(self, name: str):
        if name == "href":
            return self.href
        return None

    async def inner_text(self):
        return self.text


class _FakePage:
    def __init__(self, handles):
        self._handles = handles

    async def query_selector_all(self, _query):
        return self._handles


class BrowserVisibilityLimitTest(unittest.IsolatedAsyncioTestCase):
    async def test_limit_counts_visible_links_not_raw_nodes(self) -> None:
        page = _FakePage(
            [
                _FakeHandle(visible=False, href="/hidden", text="Hidden"),
                _FakeHandle(visible=True, href="/visible", text="Visible"),
            ]
        )

        interactables = await _build_link_interactables(page, limit=1)

        self.assertEqual(len(interactables), 1)
        self.assertEqual(interactables[0].label, "Visible")
        self.assertEqual(interactables[0].selector, "css=a[href] >> nth=1")
