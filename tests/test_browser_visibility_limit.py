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

    async def query_selector_all(self, query):
        if query == "a[href]":
            return self._handles
        if query.startswith('css=:is(a[href])[href="') and query.endswith('"]'):
            href = query[len('css=:is(a[href])[href="') : -2]
            return [h for h in self._handles if h.href == href]
        if query.startswith('role=link[name="') and query.endswith('"]'):
            name = query[len('role=link[name="') : -2]
            return [h for h in self._handles if h.text == name]
        return []


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
        self.assertEqual(interactables[0].selector, 'css=:is(a[href])[href="/visible"]')

    async def test_falls_back_to_nth_when_best_selectors_are_not_unique(self) -> None:
        page = _FakePage(
            [
                _FakeHandle(visible=True, href="/about", text="About"),
                _FakeHandle(visible=True, href="/about", text="About"),
            ]
        )

        interactables = await _build_link_interactables(page, limit=1)

        self.assertEqual(len(interactables), 1)
        self.assertEqual(interactables[0].selector, "css=a[href] >> nth=0")
