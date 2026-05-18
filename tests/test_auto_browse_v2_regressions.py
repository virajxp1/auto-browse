from __future__ import annotations

import asyncio
import unittest

from auto_browse_v2.models import SubTask
from auto_browse_v2.navigator import _simplify_page
from auto_browse_v2.site_router import _url_matches_hint, route_subtask


class _StubClient:
    def __init__(self, response: dict[str, str]) -> None:
        self._response = response

    async def json_completion(self, system: str, user: str, *, temperature: float = 0.0, retries: int = 2) -> dict[str, str]:
        return self._response


class AutoBrowseV2RegressionsTest(unittest.TestCase):
    def test_url_matches_hint_does_not_strip_non_www_prefix_chars(self) -> None:
        self.assertFalse(_url_matches_hint("https://ord.com/docs", "word.com"))
        self.assertFalse(_url_matches_hint("https://evil.ord.com/docs", "word.com"))

    def test_url_matches_hint_allows_www_equivalence(self) -> None:
        self.assertTrue(_url_matches_hint("https://example.com/docs", "www.example.com"))

    def test_route_subtask_fallback_preserves_bare_domain(self) -> None:
        subtask = SubTask(task_id="t1", description="Find rates", site_hint="hyatt.com")
        url = asyncio.run(route_subtask(subtask, client=_StubClient({"url": "not-a-url"})))
        self.assertEqual(url, "https://hyatt.com")

    def test_route_subtask_fallback_extracts_host_from_full_url_hint(self) -> None:
        subtask = SubTask(
            task_id="t2",
            description="Find flights",
            site_hint="https://www.aa.com/booking/find-flights?foo=bar",
        )
        url = asyncio.run(route_subtask(subtask, client=_StubClient({"url": "invalid"})))
        self.assertEqual(url, "https://www.aa.com")

    def test_route_subtask_fallback_rejects_empty_site_hint(self) -> None:
        subtask = SubTask(task_id="t3", description="Find data", site_hint="")
        with self.assertRaises(ValueError):
            asyncio.run(route_subtask(subtask, client=_StubClient({"url": "invalid"})))

    def test_simplify_page_preserves_title_from_head(self) -> None:
        html = (
            "<html><head><title>Example Title</title></head>"
            "<body><p>This paragraph is definitely longer than twenty characters.</p></body></html>"
        )
        page_desc = _simplify_page(html, "https://example.com")
        self.assertIn("Title: Example Title", page_desc)
