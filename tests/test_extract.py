from __future__ import annotations

import unittest

from agent.extract import html_to_markdown, page_to_markdown


class _StubPage:
    async def eval_on_selector(self, selector: str, _script: str):
        if selector == "css=table.infobox":
            return (
                "<table class='infobox'>"
                "<tr><th>Release date</th><td>May 25, 1977</td></tr>"
                "<tr><th>Director</th><td>George Lucas</td></tr>"
                "<tr><th>Producer</th><td>Gary Kurtz</td></tr>"
                "</table>"
            )
        return None

    async def content(self) -> str:
        return (
            "<html><head><title>Example</title></head>"
            "<body><main><h1>Example Domain</h1><p>Sample body text</p></main></body></html>"
        )


class ExtractMarkdownTest(unittest.IsolatedAsyncioTestCase):
    def test_html_to_markdown_falls_back_for_fragment(self) -> None:
        html = (
            "<table>"
            "<tr><th>Director</th><td>George Lucas</td></tr>"
            "<tr><th>Producer</th><td>Gary Kurtz</td></tr>"
            "</table>"
        )

        markdown = html_to_markdown(html, prefer_readability=True)

        self.assertIn("George Lucas", markdown)
        self.assertIn("Gary Kurtz", markdown)

    async def test_page_to_markdown_keeps_scoped_table_content(self) -> None:
        page = _StubPage()

        markdown = await page_to_markdown(page, selector="css=table.infobox")

        self.assertIn("May 25, 1977", markdown)
        self.assertIn("George Lucas", markdown)
        self.assertIn("Gary Kurtz", markdown)

    async def test_page_to_markdown_uses_full_content_without_selector_match(self) -> None:
        page = _StubPage()

        markdown = await page_to_markdown(page, selector="css=main article")

        self.assertIn("Example Domain", markdown)
        self.assertIn("Sample body text", markdown)
