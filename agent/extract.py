from __future__ import annotations

import re

from markdownify import markdownify
from readability import Document


def _normalize_text(value: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def html_to_markdown(html: str, max_chars: int = 12000) -> str:
    if not html:
        return ""

    doc = Document(html)
    main_html = doc.summary(html_partial=True)
    title = doc.short_title()

    markdown = markdownify(main_html, heading_style="ATX")
    if title:
        markdown = f"# {title}\n\n{markdown}"

    return _normalize_text(markdown)[:max_chars]


async def page_to_markdown(page, max_chars: int = 12000) -> str:
    html = await page.content()
    return html_to_markdown(html, max_chars=max_chars)
