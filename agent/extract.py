from __future__ import annotations

import re

from markdownify import markdownify
from readability import Document


def _normalize_text(value: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def _markdownify_fragment(
    html: str,
    *,
    title: str | None = None,
    max_chars: int = 12000,
) -> str:
    markdown = markdownify(html, heading_style="ATX")
    normalized_title = (title or "").strip()
    if normalized_title:
        markdown = f"# {normalized_title}\n\n{markdown}"
    return _normalize_text(markdown)[:max_chars]


def html_to_markdown(
    html: str,
    max_chars: int = 12000,
    *,
    prefer_readability: bool = True,
) -> str:
    if not html:
        return ""

    if not prefer_readability:
        return _markdownify_fragment(html, max_chars=max_chars)

    doc = Document(html)
    main_html = doc.summary(html_partial=True)
    title = doc.short_title()

    readable_markdown = _markdownify_fragment(main_html, title=title, max_chars=max_chars)
    if readable_markdown:
        return readable_markdown

    # Readability can strip scoped fragments (for example a selected table).
    # Fall back to direct conversion so targeted extraction still has evidence text.
    return _markdownify_fragment(html, title=title, max_chars=max_chars)


async def page_to_markdown(
    page,
    max_chars: int = 12000,
    *,
    selector: str | None = None,
) -> str:
    scoped_html: str | None = None
    if selector:
        try:
            candidate = await page.eval_on_selector(selector, "el => el.outerHTML")
            if isinstance(candidate, str) and candidate.strip():
                scoped_html = candidate
        except Exception:
            scoped_html = None

    if scoped_html is not None:
        return html_to_markdown(scoped_html, max_chars=max_chars, prefer_readability=False)

    html = await page.content()
    return html_to_markdown(html, max_chars=max_chars)
