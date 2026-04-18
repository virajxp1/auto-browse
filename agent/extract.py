from __future__ import annotations

import re

from markdownify import markdownify
from readability import Document

from agent.models import PRICE_PATTERN


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


def _commerce_signal_score(markdown: str) -> int:
    lowered = markdown.lower()
    keyword_hits = sum(
        keyword in lowered
        for keyword in (
            "buy",
            "shop",
            "trade in",
            "monthly payments",
            "add to cart",
            "add to bag",
            "from $",
            "from €",
            "from £",
        )
    )
    return len(PRICE_PATTERN.findall(markdown)) * 3 + keyword_hits


def _content_density_score(markdown: str) -> int:
    """Score content richness beyond commerce signals."""
    lines = [line.strip() for line in markdown.split("\n") if line.strip()]
    link_lines = sum(1 for line in lines if "[" in line and "](" in line)
    text_lines = len(lines) - link_lines
    return text_lines * 2 + link_lines


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
    raw_markdown = _markdownify_fragment(html, title=title, max_chars=max_chars)

    if not readable_markdown:
        return raw_markdown

    # Prefer raw when readability strips too much useful content
    if _commerce_signal_score(raw_markdown) > _commerce_signal_score(readable_markdown):
        return raw_markdown

    # If readability produced very thin content but raw has much more, prefer raw
    readable_density = _content_density_score(readable_markdown)
    raw_density = _content_density_score(raw_markdown)
    if readable_density < 10 and raw_density > readable_density * 3:
        return raw_markdown

    return readable_markdown


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
