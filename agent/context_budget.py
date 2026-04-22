from __future__ import annotations

from dataclasses import dataclass

from agent.models import PageState


@dataclass(frozen=True)
class ContextBudget:
    markdown_chars: int
    interactable_limit: int
    advisor_markdown_chars: int


_DEFAULT_MARKDOWN_CHARS = 3500
_EXTRACT_MARKDOWN_CHARS = 8000
_FORM_MARKDOWN_CHARS = 2500
_COMMERCE_MARKDOWN_CHARS = 5000

_DEFAULT_INTERACTABLE_LIMIT = 16
_FORM_INTERACTABLE_LIMIT = 24
_COMMERCE_INTERACTABLE_LIMIT = 12

_DEFAULT_ADVISOR_MARKDOWN = 5000
_EXTRACT_ADVISOR_MARKDOWN = 8000
_FORM_ADVISOR_MARKDOWN = 3000

# Archetype → (markdown_chars, interactable_limit, advisor_markdown_chars)
_ARCHETYPE_BUDGETS: dict[str, tuple[int, int, int]] = {
    "form": (_FORM_MARKDOWN_CHARS, _FORM_INTERACTABLE_LIMIT, _FORM_ADVISOR_MARKDOWN),
    "travel_search": (_FORM_MARKDOWN_CHARS, _FORM_INTERACTABLE_LIMIT, _FORM_ADVISOR_MARKDOWN),
    "product_detail": (_COMMERCE_MARKDOWN_CHARS, _COMMERCE_INTERACTABLE_LIMIT, _DEFAULT_ADVISOR_MARKDOWN),
    "quote": (_COMMERCE_MARKDOWN_CHARS, _COMMERCE_INTERACTABLE_LIMIT, _DEFAULT_ADVISOR_MARKDOWN),
    "search_results": (_DEFAULT_MARKDOWN_CHARS, _DEFAULT_INTERACTABLE_LIMIT, _DEFAULT_ADVISOR_MARKDOWN),
}


def compute_context_budget(
    page_state: PageState,
    *,
    goal_type: str | None,
    extraction_schema: dict[str, str] | None = None,
) -> ContextBudget:
    """Compute dynamic token budgets based on page archetype and goal type."""
    archetype = page_state.page_archetype or "generic"

    # Extract tasks always get maximum markdown
    if extraction_schema or goal_type == "extract":
        return ContextBudget(
            markdown_chars=_EXTRACT_MARKDOWN_CHARS,
            interactable_limit=_DEFAULT_INTERACTABLE_LIMIT,
            advisor_markdown_chars=_EXTRACT_ADVISOR_MARKDOWN,
        )

    md_chars, interactable_limit, advisor_md = _ARCHETYPE_BUDGETS.get(
        archetype,
        (_DEFAULT_MARKDOWN_CHARS, _DEFAULT_INTERACTABLE_LIMIT, _DEFAULT_ADVISOR_MARKDOWN),
    )

    # Nav-heavy hint: few interactables but lots of text → halve the interactable limit
    # to avoid padding the prompt with redundant nav links
    hints = page_state.page_hints or []
    if "nav_heavy" in hints:
        interactable_limit = max(8, interactable_limit // 2)

    return ContextBudget(
        markdown_chars=md_chars,
        interactable_limit=interactable_limit,
        advisor_markdown_chars=advisor_md,
    )
