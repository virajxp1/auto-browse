from __future__ import annotations

import re
from urllib.parse import parse_qsl, urljoin, urlsplit

from agent.models import Interactable, PageState

_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "how",
    "in",
    "its",
    "of",
    "on",
    "page",
    "report",
    "show",
    "tell",
    "the",
    "their",
    "this",
    "title",
    "to",
    "what",
}

_SEARCH_INPUT_PHRASES = (
    "search",
    "find",
    "query",
    "filter",
    "keyword",
)

_LOOKUP_INPUT_PHRASES = (
    "city",
    "location",
    "lookup",
    "symbol",
    "ticker",
    "zip",
)

_SEARCH_TRIGGER_PHRASES = (
    "search",
    "find",
    "open search",
)

_ASSISTANT_PHRASES = (
    "assistant",
    "copilot",
    "chat",
    "ask gordon",
)

_ACTION_CLAUSE_PATTERN = re.compile(
    r"(?:,|\s)+(?:and\s+)?(?:open|click|report|tell|show|return|extract|give|visit|navigate|go)\b.*$",
    flags=re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_trailing_action_clause(value: str) -> str:
    normalized = _normalize_text(value).strip(" ,.;:")
    if not normalized:
        return ""
    stripped = _ACTION_CLAUSE_PATTERN.sub("", normalized).strip(" ,.;:")
    return stripped or normalized


def _strip_site_qualifier(value: str) -> str:
    normalized = _normalize_text(value).strip(" ,.;:")
    if not normalized:
        return ""

    stripped = re.sub(
        r"\s+on\s+[^,.;]+?\bwebsite\b$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip(" ,.;:")
    return stripped or normalized


def extract_task_query(target_prompt: str) -> str | None:
    normalized = _normalize_text(target_prompt)
    if not normalized:
        return None

    patterns = (
        r"search(?: .*?)? for (?P<query>.+)$",
        r"find (?P<query>.+)$",
        r"(?:open|click)(?: on)? (?:the )?(?:first|top)\s+(?P<query>.+?)\s+result(?:s)?(?:\b|$)",
        r"go to (?:the )?(?P<query>.+?)(?: and (?:report|show|tell|give) (?:the )?page title|\.|$)",
        r"report the page title of (?P<query>.+)$",
        r"(?:price|cost) of (?P<query>.+?)(?: on | at | in | from |\.|$)",
        r"quote page for (?P<query>.+?)(?: and | on | in | at | from | then |\.|$)",
        r"look up (?P<query>.+)$",
        r"navigate to (?:the )?(?P<query>.+?)(?: page| and |\.|$)",
        r"(?:tell|show|give) (?:me )?(?:the )?(?:current )?(?:stock )?(?:price|quote|value) (?:of|for) (?P<query>.+?)(?:\.|$)",
        r"(?:what is|what's) (?:the )?(?:current )?(?:stock )?(?:price|quote|value) (?:of|for) (?P<query>.+?)(?:\?|\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match is None:
            continue
        query = _strip_site_qualifier(_strip_trailing_action_clause(match.group("query")))
        if query:
            return query
    return None


def query_tokens(value: str) -> set[str]:
    de_camelized = re.sub(r"(?<=[a-z][a-z])(?=[A-Z])", " ", value)
    normalized = re.sub(r"[^a-z0-9]+", " ", de_camelized.lower()).strip()
    if not normalized:
        return set()
    tokens = {
        token
        for token in normalized.split()
        if len(token) > 1 and token not in _STOPWORDS
    }
    if any(len(token) >= 3 for token in tokens):
        tokens = {token for token in tokens if len(token) >= 3}
    return tokens


def is_search_like_input(item: Interactable) -> bool:
    if item.kind != "input":
        return False
    text = _normalize_text(" ".join(part for part in [item.label, item.context_text or ""] if part)).lower()
    field_type = (item.field_type or "").lower()
    if any(phrase in text for phrase in _ASSISTANT_PHRASES) and "search" not in text and "find" not in text:
        return False
    if (
        field_type == "search"
        or any(phrase in text for phrase in _SEARCH_INPUT_PHRASES)
        or any(phrase in text for phrase in _LOOKUP_INPUT_PHRASES)
    ):
        return True
    # Detect search inputs by ID/name patterns in the selector (e.g. #search-bar, [name="q"])
    selector_lower = (item.selector or "").lower()
    id_name_part = re.search(r'(?:#|id="|name="|id=)([a-z0-9_-]+)', selector_lower)
    if id_name_part:
        id_val = id_name_part.group(1)
        if any(token in id_val for token in ("search", "query", "autocomplete")):
            return True
    return False


def is_search_trigger(item: Interactable) -> bool:
    if item.kind not in {"button", "link"}:
        return False
    text = _normalize_text(" ".join(part for part in [item.label, item.context_text or "", item.selector] if part)).lower()
    return any(phrase in text for phrase in _SEARCH_TRIGGER_PHRASES)


def task_requires_destination_page(target_prompt: str) -> bool:
    normalized = target_prompt.lower()
    phrases = (
        "open the first result",
        "click the first result",
        "click on the first result",
        "open the result",
        "click the result",
        "destination page",
        "relevant page",
        "open the repo",
        "open the repository",
        "download page",
        "repo title",
        "package title",
    )
    if any(phrase in normalized for phrase in phrases):
        return True
    return bool(re.search(r"\b(open|click|visit|navigate to|go to)\b", normalized))


def _required_query_hits(tokens: set[str]) -> int:
    if len(tokens) <= 2:
        return len(tokens)
    return min(3, max(2, len(tokens) // 2))


def _related_host_score(page_url: str, href: str | None) -> int:
    if not href:
        return 0
    try:
        current_host = urlsplit(page_url).hostname or ""
        target_host = urlsplit(urljoin(page_url, href)).hostname or ""
    except Exception:
        return 0
    if not current_host or not target_host:
        return 0
    if target_host == current_host:
        return 10
    if target_host.endswith(f".{current_host}") or current_host.endswith(f".{target_host}"):
        return 6
    return -8


def _is_pagination_like(page_url: str, link: Interactable) -> bool:
    href = link.href or ""
    if not href:
        return False
    try:
        current = urlsplit(page_url)
        target = urlsplit(urljoin(page_url, href))
    except Exception:
        return False

    if not current.hostname or not target.hostname:
        return False

    normalized_label = _normalize_text(link.label).lower()
    if normalized_label in {"next", "previous", "prev", "next page", "previous page"}:
        return True
    if re.fullmatch(r"\d+", normalized_label):
        return True

    if current.hostname != target.hostname or current.path != target.path:
        return False

    pagination_keys = {"page", "p", "offset", "start"}
    target_query_keys = {key.lower() for key, _ in parse_qsl(target.query, keep_blank_values=True)}
    return bool(pagination_keys & target_query_keys)


def canonical_result_url(page_url: str, href: str | None, query: str | None) -> str | None:
    if not href or not query:
        return None
    normalized_query = query.strip().strip("/")
    if "/" not in normalized_query or " " in normalized_query:
        return None

    try:
        resolved_url = urljoin(page_url, href)
        parsed = urlsplit(resolved_url)
    except Exception:
        return None

    lowered_path = parsed.path.lower()
    slug = f"/{normalized_query.lower()}"
    index = lowered_path.find(slug)
    if index < 0:
        return None
    after_slug = lowered_path[index + len(slug) :]
    if after_slug and not after_slug.startswith("/"):
        return None
    canonical_path = parsed.path[: index + len(slug)]
    if canonical_path == parsed.path:
        return resolved_url
    return parsed._replace(path=canonical_path, query="", fragment="").geturl()


def page_matches_query(page_state: PageState, query: str | None) -> bool:
    if not query:
        return False
    tokens = query_tokens(query)
    if not tokens:
        return False

    identity_hits = len(tokens & query_tokens(f"{page_state.url} {page_state.title}"))
    markdown_hits = len(tokens & query_tokens(page_state.markdown[:4000]))
    required = _required_query_hits(tokens)

    if identity_hits >= required:
        return True
    if identity_hits >= max(1, required - 1) and markdown_hits >= 1:
        return True
    return markdown_hits >= required


def page_identity_matches_query(page_state: PageState, query: str | None) -> bool:
    if not query:
        return False
    tokens = query_tokens(query)
    if not tokens:
        return False
    identity_hits = len(tokens & query_tokens(f"{page_state.url} {page_state.title}"))
    required = _required_query_hits(tokens)
    return identity_hits >= max(1, required - 1)


def search_progress_state(page_state: PageState, target_prompt: str) -> str:
    query = extract_task_query(target_prompt)
    if not query:
        return "none"

    content = f"{page_state.url}\n{page_state.title}\n{page_state.markdown[:3000]}".lower()
    body_link_count = sum(
        item.kind == "link" and (item.region or "body") in {"main", "form", "body", "aside"}
        for item in page_state.interactables
    )
    search_input_present = any(is_search_like_input(item) for item in page_state.interactables)
    search_trigger_present = any(is_search_trigger(item) for item in page_state.interactables)
    query_anchored_result = best_result_link(page_state, query)
    query_matches = page_matches_query(page_state, query)
    strong_identity_match = page_identity_matches_query(page_state, query)
    query_relevant_body_link_count = sum(
        1
        for item in page_state.interactables
        if item.kind == "link"
        and (item.region or "body") in {"main", "form", "body", "aside"}
        and result_link_relevance(item, query, page_url=page_state.url) >= 0
    )
    result_signal = any(
        phrase in content
        for phrase in (
            "results for",
            "search results",
            "showing results",
            "result",
            "sort by",
            "filter",
        )
    )

    # Detect URL-level search signals (e.g. ?q=foo, /search?query=bar)
    url_has_search_query = False
    try:
        parsed = urlsplit(page_state.url)
        url_has_search_query = any(
            token in parsed.query.lower()
            for token in ("q=", "query=", "search=", "keyword=")
        ) or "search" in parsed.path.lower()
    except Exception:
        pass

    # Results list detection: prioritize structural signals over content matching
    is_results_page = (
        (query_anchored_result is not None and not strong_identity_match and (
            page_state.page_archetype == "search_results"
            or "result_links" in page_state.page_hints
            or (result_signal and body_link_count >= 1)
            or query_relevant_body_link_count >= 2
            or (url_has_search_query and body_link_count >= 2)
        ))
        # Trust archetype classification even without anchored results
        # (e.g. thin results page during hydration, or search URL with result signals)
        or (page_state.page_archetype == "search_results"
            and "result_links" in page_state.page_hints
            and (result_signal or url_has_search_query))
        # URL-level search signals dominate identity matching (e.g. ?q=foo on a results page)
        or (url_has_search_query and result_signal and not strong_identity_match)
    )
    if is_results_page:
        return "results_list"

    # Destination candidate: require strong identity match, not just content overlap
    # But not when URL clearly indicates a search results page
    if strong_identity_match and query_matches and not url_has_search_query:
        return "destination_candidate"

    # Weaker destination signal: content matches but no identity match
    # Only treat as destination if not on a results-like page
    if query_matches and not (search_input_present and body_link_count >= 3):
        return "destination_candidate"

    # Search entry: page has a search box and no matching results yet
    if (
        (search_input_present or search_trigger_present or "search_input" in page_state.page_hints)
        and query_anchored_result is None
    ):
        return "search_entry"

    # Weak results signal: body links present with some query overlap but not enough for full results_list
    if body_link_count >= 2 and query_relevant_body_link_count >= 1 and not query_matches:
        return "results_list"

    return "unknown"


def result_link_relevance(link: Interactable, query: str | None, *, page_url: str = "") -> int:
    if link.kind != "link":
        return -10_000

    score = 0
    region_scores = {"main": 15, "form": 12, "body": 10, "aside": 4, "header": -4, "nav": -6, "footer": -8}
    score += region_scores.get(link.region or "body", 0)

    candidate_text = _normalize_text(" ".join(part for part in [link.label, link.context_text, link.href] if part))
    if query:
        query_token_set = query_tokens(query)
        candidate_token_set = query_tokens(candidate_text)
        overlap = len(query_token_set & candidate_token_set)
        if overlap == 0:
            score -= 20
        else:
            score += overlap * 12
            normalized_candidate = candidate_text.lower()
            normalized_query = query.lower()
            if normalized_candidate.startswith(normalized_query):
                score += 10
            elif normalized_query in normalized_candidate:
                score += 4
            score += _related_host_score(page_url, link.href)
    else:
        score += _related_host_score(page_url, link.href)

    if link.href and not link.href.startswith(("#", "javascript:")):
        score += 3
    if 3 <= len(link.label) <= 80:
        score += 1

    normalized_label = link.label.lower()
    normalized_href = (link.href or "").lower()
    if page_url and _is_pagination_like(page_url, link):
        score -= 24
    if normalized_label.startswith('search "') or normalized_label.startswith("search "):
        score -= 12
    if normalized_label.startswith("link_"):
        score -= 10
    if any(token in normalized_href for token in ("google.com/search", "cse.google.com", "programmablesearchengine.google.com")):
        score -= 12
    # Boost links whose href path contains query tokens (e.g. /docs/asyncio-gather)
    if query and link.href:
        href_path = normalized_href.split("?")[0].split("#")[0]
        path_tokens = set(re.split(r"[/\-_.]", href_path))
        q_tokens = query_tokens(query)
        path_overlap = len(q_tokens & path_tokens)
        if path_overlap >= 1:
            score += path_overlap * 5
    return score


def best_result_link(page_state: PageState, query: str | None) -> Interactable | None:
    body_links = [
        item
        for item in page_state.interactables
        if item.kind == "link" and (item.region or "body") in {"main", "form", "body", "aside"}
    ]
    if not body_links:
        return None

    ranked = sorted(
        body_links,
        key=lambda item: (-result_link_relevance(item, query, page_url=page_state.url), item.label.lower()),
    )
    best = ranked[0]
    if query and result_link_relevance(best, query, page_url=page_state.url) < 0:
        return None
    return best
