from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlsplit

from agent.models import PRICE_PATTERN, PageArchetype, PageState

_QUOTE_VALUE_PATTERN = re.compile(r"\b\d{1,4}(?:,\d{3})*(?:\.\d{1,2})\b")
_TRAVEL_KEYWORDS = (
    "depart",
    "return",
    "flight",
    "traveller",
    "traveler",
    "passenger",
    "airport",
    "hotel",
    "check in",
    "check-in",
    "check out",
    "check-out",
    "stay",
    "guests",
    "rooms",
)
_PRODUCT_KEYWORDS = (
    "add to cart",
    "add to bag",
    "buy",
    "shop",
    "trade in",
    "monthly payments",
    "starting at",
    "from $",
    "from €",
    "from £",
)
_RESULT_KEYWORDS = (
    "results for",
    "search results",
    "showing results",
    "sort by",
    "filters",
    "filter by",
)
_QUOTE_KEYWORDS = (
    "market cap",
    "previous close",
    "open",
    "day range",
    "52 week range",
    "52-week range",
    "quote",
    "volume",
    "pe ratio",
)
_FORM_KEYWORDS = (
    "sign up",
    "sign in",
    "log in",
    "login",
    "create account",
    "join",
    "password",
    "email",
)
_CONTENT_BUDGET_CHARS = 8000
_QUOTE_CONTENT_BUDGET_CHARS = 4000
_MIN_MULTI_SIGNAL_COUNT = 2
_MIN_RESULTS_LINK_COUNT_WITH_SEARCH = 5
_MIN_RESULTS_LINK_COUNT_GENERIC = 4
_MIN_FORM_CONTROL_COUNT = 3
_BODY_REGIONS = {"main", "form", "body"}
_NAV_REGIONS = {"header", "nav", "footer"}
_SEARCH_QUERY_TOKENS = ("q=", "query=", "search=", "keyword=")


@dataclass(frozen=True)
class _PageSignals:
    content: str
    title_lower: str
    is_homepage: bool
    search_url_signal: bool
    input_count: int
    select_count: int
    checkable_count: int
    link_count: int
    main_like_count: int
    nav_like_count: int
    search_input_count: int
    date_signal_count: int
    price_signal_count: int
    quote_value_signal_count: int
    option_signal_count: int
    body_link_count: int
    body_button_count: int
    buy_signal_count: int
    quote_keyword_count: int
    commerce_keyword_count: int
    strong_travel_signal: bool
    route_pair_signal: bool


def _contains_any(content: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in content for keyword in keywords)


def _is_body_region(region: str | None) -> bool:
    return (region or "body") in _BODY_REGIONS


def _is_nav_region(region: str | None) -> bool:
    return (region or "") in _NAV_REGIONS


def _is_search_input(item) -> bool:
    if item.kind != "input":
        return False
    text = f"{item.label.lower()} {(item.context_text or '').lower()}"
    return (item.field_type or "").lower() == "search" or "search" in text


def _has_date_signal(item) -> bool:
    text = f"{item.label.lower()} {(item.context_text or '').lower()}"
    return "date" in text or (item.field_type or "").lower() == "date"


def _has_buy_signal(item) -> bool:
    if item.kind not in {"link", "button"}:
        return False
    text = f"{item.label.lower()} {(item.context_text or '').lower()}"
    href = (item.href or "").lower()
    return "buy" in text or "/buy_" in href or "/shop" in href


def _build_page_signals(state: PageState) -> _PageSignals:
    content = f"{state.url}\n{state.title}\n{state.markdown[:_CONTENT_BUDGET_CHARS]}".lower()
    parsed_url = urlsplit(state.url)
    path_lower = parsed_url.path.lower()
    query_lower = parsed_url.query.lower()

    interactables = state.interactables
    input_count = sum(item.kind == "input" for item in interactables)
    select_count = sum(item.kind == "select" for item in interactables)
    checkable_count = sum(item.kind in {"checkbox", "radio"} for item in interactables)
    link_count = sum(item.kind == "link" for item in interactables)
    main_like_count = sum(_is_body_region(item.region) for item in interactables)
    nav_like_count = sum(_is_nav_region(item.region) for item in interactables)
    search_input_count = sum(_is_search_input(item) for item in interactables)
    date_signal_count = sum(_has_date_signal(item) for item in interactables)
    price_signal_count = len(PRICE_PATTERN.findall(state.markdown[:_CONTENT_BUDGET_CHARS]))
    quote_value_signal_count = len(_QUOTE_VALUE_PATTERN.findall(state.markdown[:_QUOTE_CONTENT_BUDGET_CHARS]))
    option_signal_count = sum(bool(item.options) for item in interactables)
    body_link_count = sum(item.kind == "link" and _is_body_region(item.region) for item in interactables)
    body_button_count = sum(item.kind == "button" and _is_body_region(item.region) for item in interactables)
    buy_signal_count = sum(_has_buy_signal(item) for item in interactables)

    return _PageSignals(
        content=content,
        title_lower=state.title.lower(),
        is_homepage=parsed_url.path in {"", "/"} and not parsed_url.query and not parsed_url.fragment,
        search_url_signal="search" in path_lower or any(token in query_lower for token in _SEARCH_QUERY_TOKENS),
        input_count=input_count,
        select_count=select_count,
        checkable_count=checkable_count,
        link_count=link_count,
        main_like_count=main_like_count,
        nav_like_count=nav_like_count,
        search_input_count=search_input_count,
        date_signal_count=date_signal_count,
        price_signal_count=price_signal_count,
        quote_value_signal_count=quote_value_signal_count,
        option_signal_count=option_signal_count,
        body_link_count=body_link_count,
        body_button_count=body_button_count,
        buy_signal_count=buy_signal_count,
        quote_keyword_count=sum(keyword in content for keyword in _QUOTE_KEYWORDS),
        commerce_keyword_count=sum(keyword in content for keyword in _PRODUCT_KEYWORDS),
        strong_travel_signal=_contains_any(content, _TRAVEL_KEYWORDS),
        route_pair_signal="from" in content
        and "to" in content
        and date_signal_count >= 1
        and select_count >= 1,
    )


def _base_hints(signals: _PageSignals) -> set[str]:
    hints: set[str] = set()
    if signals.price_signal_count:
        hints.add("price_signals")
    if signals.search_input_count:
        hints.add("search_input")
    if signals.date_signal_count:
        hints.add("date_controls")
    if signals.option_signal_count:
        hints.add("dropdowns")
    if signals.buy_signal_count >= _MIN_MULTI_SIGNAL_COUNT:
        hints.add("commerce_links")
    if signals.body_link_count >= _MIN_MULTI_SIGNAL_COUNT:
        hints.add("body_links")
    if signals.main_like_count:
        hints.add("body_interactables")
    if signals.nav_like_count and signals.nav_like_count >= signals.main_like_count:
        hints.add("nav_heavy")
    return hints


def classify_page_state(state: PageState) -> tuple[PageArchetype, list[str]]:
    signals = _build_page_signals(state)
    hints = _base_hints(signals)

    is_quote_page = signals.quote_keyword_count >= _MIN_MULTI_SIGNAL_COUNT and (
        signals.price_signal_count > 0 or signals.quote_value_signal_count >= _MIN_MULTI_SIGNAL_COUNT
    )
    if is_quote_page:
        hints.add("quote_module")
        return "quote", sorted(hints)

    is_product_detail_page = signals.price_signal_count > 0 and any(
        value >= 1
        for value in (
            signals.commerce_keyword_count,
            signals.buy_signal_count,
            signals.option_signal_count,
            signals.body_button_count,
        )
    )
    if is_product_detail_page:
        hints.add("commerce_signals")
        return "product_detail", sorted(hints)

    has_search_results_signal = (
        signals.search_url_signal
        or _contains_any(signals.content, _RESULT_KEYWORDS)
        or "search results" in signals.title_lower
    )
    if signals.search_input_count >= 1 and signals.link_count >= _MIN_RESULTS_LINK_COUNT_WITH_SEARCH and has_search_results_signal:
        hints.add("result_links")
        return "search_results", sorted(hints)

    if _contains_any(signals.content, _RESULT_KEYWORDS) and signals.link_count >= _MIN_RESULTS_LINK_COUNT_GENERIC:
        hints.add("result_links")
        return "search_results", sorted(hints)

    if (
        not signals.is_homepage
        and signals.buy_signal_count >= _MIN_MULTI_SIGNAL_COUNT
        and signals.body_link_count >= _MIN_MULTI_SIGNAL_COUNT
    ):
        hints.add("result_links")
        return "search_results", sorted(hints)

    has_travel_controls = (
        signals.search_input_count >= _MIN_MULTI_SIGNAL_COUNT
        or signals.date_signal_count >= 1
        or signals.select_count >= 1
    )
    if (signals.strong_travel_signal or signals.route_pair_signal) and has_travel_controls:
        hints.add("travel_form")
        return "travel_search", sorted(hints)

    has_form_controls = (
        signals.input_count + signals.select_count + signals.checkable_count >= _MIN_FORM_CONTROL_COUNT
    )
    if has_form_controls or _contains_any(signals.content, _FORM_KEYWORDS):
        hints.add("form_controls")
        return "form", sorted(hints)

    return "generic", sorted(hints)


def enrich_page_state(state: PageState) -> PageState:
    archetype, hints = classify_page_state(state)
    state.page_archetype = archetype
    state.page_hints = hints
    return state
