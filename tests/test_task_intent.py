from __future__ import annotations

import unittest

from agent.deep_advisor import fallback_page_analysis
from agent.models import Interactable, PageState
from agent.task_intent import (
    best_result_link,
    canonical_result_url,
    extract_task_query,
    is_search_like_input,
    query_tokens,
    search_progress_state,
    task_requires_destination_page,
)


class TaskIntentTest(unittest.TestCase):
    def test_extract_task_query_preserves_compound_terms(self) -> None:
        prompt = (
            "Search the Kubernetes documentation for taints and tolerations, "
            "open the relevant page, and report the page title."
        )

        self.assertEqual(extract_task_query(prompt), "taints and tolerations")

    def test_extract_task_query_handles_navigation_title_prompt(self) -> None:
        prompt = "Go to the Firefox download page and report the page title."

        self.assertEqual(extract_task_query(prompt), "Firefox download page")

    def test_extract_task_query_strips_trailing_site_qualifier(self) -> None:
        prompt = "Go to the iPhone 17 buy page on Apple's website and report the page title."

        self.assertEqual(extract_task_query(prompt), "iPhone 17 buy page")

    def test_extract_task_query_handles_dotted_domain_search_prompt(self) -> None:
        prompt = "Search weather.gov for Chicago, Illinois and report the page title of the forecast page."

        self.assertEqual(extract_task_query(prompt), "Chicago, Illinois")

    def test_extract_task_query_handles_open_first_result_prompt(self) -> None:
        prompt = "Open the first Nintendo DS result and report its title."

        self.assertEqual(extract_task_query(prompt), "Nintendo DS")

    def test_query_tokens_drop_weak_two_character_fragments_when_stronger_terms_exist(self) -> None:
        self.assertEqual(query_tokens("langchain-ai/langchain"), {"langchain"})

    def test_query_tokens_split_camel_case_symbols(self) -> None:
        self.assertEqual(query_tokens("defineConfig"), {"define", "config"})

    def test_query_tokens_do_not_split_single_letter_brand_prefixes(self) -> None:
        self.assertIn("iphone", query_tokens("iPhone 17 buy page"))

    def test_location_lookup_input_counts_as_search_like(self) -> None:
        item = Interactable(
            ref="forecast-input",
            kind="input",
            label='Local forecast by "City, St" or ZIP code',
            selector='css=input[id="inputstring"]',
            field_type="text",
            region="form",
        )

        self.assertTrue(is_search_like_input(item))

    def test_search_progress_prefers_search_entry_when_results_are_not_query_anchored(self) -> None:
        page_state = PageState(
            url="https://kubernetes.io/docs/home/",
            title="Kubernetes Documentation | Kubernetes",
            markdown="Kubernetes Documentation home page",
            interactables=[
                Interactable(
                    ref="search-input",
                    kind="input",
                    label="Search this site",
                    selector='css=input[type="search"]',
                    field_type="search",
                    region="form",
                ),
                Interactable(
                    ref="cncf-link",
                    kind="link",
                    label="CNCF",
                    selector='css=a[href="https://www.cncf.io/about"]',
                    href="https://www.cncf.io/about",
                    region="main",
                    context_text="Kubernetes Documentation Kubernetes is an open source container platform",
                ),
                Interactable(
                    ref="components-link",
                    kind="link",
                    label="Components of a cluster",
                    selector='css=a[href="/docs/concepts/overview/components/"]',
                    href="/docs/concepts/overview/components/",
                    region="main",
                    context_text="Components of a cluster",
                ),
            ],
            page_archetype="search_results",
            page_hints=["result_links", "search_input", "body_links"],
        )

        progress = search_progress_state(
            page_state,
            "Search the Kubernetes documentation for taints and tolerations, open the relevant page, and report the page title.",
        )

        self.assertEqual(progress, "search_entry")

    def test_fallback_page_analysis_prefers_search_input_over_generic_body_links(self) -> None:
        page_state = PageState(
            url="https://kubernetes.io/docs/home/",
            title="Kubernetes Documentation | Kubernetes",
            markdown="Kubernetes Documentation home page",
            interactables=[
                Interactable(
                    ref="search-input",
                    kind="input",
                    label="Search this site",
                    selector='css=input[type="search"]',
                    field_type="search",
                    region="form",
                ),
                Interactable(
                    ref="cncf-link",
                    kind="link",
                    label="CNCF",
                    selector='css=a[href="https://www.cncf.io/about"]',
                    href="https://www.cncf.io/about",
                    region="main",
                    context_text="Kubernetes Documentation Kubernetes is an open source container platform",
                ),
            ],
            page_archetype="search_results",
            page_hints=["result_links", "search_input", "body_links"],
        )

        result = fallback_page_analysis(
            page_state=page_state,
            target_prompt=(
                "Search the Kubernetes documentation for taints and tolerations, "
                "open the relevant page, and report the page title."
            ),
            question="What is the strongest next action on this page?",
        )

        self.assertEqual(result.recommended_action, "type_and_submit")
        self.assertEqual(result.recommended_interactable_ref, "search-input")

    def test_destination_page_prompt_is_treated_as_needing_progression(self) -> None:
        prompt = "Search for Nintendo DS and report the destination page title."

        self.assertTrue(task_requires_destination_page(prompt))

    def test_best_result_link_prefers_same_site_result_over_search_branding(self) -> None:
        page_state = PageState(
            url="https://kubernetes.io/search/?q=taints%20and%20tolerations",
            title="Search Results | Kubernetes",
            markdown="Search Results",
            interactables=[
                Interactable(
                    ref="search-on-google",
                    kind="link",
                    label='Search "taints and tolerations" on Google',
                    selector='css=a[href*="google.com/search"]',
                    href="https://www.google.com/search?client=ms-google-coop&q=taints+and+tolerations",
                    region="main",
                    context_text="Taints and Tolerations | Kubernetes Kubernetes concepts scheduling eviction",
                ),
                Interactable(
                    ref="cse-brand",
                    kind="link",
                    label="link_64",
                    selector='css=a[href="https://cse.google.com/?ref=b&hl=en"]',
                    href="https://cse.google.com/?ref=b&hl=en",
                    region="main",
                    context_text="Taints and Tolerations | Kubernetes Kubernetes concepts scheduling eviction",
                ),
                Interactable(
                    ref="real-result",
                    kind="link",
                    label="Taints and Tolerations | Kubernetes",
                    selector='role=link[name="Taints and Tolerations | Kubernetes"]',
                    href="https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/",
                    region="main",
                    context_text="Taints and Tolerations | Kubernetes Kubernetes concepts scheduling eviction",
                ),
            ],
            page_archetype="search_results",
            page_hints=["result_links"],
        )

        best = best_result_link(page_state, "taints and tolerations")

        self.assertIsNotNone(best)
        self.assertEqual(best.ref, "real-result")

    def test_best_result_link_ignores_pagination_like_links(self) -> None:
        page_state = PageState(
            url="https://github.com/search?q=langchain-ai%2Flangchain&type=repositories",
            title="Repository search results · GitHub",
            markdown="Repository search results",
            interactables=[
                Interactable(
                    ref="pagination-link",
                    kind="link",
                    label="100",
                    selector='css=a[href*="&p=100"]',
                    href="https://github.com/search?q=langchain-ai%2Flangchain&type=repositories&p=100",
                    region="main",
                    context_text="langchain-ai/langchain Repository search results GitHub",
                ),
                Interactable(
                    ref="repo-link",
                    kind="link",
                    label="langchain-ai/langchain",
                    selector='role=link[name="langchain-ai/langchain"]',
                    href="https://github.com/langchain-ai/langchain",
                    region="main",
                    context_text="langchain-ai/langchain LangChain Python framework",
                ),
            ],
            page_archetype="search_results",
            page_hints=["result_links"],
        )

        best = best_result_link(page_state, "langchain-ai/langchain")

        self.assertIsNotNone(best)
        self.assertEqual(best.ref, "repo-link")

    def test_canonical_result_url_normalizes_subpage_to_requested_slug(self) -> None:
        canonical = canonical_result_url(
            "https://github.com/search?q=langchain-ai%2Flangchain&type=repositories",
            "/langchain-ai/langchain/stargazers",
            "langchain-ai/langchain",
        )

        self.assertEqual(canonical, "https://github.com/langchain-ai/langchain")

    def test_search_progress_recognizes_modal_result_lists_from_query_relevant_links(self) -> None:
        page_state = PageState(
            url="https://playwright.dev/",
            title="Fast and reliable end-to-end testing for modern web apps | Playwright",
            markdown="Search docs",
            interactables=[
                Interactable(
                    ref="search-input",
                    kind="input",
                    label="Search docs",
                    selector='css=input[id="docsearch-input"]',
                    field_type="search",
                    region="form",
                ),
                Interactable(
                    ref="locators-overview",
                    kind="link",
                    label="Locators",
                    selector='role=link[name="Locators"]',
                    href="/docs/locators",
                    region="body",
                    context_text="Locators Playwright Docs",
                ),
                Interactable(
                    ref="locators-filtering",
                    kind="link",
                    label="Filtering Locators",
                    selector='role=link[name="Filtering Locators"]',
                    href="/docs/locators#filtering-locators",
                    region="body",
                    context_text="Filtering Locators Playwright Docs",
                ),
            ],
            page_archetype="generic",
        )

        progress = search_progress_state(
            page_state,
            "Go to Playwright docs and search for locators. Report the page title.",
        )

        self.assertEqual(progress, "results_list")


if __name__ == "__main__":
    unittest.main()
