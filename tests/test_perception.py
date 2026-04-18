from __future__ import annotations

import unittest

from agent.models import Interactable, PageState
from agent.perception import classify_page_state, enrich_page_state


class PerceptionTest(unittest.TestCase):
    def test_classifies_product_detail_pages(self) -> None:
        state = PageState(
            url="https://www.apple.com/shop/buy-iphone/iphone-17",
            title="Buy iPhone 17 - Apple",
            markdown="iPhone 17\nFrom $799 or $33.29/mo. Buy. Trade in available.",
            interactables=[
                Interactable(
                    kind="button",
                    label="Buy",
                    selector='role=button[name="Buy"]',
                    region="main",
                    context_text="iPhone 17 From $799 Buy",
                ),
                Interactable(
                    kind="link",
                    label="iPhone 17",
                    selector='role=link[name="iPhone 17"]',
                    href="/shop/buy-iphone/iphone-17",
                    region="main",
                    context_text="iPhone 17 From $799 Buy",
                ),
            ],
        )

        archetype, hints = classify_page_state(state)

        self.assertEqual(archetype, "product_detail")
        self.assertIn("price_signals", hints)
        self.assertIn("commerce_signals", hints)

    def test_classifies_commerce_hub_pages_from_buy_links_as_results(self) -> None:
        state = PageState(
            url="https://www.apple.com/iphone/",
            title="iPhone - Apple",
            markdown="iPhone lineup and shop links",
            interactables=[
                Interactable(
                    kind="link",
                    label="Buy",
                    selector='role=link[name="Buy"] >> nth=0',
                    href="/us/shop/goto/buy_iphone/iphone_17",
                    region="main",
                    context_text="iPhone 17 Buy",
                ),
                Interactable(
                    kind="link",
                    label="Buy",
                    selector='role=link[name="Buy"] >> nth=1',
                    href="/us/shop/goto/buy_iphone/iphone_air",
                    region="main",
                    context_text="iPhone Air Buy",
                ),
            ],
        )

        archetype, hints = classify_page_state(state)

        self.assertEqual(archetype, "search_results")
        self.assertIn("commerce_links", hints)
        self.assertIn("result_links", hints)

    def test_classifies_search_result_pages(self) -> None:
        state = PageState(
            url="https://craigslist.org/search/vga?query=nintendo+ds",
            title="nintendo ds - craigslist",
            markdown="Search results for Nintendo DS",
            interactables=[
                Interactable(
                    kind="input",
                    label="Search",
                    selector='css=input[type="search"]',
                    field_type="search",
                    region="form",
                    context_text="Search Craigslist",
                ),
                Interactable(
                    kind="link",
                    label="Nintendo DS Lite",
                    selector='role=link[name="Nintendo DS Lite"]',
                    href="/post/1",
                    region="main",
                    context_text="Nintendo DS Lite $80",
                ),
                Interactable(
                    kind="link",
                    label="Nintendo DS Bundle",
                    selector='role=link[name="Nintendo DS Bundle"]',
                    href="/post/2",
                    region="main",
                    context_text="Nintendo DS Bundle $120",
                ),
                Interactable(
                    kind="link",
                    label="Help",
                    selector='role=link[name="Help"]',
                    href="/help",
                    region="nav",
                    context_text="Craigslist Help",
                ),
                Interactable(
                    kind="link",
                    label="Terms",
                    selector='role=link[name="Terms"]',
                    href="/terms",
                    region="footer",
                    context_text="Terms",
                ),
                Interactable(
                    kind="link",
                    label="Nintendo DS Charger",
                    selector='role=link[name="Nintendo DS Charger"]',
                    href="/post/3",
                    region="main",
                    context_text="Nintendo DS Charger $15",
                ),
                Interactable(
                    kind="link",
                    label="Nintendo DS Case",
                    selector='role=link[name="Nintendo DS Case"]',
                    href="/post/4",
                    region="main",
                    context_text="Nintendo DS Case $10",
                ),
            ],
        )

        enriched = enrich_page_state(state)

        self.assertEqual(enriched.page_archetype, "search_results")
        self.assertIn("result_links", enriched.page_hints)
        self.assertIn("search_input", enriched.page_hints)

    def test_does_not_misclassify_commerce_page_as_travel(self) -> None:
        state = PageState(
            url="https://www.apple.com/shop/buy-iphone/iphone-17",
            title="Buy iPhone 17 - Apple",
            markdown=(
                "iPhone 17\n"
                "From $799 or $33.29/mo.\n"
                "Choose your finish. Choose your storage.\n"
                "Buy with monthly payments or pay in full."
            ),
            interactables=[
                Interactable(
                    kind="select",
                    label="Finish",
                    selector='css=select[name="finish"]',
                    region="main",
                    options=["Blue", "Black"],
                    context_text="Choose your finish",
                ),
                Interactable(
                    kind="select",
                    label="Storage",
                    selector='css=select[name="storage"]',
                    region="main",
                    options=["128GB", "256GB"],
                    context_text="Choose your storage",
                ),
                Interactable(
                    kind="button",
                    label="Buy",
                    selector='role=button[name="Buy"]',
                    region="main",
                    context_text="From $799 Buy",
                ),
            ],
        )

        archetype, hints = classify_page_state(state)

        self.assertEqual(archetype, "product_detail")
        self.assertNotEqual(archetype, "travel_search")
        self.assertIn("price_signals", hints)

    def test_homepage_is_not_treated_as_product_detail(self) -> None:
        state = PageState(
            url="https://www.apple.com/",
            title="Apple",
            markdown="iPhone 17 Buy MacBook Air Buy Apple Watch Buy",
            interactables=[
                Interactable(
                    kind="link",
                    label="Shop iPhone",
                    selector='role=link[name="Shop iPhone"]',
                    href="/shop/buy-iphone",
                    region="main",
                    context_text="iPhone 17 Buy",
                ),
                Interactable(
                    kind="link",
                    label="Shop MacBook Air",
                    selector='role=link[name="Shop MacBook Air"]',
                    href="/macbook-air",
                    region="main",
                    context_text="MacBook Air Buy",
                ),
                Interactable(
                    kind="link",
                    label="Shop Apple Watch",
                    selector='role=link[name="Shop Apple Watch"]',
                    href="/apple-watch",
                    region="main",
                    context_text="Apple Watch Buy",
                ),
            ],
        )

        archetype, _ = classify_page_state(state)

        self.assertNotEqual(archetype, "product_detail")

    def test_requires_stronger_travel_signals_than_from_and_to(self) -> None:
        state = PageState(
            url="https://houston.craigslist.org/",
            title="craigslist houston",
            markdown="Search all categories from community to for sale.",
            interactables=[
                Interactable(
                    kind="input",
                    label="Search craigslist",
                    selector='css=input[type="search"]',
                    field_type="search",
                    region="form",
                    context_text="Search craigslist",
                ),
                Interactable(
                    kind="select",
                    label="Category",
                    selector='css=select[name="catAbb"]',
                    region="form",
                    options=["for sale", "jobs"],
                    context_text="Category",
                ),
                Interactable(
                    kind="link",
                    label="For sale",
                    selector='role=link[name="for sale"]',
                    href="/search/sss",
                    region="main",
                    context_text="Browse for sale listings",
                ),
                Interactable(
                    kind="link",
                    label="Electronics",
                    selector='role=link[name="electronics"]',
                    href="/search/ela",
                    region="main",
                    context_text="Electronics listings",
                ),
                Interactable(
                    kind="link",
                    label="Video gaming",
                    selector='role=link[name="video gaming"]',
                    href="/search/vga",
                    region="main",
                    context_text="Video gaming listings",
                ),
                Interactable(
                    kind="link",
                    label="Help",
                    selector='role=link[name="help"]',
                    href="/help",
                    region="footer",
                    context_text="Help",
                ),
                Interactable(
                    kind="link",
                    label="Terms",
                    selector='role=link[name="terms"]',
                    href="/terms",
                    region="footer",
                    context_text="Terms",
                ),
            ],
        )

        archetype, _ = classify_page_state(state)

        self.assertNotEqual(archetype, "travel_search")

    def test_docs_article_with_search_box_is_not_misclassified_as_search_results(self) -> None:
        state = PageState(
            url="https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/",
            title="Taints and Tolerations | Kubernetes",
            markdown=(
                "Taints and Tolerations\n"
                "Taints and tolerations work together to ensure that pods are not scheduled onto inappropriate nodes.\n"
                "Read about device taints and tolerations.\n"
            ),
            interactables=[
                Interactable(
                    kind="input",
                    label="Search this site",
                    selector='css=input[type="search"]',
                    field_type="search",
                    region="form",
                ),
                Interactable(
                    kind="link",
                    label="device taints and tolerations",
                    selector='role=link[name="device taints and tolerations"]',
                    href="/docs/concepts/scheduling-eviction/dynamic-resource-allocation/#device-taints-and-tolerations",
                    region="main",
                    context_text="Read about device taints and tolerations",
                ),
                Interactable(
                    kind="link",
                    label="kubectl taint",
                    selector='role=link[name="kubectl taint"]',
                    href="/docs/reference/generated/kubectl/kubectl-commands#taint",
                    region="main",
                    context_text="You add a taint to a node using kubectl taint",
                ),
                Interactable(
                    kind="link",
                    label="DaemonSet",
                    selector='role=link[name="DaemonSet"]',
                    href="/docs/concepts/workloads/controllers/daemonset/",
                    region="main",
                    context_text="DaemonSet Pods can be scheduled onto nodes with process pressure issues",
                ),
                Interactable(
                    kind="link",
                    label="Documentation",
                    selector='role=link[name="Documentation"]',
                    href="/docs/home/",
                    region="nav",
                    context_text="Documentation",
                ),
            ],
        )

        archetype, hints = classify_page_state(state)

        self.assertEqual(archetype, "generic")
        self.assertIn("search_input", hints)
        self.assertIn("body_links", hints)

    def test_classifies_travel_pages_with_strong_travel_signals(self) -> None:
        state = PageState(
            url="https://www.booking.com/searchresults.en-us.html",
            title="Search flights",
            markdown="Search flights. Depart. Return. Passengers. Airport.",
            interactables=[
                Interactable(
                    kind="input",
                    label="From",
                    selector='css=input[name="origin"]',
                    field_type="search",
                    region="form",
                    context_text="Airport origin",
                ),
                Interactable(
                    kind="input",
                    label="To",
                    selector='css=input[name="destination"]',
                    field_type="search",
                    region="form",
                    context_text="Airport destination",
                ),
                Interactable(
                    kind="input",
                    label="Depart date",
                    selector='css=input[name="depart"]',
                    field_type="date",
                    region="form",
                    context_text="Depart date",
                ),
                Interactable(
                    kind="select",
                    label="Passengers",
                    selector='css=select[name="passengers"]',
                    region="form",
                    options=["1", "2", "3"],
                    context_text="Passengers",
                ),
            ],
        )

        archetype, hints = classify_page_state(state)

        self.assertEqual(archetype, "travel_search")
        self.assertIn("travel_form", hints)

    def test_classifies_quote_pages(self) -> None:
        state = PageState(
            url="https://finance.yahoo.com/quote/AAPL",
            title="AAPL Stock Price, News, Quote & History",
            markdown="AAPL Apple Inc. 213.44 Previous Close 212.10 Open 213.02 Market Cap",
            interactables=[],
        )

        archetype, hints = classify_page_state(state)

        self.assertEqual(archetype, "quote")
        self.assertIn("quote_module", hints)

    def test_classifies_generic_product_pages_without_brand_tokens(self) -> None:
        state = PageState(
            url="https://shop.example.com/products/widget-3000",
            title="Widget 3000",
            markdown="Widget 3000. Starting at $79. Buy now. Monthly payments available.",
            interactables=[
                Interactable(
                    kind="button",
                    label="Buy now",
                    selector='role=button[name="Buy now"]',
                    region="main",
                    context_text="Widget 3000 Starting at $79 Buy now",
                ),
                Interactable(
                    kind="select",
                    label="Color",
                    selector='css=select[name="color"]',
                    region="main",
                    options=["Blue", "Gray"],
                    context_text="Choose a color",
                ),
            ],
        )

        archetype, hints = classify_page_state(state)

        self.assertEqual(archetype, "product_detail")
        self.assertIn("commerce_signals", hints)
        self.assertIn("price_signals", hints)

    def test_classifies_generic_quote_pages_without_domain_special_case(self) -> None:
        state = PageState(
            url="https://quotes.example.com/stocks/acme",
            title="ACME Quote",
            markdown=(
                "ACME 213.44 Previous Close 212.10 Open 213.02 Day Range 210.00 - 215.00 "
                "Market Cap 12B Volume 8.2M"
            ),
            interactables=[],
        )

        archetype, hints = classify_page_state(state)

        self.assertEqual(archetype, "quote")
        self.assertIn("quote_module", hints)


if __name__ == "__main__":
    unittest.main()
