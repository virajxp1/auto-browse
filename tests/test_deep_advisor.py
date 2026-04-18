from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from agent.deep_advisor import PageAnalysisResult, analyze_page_with_deep_agents, fallback_page_analysis
from agent.models import Interactable, PageState


class DeepAdvisorTest(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_page_uses_filesystem_skills(self) -> None:
        advisor = AsyncMock()
        advisor.ainvoke.return_value = {
            "structured_response": PageAnalysisResult(
                summary="The main result card is visible.",
                recommended_action="click",
                recommended_selector='role=link[name="Nintendo DS"]',
                completion_signals=[],
                confidence=0.81,
            )
        }

        with patch("agent.deep_advisor.create_deep_agent", return_value=advisor) as create_agent:
            result = await analyze_page_with_deep_agents(
                model=object(),
                page_state=PageState(
                    url="https://example.com/results",
                    title="Results",
                    markdown="Nintendo DS results",
                    interactables=[],
                    page_archetype="search_results",
                    page_hints=["result_links"],
                ),
                target_prompt="Open the first Nintendo DS listing.",
                question="What is the best next action?",
            )

        self.assertEqual(result.recommended_action, "click")
        kwargs = create_agent.call_args.kwargs
        self.assertEqual(kwargs["skills"], ["/agent/skills/"])
        self.assertIsNotNone(kwargs["backend"])
        self.assertTrue(any(spec.get("skills") == ["/agent/skills/"] for spec in kwargs["subagents"]))

    def test_fallback_page_analysis_prefers_best_matching_result_link(self) -> None:
        page_state = PageState(
            url="https://example.com/results",
            title="Search results",
            markdown="Search results for Nintendo DS",
            interactables=[
                Interactable(
                    ref="el1",
                    kind="link",
                    label="Support",
                    selector="css=a >> nth=0",
                    href="/support",
                    region="main",
                    context_text="Support Home Help",
                ),
                Interactable(
                    ref="el2",
                    kind="link",
                    label="Nintendo DS console",
                    selector="css=a >> nth=1",
                    href="/products/nintendo-ds",
                    region="main",
                    context_text="Nintendo DS console retro handheld product details",
                ),
            ],
            page_archetype="search_results",
            page_hints=["result_links"],
        )

        result = fallback_page_analysis(
            page_state=page_state,
            target_prompt="Search for Nintendo DS and open the result page.",
            question="What is the strongest next action on this page?",
            error=RuntimeError("advisor unavailable"),
        )

        self.assertEqual(result.recommended_action, "navigate")
        self.assertEqual(result.recommended_interactable_ref, "el2")
        self.assertEqual(result.recommended_value, "/products/nintendo-ds")

    def test_fallback_page_analysis_uses_search_trigger_when_visible(self) -> None:
        page_state = PageState(
            url="https://docs.example.com/",
            title="Docs Home",
            markdown="Documentation home page",
            interactables=[
                Interactable(
                    ref="search-btn",
                    kind="button",
                    label="Search",
                    selector='css=button[id="search-modal-trigger"]',
                    region="header",
                )
            ],
        )

        result = fallback_page_analysis(
            page_state=page_state,
            target_prompt="Search for buildx and report the destination page title.",
            question="What is the strongest next action on this page?",
        )

        self.assertEqual(result.recommended_action, "click")
        self.assertEqual(result.recommended_interactable_ref, "search-btn")

    def test_fallback_page_analysis_prefers_search_trigger_over_assistant_input(self) -> None:
        page_state = PageState(
            url="https://docs.example.com/",
            title="Docs Home",
            markdown="Ask the docs assistant",
            interactables=[
                Interactable(
                    ref="assistant-input",
                    kind="input",
                    label="Ask anything about the docs",
                    selector='css=:is(input:not([type]), input[type="text"], input[type="search"])[placeholder="Ask anything about the docs"]',
                    field_type="text",
                    region="main",
                ),
                Interactable(
                    ref="search-btn",
                    kind="button",
                    label="Search",
                    selector='css=button[id="search-modal-trigger"]',
                    region="header",
                ),
            ],
        )

        result = fallback_page_analysis(
            page_state=page_state,
            target_prompt="Search for buildx and report the destination page title.",
            question="What is the strongest next action on this page?",
        )

        self.assertEqual(result.recommended_action, "click")
        self.assertEqual(result.recommended_interactable_ref, "search-btn")


if __name__ == "__main__":
    unittest.main()
