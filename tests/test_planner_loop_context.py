from __future__ import annotations

import unittest

from agent.models import AgentDecision, AgentStepTrace, PageState
from agent.planner import build_prompt


class PlannerLoopContextTest(unittest.TestCase):
    def test_prompt_includes_detailed_recent_trace(self) -> None:
        state = PageState(
            url="https://example.com/search",
            title="Example Search",
            markdown="Example content",
            interactables=[],
        )
        history = [
            AgentStepTrace(
                step=2,
                url="https://example.com/search",
                title="Example Search",
                decision=AgentDecision(
                    action="navigate",
                    url="https://example.com/results",
                    step_summary="Navigate to results page.",
                    next_step="Extract answer from results.",
                ),
            )
        ]

        prompt = build_prompt(state, "find release date", history=history)

        self.assertIn("RECENT TRACE:", prompt)
        self.assertIn("page_url=https://example.com/search", prompt)
        self.assertIn("action=navigate", prompt)
        self.assertIn("url=https://example.com/results", prompt)

    def test_prompt_emits_loop_alerts_for_repeated_attempts(self) -> None:
        state = PageState(
            url="https://example.com/search",
            title="Example Search",
            markdown="Example content",
            interactables=[],
        )
        repeated = AgentDecision(
            action="type_and_submit",
            selector="css=input[type='text'] >> nth=0",
            text="python 3.14 release date",
            step_summary="Search for release date",
            next_step="Read results",
        )
        history = [
            AgentStepTrace(
                step=0,
                url="https://example.com/search",
                title="Example Search",
                decision=repeated,
            ),
            AgentStepTrace(
                step=1,
                url="https://example.com/search",
                title="Example Search",
                decision=repeated,
            ),
        ]

        prompt = build_prompt(state, "find release date", history=history)

        self.assertIn("LOOP ALERTS:", prompt)
        self.assertIn("repeated:", prompt)
        self.assertIn("do_not_repeat", prompt)

    def test_prompt_emits_blocker_alerts(self) -> None:
        state = PageState(
            url="https://www.google.com/sorry/index",
            title="Access denied",
            markdown="Enable JavaScript to continue. Unusual traffic detected.",
            interactables=[],
        )

        prompt = build_prompt(state, "find release date", history=None)

        self.assertIn("BLOCKER ALERTS:", prompt)
        self.assertIn("detected:", prompt)
        self.assertIn("blocked_guidance", prompt)

    def test_prompt_detects_google_sorry_from_url(self) -> None:
        state = PageState(
            url="https://www.google.com/sorry/index",
            title="Continue",
            markdown="Please continue",
            interactables=[],
        )

        prompt = build_prompt(state, "find release date", history=None)

        self.assertIn("google_sorry", prompt)
