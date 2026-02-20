from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from agent.models import Interactable, PageState
from agent.run import run_agent


class _DummyBrowser:
    async def close(self) -> None:
        return None


class _DummyPlaywright:
    async def stop(self) -> None:
        return None


class _StubChatModel:
    def __init__(self, tool_calls):
        self._tool_calls = tool_calls

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        return AIMessage(
            content="",
            tool_calls=self._tool_calls,
        )


class _StubOpenRouterClient:
    def __init__(self, tool_calls):
        self._tool_calls = tool_calls

    def chat_model(self):
        return _StubChatModel(self._tool_calls)


def _extract_tool_call():
    return {
        "name": "extract_answer",
        "args": {
            "answer": "May 25, 1977",
            "evidence": "Release date May 25, 1977",
            "confidence": 0.86,
            "step_summary": "The page includes the release date in the infobox.",
            "next_step": "Return extracted answer now.",
        },
        "id": "call_1",
        "type": "tool_call",
    }


class StarWarsExampleTest(unittest.IsolatedAsyncioTestCase):
    async def test_star_wars_release_date_example(self) -> None:
        state = PageState(
            url="https://en.wikipedia.org/wiki/Star_Wars_(film)",
            title="Star Wars (film) - Wikipedia",
            markdown="",
            interactables=[
                Interactable(
                    kind="link",
                    label="Release date",
                    selector="css=a[href] >> nth=0",
                    href="#Release",
                )
            ],
        )
        markdown = "# Star Wars\n\nRelease date May 25, 1977\n"
        step_updates = []

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), object())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient([_extract_tool_call()]),
                start_url="https://en.wikipedia.org/wiki/Star_Wars_(film)",
                target_prompt="release date of Star Wars",
                max_steps=3,
                on_step=step_updates.append,
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "May 25, 1977")
        self.assertEqual(result.source_url, "https://en.wikipedia.org/wiki/Star_Wars_(film)")
        self.assertEqual(result.evidence, "Release date May 25, 1977")
        self.assertAlmostEqual(result.confidence or 0.0, 0.86, places=2)
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(
            result.trace[0].decision.step_summary,
            "The page includes the release date in the infobox.",
        )
        self.assertEqual(result.trace[0].decision.next_step, "Return extracted answer now.")
        self.assertEqual(len(step_updates), 1)

    async def test_rejects_multiple_tool_calls_in_one_turn(self) -> None:
        state = PageState(
            url="https://example.com",
            title="Example",
            markdown="",
            interactables=[],
        )
        markdown = "# Example\n\nTest page\n"
        step_updates = []
        second_call = _extract_tool_call()
        second_call["id"] = "call_2"
        second_call["args"]["answer"] = "May 1977"

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), object())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient([_extract_tool_call(), second_call]),
                start_url="https://example.com",
                target_prompt="release date",
                max_steps=3,
                on_step=step_updates.append,
            )

        self.assertEqual(result.error, "llm_returned_multiple_tool_calls")
        self.assertEqual(len(result.trace), 0)
        self.assertEqual(len(step_updates), 0)
