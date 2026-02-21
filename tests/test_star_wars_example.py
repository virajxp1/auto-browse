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
    def __init__(self, tool_call_responses):
        self._tool_call_responses = tool_call_responses
        self._index = 0
        self.invocations = []

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, _messages, **kwargs):
        self.invocations.append(kwargs)
        if self._index < len(self._tool_call_responses):
            tool_calls = self._tool_call_responses[self._index]
            self._index += 1
        else:
            tool_calls = self._tool_call_responses[-1]
        return AIMessage(
            content="",
            tool_calls=tool_calls,
        )


class _StubOpenRouterClient:
    def __init__(self, tool_call_responses):
        self._chat_model = _StubChatModel(tool_call_responses)

    def chat_model(self):
        return self._chat_model


class _DummyClickPage:
    def __init__(self) -> None:
        self.url = "https://example.com"
        self._title = "Example"
        self._dom_version = 0
        self.evaluate_payloads = []

    async def wait_for_selector(self, _selector: str, state: str = "visible", timeout: int = 0) -> None:
        return None

    async def click(self, _selector: str, timeout: int = 0) -> None:
        raise RuntimeError("native click failed")

    async def title(self) -> str:
        return self._title

    async def evaluate(self, _script: str, payload=None):
        if payload is None:
            return f"dom-{self._dom_version}"

        self.evaluate_payloads.append(payload)
        clicked = (
            isinstance(payload, dict)
            and payload.get("baseSelector") == "a[href]"
            and payload.get("nthIndex") == 0
        )
        if clicked:
            self.url = "https://example.com/result"
            self._dom_version += 1
        return clicked

    async def wait_for_timeout(self, _ms: int) -> None:
        return None


class _DummyNoEffectActionPage:
    def __init__(self) -> None:
        self.url = "https://example.com"
        self._title = "Example"
        self.last_goto_url: str | None = None

    async def title(self) -> str:
        return self._title

    async def wait_for_selector(self, _selector: str, state: str = "visible", timeout: int = 0) -> None:
        return None

    async def click(self, _selector: str, timeout: int = 0) -> None:
        return None

    async def focus(self, _selector: str) -> None:
        return None

    async def fill(self, _selector: str, _text: str) -> None:
        return None

    async def press(self, _selector: str, _key: str) -> None:
        return None

    async def goto(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 0) -> None:
        self.last_goto_url = url
        return None

    async def evaluate(self, _script: str, payload=None):
        if isinstance(payload, dict):
            return False
        return "dom-constant"

    async def wait_for_timeout(self, _ms: int) -> None:
        return None


def _click_tool_call(selector: str):
    return {
        "name": "click",
        "args": {
            "selector": selector,
            "step_summary": "Click the first result link.",
            "next_step": "Extract answer now.",
        },
        "id": "call_click_1",
        "type": "tool_call",
    }


def _type_and_submit_tool_call(selector: str, text: str):
    return {
        "name": "type_and_submit",
        "args": {
            "selector": selector,
            "text": text,
            "step_summary": "Type query and submit.",
            "next_step": "Wait for results.",
        },
        "id": "call_type_1",
        "type": "tool_call",
    }


def _navigate_tool_call(url: str):
    return {
        "name": "navigate",
        "args": {
            "url": url,
            "step_summary": "Navigate to requested URL.",
            "next_step": "Extract answer from destination.",
        },
        "id": "call_navigate_1",
        "type": "tool_call",
    }


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
                openrouter_client=_StubOpenRouterClient([[_extract_tool_call()]]),
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

    async def test_passes_trace_metadata_on_each_llm_invocation(self) -> None:
        state = PageState(
            url="https://example.com",
            title="Example",
            markdown="",
            interactables=[],
        )
        markdown = "# Example\n\nTest page\n"
        client = _StubOpenRouterClient([[_extract_tool_call()]])

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), object())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=markdown)),
        ):
            result = await run_agent(
                openrouter_client=client,
                start_url="https://example.com",
                target_prompt="release date",
                max_steps=3,
                trace_id="trace-123",
            )

        self.assertIsNone(result.error)
        self.assertEqual(len(client._chat_model.invocations), 1)
        invocation = client._chat_model.invocations[0]
        self.assertEqual(invocation["extra_body"]["session_id"], "trace-123")
        self.assertEqual(invocation["extra_body"]["trace"]["trace_id"], "trace-123")
        self.assertEqual(invocation["extra_body"]["trace"]["trace_name"], "auto_browse_agent_run")
        self.assertEqual(invocation["extra_body"]["trace"]["generation_name"], "planner.1")

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
                openrouter_client=_StubOpenRouterClient([[_extract_tool_call(), second_call]]),
                start_url="https://example.com",
                target_prompt="release date",
                max_steps=3,
                on_step=step_updates.append,
            )

        self.assertEqual(result.error, "llm_returned_multiple_tool_calls")
        self.assertEqual(len(result.trace), 0)
        self.assertEqual(len(step_updates), 0)

    async def test_retries_when_first_llm_response_has_no_tool_call(self) -> None:
        state = PageState(
            url="https://example.com",
            title="Example",
            markdown="",
            interactables=[],
        )
        markdown = "# Example\n\nRelease date May 25, 1977\n"
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
                openrouter_client=_StubOpenRouterClient([[], [_extract_tool_call()]]),
                start_url="https://example.com",
                target_prompt="release date",
                max_steps=3,
                on_step=step_updates.append,
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "May 25, 1977")
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(len(step_updates), 1)

    async def test_click_fallback_supports_css_nth_selector(self) -> None:
        state = PageState(
            url="https://example.com",
            title="Example",
            markdown="",
            interactables=[
                Interactable(
                    kind="link",
                    label="Result",
                    selector="css=a[href] >> nth=0",
                    href="/result",
                )
            ],
        )
        markdown = "# Example\n\nRelease date May 25, 1977\n"
        step_updates = []
        page = _DummyClickPage()

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [[_click_tool_call("css=a[href] >> nth=0")], [_extract_tool_call()]]
                ),
                start_url="https://example.com",
                target_prompt="release date",
                max_steps=4,
                on_step=step_updates.append,
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "May 25, 1977")
        self.assertEqual(len(step_updates), 2)
        self.assertEqual(page.evaluate_payloads[0]["baseSelector"], "a[href]")
        self.assertEqual(page.evaluate_payloads[0]["nthIndex"], 0)

    async def test_click_fallback_rejects_malformed_css_nth_selector(self) -> None:
        state = PageState(
            url="https://example.com",
            title="Example",
            markdown="",
            interactables=[
                Interactable(
                    kind="link",
                    label="Result",
                    selector="css=a[href] >> nth=oops",
                    href="/result",
                )
            ],
        )
        markdown = "# Example\n\nTest page\n"
        page = _DummyClickPage()
        step_updates = []

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient([[_click_tool_call("css=a[href] >> nth=oops")]]),
                start_url="https://example.com",
                target_prompt="test click",
                max_steps=2,
                on_step=step_updates.append,
            )

        self.assertEqual(result.error, "click_failed")
        self.assertEqual(len(step_updates), 1)
        self.assertEqual(page.evaluate_payloads, [])

    async def test_click_without_state_change_is_treated_as_failure(self) -> None:
        state = PageState(
            url="https://example.com",
            title="Example",
            markdown="",
            interactables=[
                Interactable(
                    kind="link",
                    label="Result",
                    selector="css=a[href] >> nth=0",
                    href="/result",
                )
            ],
        )
        markdown = "# Example\n\nTest page\n"
        page = _DummyNoEffectActionPage()
        step_updates = []

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient([[_click_tool_call("css=a[href] >> nth=0")]]),
                start_url="https://example.com",
                target_prompt="test click",
                max_steps=2,
                on_step=step_updates.append,
            )

        self.assertEqual(result.error, "click_failed")
        self.assertEqual(len(step_updates), 1)

    async def test_type_and_submit_without_state_change_is_treated_as_failure(self) -> None:
        state = PageState(
            url="https://example.com",
            title="Example",
            markdown="",
            interactables=[
                Interactable(
                    kind="input",
                    label="Search",
                    selector="css=input[type='text'] >> nth=0",
                    href=None,
                )
            ],
        )
        markdown = "# Example\n\nTest page\n"
        page = _DummyNoEffectActionPage()
        step_updates = []

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [[_type_and_submit_tool_call("css=input[type='text'] >> nth=0", "openai docs")]]
                ),
                start_url="https://example.com",
                target_prompt="search for docs",
                max_steps=2,
                on_step=step_updates.append,
            )

        self.assertEqual(result.error, "type_and_submit_failed")
        self.assertEqual(len(step_updates), 1)

    async def test_navigate_without_state_change_is_treated_as_failure(self) -> None:
        state = PageState(
            url="https://example.com",
            title="Example",
            markdown="",
            interactables=[],
        )
        markdown = "# Example\n\nTest page\n"
        page = _DummyNoEffectActionPage()
        step_updates = []

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [[_navigate_tool_call("https://example.com/next-page")]]
                ),
                start_url="https://example.com",
                target_prompt="navigate test",
                max_steps=2,
                on_step=step_updates.append,
            )

        self.assertEqual(result.error, "navigate_failed")
        self.assertEqual(len(step_updates), 1)
        self.assertEqual(page.last_goto_url, "https://example.com/next-page")
