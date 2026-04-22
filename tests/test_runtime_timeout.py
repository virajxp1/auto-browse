from __future__ import annotations

import asyncio
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.models import AgentResult
from agent.run import run_agent


class _DummyBrowser:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _DummyPlaywright:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class _DummyPage:
    url = "https://example.com"


class _SlowGraph:
    async def ainvoke(self, _initial_state):
        await asyncio.sleep(1.1)
        return {}


class _FastGraph:
    async def ainvoke(self, _initial_state):
        return {
            "result": AgentResult(
                answer="ok",
                source_url="https://example.com",
                evidence="ok",
                confidence=0.9,
                trace=[],
            )
        }


class RuntimeTimeoutTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_agent_enforces_max_runtime_seconds(self) -> None:
        browser = _DummyBrowser()
        playwright = _DummyPlaywright()

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(playwright, browser, _DummyPage())),
            ),
            patch("agent.run._build_graph", return_value=_SlowGraph()),
        ):
            with self.assertRaises(asyncio.TimeoutError):
                await run_agent(
                    openrouter_client=object(),  # type: ignore[arg-type]
                    start_url="https://example.com",
                    target_prompt="Slow task",
                    max_runtime_seconds=1,
                )

        self.assertTrue(browser.closed)
        self.assertTrue(playwright.stopped)

    async def test_run_agent_roots_execution_in_braintrust_span(self) -> None:
        browser = _DummyBrowser()
        playwright = _DummyPlaywright()
        start_span_calls: list[dict[str, object]] = []

        @contextmanager
        def fake_start_span(name: str, **kwargs: object):
            start_span_calls.append({"name": name, **kwargs})
            yield SimpleNamespace()

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(playwright, browser, _DummyPage())),
            ),
            patch("agent.run._build_graph", return_value=_FastGraph()),
            patch("agent.run.export_current_span_parent", return_value="braintrust-parent"),
            patch("agent.run.start_span", side_effect=fake_start_span),
            patch("agent.run.span_log"),
            patch("agent.run.flush") as mock_flush,
        ):
            result = await run_agent(
                openrouter_client=object(),  # type: ignore[arg-type]
                start_url="https://example.com",
                target_prompt="Quick task",
                trace_id="trace-123",
            )

        self.assertIsNone(result.error)
        self.assertEqual(len(start_span_calls), 2)
        self.assertEqual(start_span_calls[0]["name"], "agent.run")
        self.assertEqual(start_span_calls[0]["parent"], "braintrust-parent")
        self.assertEqual(start_span_calls[0]["metadata"]["trace_id"], "trace-123")
        self.assertEqual(start_span_calls[1]["name"], "startup.browser")
        mock_flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()
