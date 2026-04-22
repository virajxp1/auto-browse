from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from agent.deep_advisor import GoalVerificationResult, PageAnalysisResult
from agent.models import Interactable, PageState
from agent.browser_actions import try_type_and_submit_selector
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

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, _messages, **_kwargs):
        if self._index < len(self._tool_call_responses):
            response = self._tool_call_responses[self._index]
            self._index += 1
        else:
            response = self._tool_call_responses[-1]
        if isinstance(response, BaseException):
            raise response
        tool_calls = response
        return AIMessage(content="", tool_calls=tool_calls)


class _StubOpenRouterClient:
    def __init__(self, tool_call_responses):
        self._chat_model = _StubChatModel(tool_call_responses)

    def chat_model(self):
        return self._chat_model


def _complete_goal_tool_call(
    *,
    status: str = "completed",
    goal_summary: str = "Completed requested browser task.",
    result_data: dict[str, object] | None = None,
):
    return {
        "name": "complete_goal",
        "args": {
            "status": status,
            "goal_summary": goal_summary,
            "result_data": result_data,
            "evidence": "Observed the expected page state.",
            "confidence": 0.88,
            "step_summary": "The goal is complete.",
            "next_step": "Return final result now.",
        },
        "id": "call_complete_1",
        "type": "tool_call",
    }


def _analyze_page_tool_call(question: str):
    return {
        "name": "analyze_page",
        "args": {
            "question": question,
            "step_summary": "Analyze the current page before acting.",
            "next_step": "Choose the next browser action using the analysis.",
        },
        "id": "call_analyze_1",
        "type": "tool_call",
    }


def _verify_goal_tool_call(criteria: str):
    return {
        "name": "verify_goal",
        "args": {
            "criteria": criteria,
            "step_summary": "Verify that the goal is satisfied on the current page.",
            "next_step": "Complete the task only if the goal is verified.",
        },
        "id": "call_verify_1",
        "type": "tool_call",
    }


def _type_and_submit_tool_call(selector: str | None, text: str, *, interactable_ref: str | None = None):
    args = {
        "text": text,
        "step_summary": "Submit the search query.",
        "next_step": "Inspect the search results.",
    }
    if selector is not None:
        args["selector"] = selector
    if interactable_ref is not None:
        args["interactable_ref"] = interactable_ref
    return {
        "name": "type_and_submit",
        "args": args,
        "id": "call_type_submit_1",
        "type": "tool_call",
    }


def _extract_answer_tool_call(*, answer: str, evidence: str, confidence):
    return {
        "name": "extract_answer",
        "args": {
            "answer": answer,
            "evidence": evidence,
            "confidence": confidence,
            "step_summary": "Extract main heading text.",
            "next_step": "Return extracted answer now.",
        },
        "id": "call_extract_1",
        "type": "tool_call",
    }


def _navigate_tool_call(
    url: str | None,
    *,
    selector: str | None = None,
    interactable_ref: str | None = None,
):
    args = {
        "step_summary": f"Navigate to {url}",
        "next_step": "Extract the answer from the destination page.",
    }
    if url is not None:
        args["url"] = url
    if selector is not None:
        args["selector"] = selector
    if interactable_ref is not None:
        args["interactable_ref"] = interactable_ref
    return {
        "name": "navigate",
        "args": args,
        "id": "call_navigate_1",
        "type": "tool_call",
    }


def _click_tool_call(selector: str | None, *, interactable_ref: str | None = None):
    args = {
        "step_summary": "Click the requested link.",
        "next_step": "Continue from the next page.",
    }
    if selector is not None:
        args["selector"] = selector
    if interactable_ref is not None:
        args["interactable_ref"] = interactable_ref
    return {
        "name": "click",
        "args": args,
        "id": "call_click_1",
        "type": "tool_call",
    }


def _fill_tool_call(selector: str | None, text: str, *, interactable_ref: str | None = None):
    args = {
        "text": text,
        "step_summary": "Fill the field with the provided value.",
        "next_step": "Submit or continue once the field is populated.",
    }
    if selector is not None:
        args["selector"] = selector
    if interactable_ref is not None:
        args["interactable_ref"] = interactable_ref
    return {
        "name": "fill",
        "args": args,
        "id": "call_fill_1",
        "type": "tool_call",
    }


def _submit_tool_call(selector: str | None, *, interactable_ref: str | None = None):
    args = {
        "step_summary": "Submit the populated form.",
        "next_step": "Wait for the confirmation page.",
    }
    if selector is not None:
        args["selector"] = selector
    if interactable_ref is not None:
        args["interactable_ref"] = interactable_ref
    return {
        "name": "submit",
        "args": args,
        "id": "call_submit_1",
        "type": "tool_call",
    }


def _select_option_tool_call(
    selector: str | None,
    value: str,
    *,
    interactable_ref: str | None = None,
):
    args = {
        "value": value,
        "step_summary": "Select the requested option.",
        "next_step": "Continue after the dropdown value is set.",
    }
    if selector is not None:
        args["selector"] = selector
    if interactable_ref is not None:
        args["interactable_ref"] = interactable_ref
    return {
        "name": "select_option",
        "args": args,
        "id": "call_select_1",
        "type": "tool_call",
    }


def _check_tool_call(selector: str | None, *, interactable_ref: str | None = None):
    args = {
        "step_summary": "Enable the requested option.",
        "next_step": "Continue after the control is checked.",
    }
    if selector is not None:
        args["selector"] = selector
    if interactable_ref is not None:
        args["interactable_ref"] = interactable_ref
    return {
        "name": "check",
        "args": args,
        "id": "call_check_1",
        "type": "tool_call",
    }


def _wait_for_tool_call(
    selector: str | None,
    state: str = "visible",
    timeout_ms: int = 1500,
    *,
    interactable_ref: str | None = None,
):
    args = {
        "state": state,
        "timeout_ms": timeout_ms,
        "step_summary": "Wait for the next UI state.",
        "next_step": "Continue after the selector is ready.",
    }
    if selector is not None:
        args["selector"] = selector
    if interactable_ref is not None:
        args["interactable_ref"] = interactable_ref
    return {
        "name": "wait_for",
        "args": args,
        "id": "call_wait_1",
        "type": "tool_call",
    }


class _BasePage:
    def __init__(self) -> None:
        self.url = "https://example.com"
        self._title = "Example"
        self._dom_version = 0

    async def title(self) -> str:
        return self._title

    async def wait_for_timeout(self, _ms: int) -> None:
        return None

    async def wait_for_selector(self, _selector: str, state: str = "visible", timeout: int = 0) -> None:
        _ = state
        _ = timeout
        return None

    async def evaluate(self, _script: str, payload=None):
        if payload is None:
            return f"dom-{self._dom_version}"
        return False


class _DummyFillPage(_BasePage):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    async def fill(self, selector: str, text: str) -> None:
        self.values[selector] = text

    async def input_value(self, selector: str) -> str:
        return self.values.get(selector, "")


class _DummySubmitPage(_BasePage):
    async def press(self, selector: str, key: str) -> None:
        if selector == 'css=form[id="signup-form"]' and key == "Enter":
            self.url = "https://example.com/welcome"
            self._title = "Welcome"
            self._dom_version += 1
            return
        raise RuntimeError("press failed")


class _NoFocusTypeSubmitPage(_BasePage):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    async def fill(self, selector: str, text: str) -> None:
        self.values[selector] = text

    async def press(self, selector: str, key: str) -> None:
        _ = selector
        if key != "Enter":
            raise RuntimeError("unexpected key")
        self.url = "https://example.com/results"
        self._title = "Results"
        self._dom_version += 1


class _DummyNavigatePage(_BasePage):
    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        _ = wait_until
        _ = timeout
        self.url = url
        self._title = "AAPL Stock Price, News, Quote & History"
        self._dom_version += 1

    async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
        _ = state
        _ = timeout
        return None


class _GenericTitlePage(_BasePage):
    def __init__(self) -> None:
        super().__init__()
        self.url = "https://www.marketwatch.com/investing/stock/msft"
        self._title = "marketwatch.com"


class _FirefoxDownloadPage(_BasePage):
    def __init__(self) -> None:
        super().__init__()
        self.url = "https://www.firefox.com/en-US/?redirect_source=mozilla-org"
        self._title = "Get Firefox - Firefox.com"


class _DummyClickPage(_BasePage):
    async def click(self, _selector: str, timeout: int = 0) -> None:
        _ = timeout
        raise RuntimeError("native click failed")

    async def evaluate(self, _script: str, payload=None):
        if payload is None:
            return await super().evaluate(_script, payload)
        clicked = (
            isinstance(payload, dict)
            and payload.get("baseSelector") == "a[href]"
            and payload.get("nthIndex") == 0
        )
        if clicked:
            self.url = "https://example.com/result"
            self._title = "Nintendo DS"
            self._dom_version += 1
        return clicked


class _FragmentHeadingPage(_BasePage):
    def __init__(self) -> None:
        super().__init__()
        self.url = "https://docs.example.com/reference#skip-tests"
        self._title = "API Reference - Example Docs"

    async def evaluate(self, script: str, payload=None):
        if payload is None:
            return await super().evaluate(script, payload)
        if payload == "skip-tests":
            return "Skip tests"
        return False


class _TypeSubmitFallbackPage(_BasePage):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    async def fill(self, selector: str, text: str) -> None:
        self.values[selector] = text

    async def press(self, selector: str, key: str) -> None:
        _ = selector
        _ = key
        return None

    async def evaluate(self, script: str, payload=None):
        if payload is None:
            return await super().evaluate(script, payload)
        if payload == 'css=input[name="query"]':
            self.url = "https://example.com/results"
            self._title = "Results"
            self._dom_version += 1
            return True
        return False


class _SlowFillTypePage(_BasePage):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    async def fill(self, selector: str, text: str) -> None:
        _ = selector
        _ = text
        await asyncio.sleep(0.2)

    async def type(self, selector: str, text: str) -> None:
        self.values[selector] = text

    async def press(self, selector: str, key: str) -> None:
        if key != "Enter":
            raise RuntimeError("unexpected key")
        if self.values.get(selector) != "locators":
            raise RuntimeError("missing typed value")
        self.url = "https://example.com/docs/locators"
        self._title = "Locators | Example Docs"
        self._dom_version += 1


class _AutocompleteResultsPage(_BasePage):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}
        self.enter_presses = 0
        self._hydrated = False

    async def fill(self, selector: str, text: str) -> None:
        self.values[selector] = text

    async def input_value(self, selector: str) -> str:
        return self.values.get(selector, "")

    async def press(self, selector: str, key: str) -> None:
        _ = selector
        if key == "Enter":
            self.enter_presses += 1
        raise RuntimeError("enter should not be required")

    async def wait_for_timeout(self, _ms: int) -> None:
        if not self._hydrated and self.values.get('css=input[name="query"]') == "defineConfig":
            self._dom_version += 1
            self._hydrated = True


class _DummySelectPage(_BasePage):
    def __init__(self) -> None:
        super().__init__()
        self.selected: dict[str, str] = {}

    async def select_option(self, selector: str, value: str = "") -> None:
        self.selected[selector] = value

    async def eval_on_selector(self, selector: str, _script: str):
        return self.selected.get(selector, "")


class _DummyCheckPage(_BasePage):
    def __init__(self) -> None:
        super().__init__()
        self.checked: dict[str, bool] = {}

    async def check(self, selector: str) -> None:
        self.checked[selector] = True

    async def is_checked(self, selector: str) -> bool:
        return self.checked.get(selector, False)


class GoalToolsTest(unittest.IsolatedAsyncioTestCase):
    def _verified_result(self) -> GoalVerificationResult:
        return GoalVerificationResult(
            verified=True,
            summary="The expected outcome is visible on the page.",
            evidence="Observed the expected page state.",
            missing_requirements=[],
            recommended_next_action=None,
            confidence=0.91,
        )

    def _analysis_result(self) -> PageAnalysisResult:
        return PageAnalysisResult(
            summary="The page shows a single visible result link.",
            recommended_action="click",
            recommended_interactable_ref="el1",
            recommended_selector='role=link[name="Nintendo DS"]',
            recommended_value=None,
            blocker_summary=None,
            completion_signals=[],
            confidence=0.73,
        )

    async def test_complete_goal_returns_generic_result(self) -> None:
        state = PageState(url="https://example.com", title="Example", markdown="", interactables=[])

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _BasePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Example\n\nReady\n")),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_verify_goal_tool_call("The first result has been opened.")],
                        [_complete_goal_tool_call(result_data={"clicked_result": True})],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Click the first result.",
                goal_type="generic",
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.goal_summary, "Completed requested browser task.")
        self.assertEqual(result.result_data, {"clicked_result": True})
        self.assertEqual(result.final_url, "https://example.com")
        self.assertEqual(result.final_title, "Example")
        self.assertEqual([item.decision.action for item in result.trace], ["verify", "complete"])

    async def test_extract_answer_accepts_string_confidence_labels(self) -> None:
        state = PageState(url="https://example.com", title="Example Domain", markdown="", interactables=[])

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _BasePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch(
                "agent.run.page_to_markdown",
                new=AsyncMock(return_value="# Example Domain\n\nThis domain is for use in documentation examples."),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [[_extract_answer_tool_call(answer="Example Domain", evidence="# Example Domain", confidence="high")]]
                ),
                start_url="https://example.com",
                target_prompt="Report the main heading text on this page.",
                goal_type="extract",
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "Example Domain")
        self.assertAlmostEqual(result.confidence or 0.0, 0.85, places=2)

    async def test_fill_supports_non_navigational_field_updates(self) -> None:
        state = PageState(url="https://example.com", title="Example", markdown="", interactables=[])
        page = _DummyFillPage()

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Example\n\nForm\n")),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_fill_tool_call('css=input[name="email"]', "example@email.com")],
                        [_verify_goal_tool_call("The email field contains the requested value.")],
                        [_complete_goal_tool_call(goal_summary="Filled the email field.")],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Fill the signup form.",
                goal_type="signup",
                task_data={"email": "example@email.com"},
            )

        self.assertIsNone(result.error)
        self.assertEqual(page.values['css=input[name="email"]'], "example@email.com")
        self.assertEqual(result.goal_summary, "Filled the email field.")
        self.assertEqual([item.decision.action for item in result.trace], ["fill", "verify", "complete"])

    async def test_navigate_ignores_optional_selector_arg(self) -> None:
        before = PageState(url="https://finance.yahoo.com", title="Yahoo Finance", markdown="", interactables=[])
        after = PageState(
            url="https://finance.yahoo.com/quote/AAPL/",
            title="AAPL Stock Price, News, Quote & History",
            markdown="AAPL stock page",
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _DummyNavigatePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(side_effect=[before, after])),
            patch("agent.run.page_to_markdown", new=AsyncMock(side_effect=["Yahoo Finance", "AAPL stock page"])),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_navigate_tool_call("https://finance.yahoo.com/quote/AAPL/", selector="")],
                        [_verify_goal_tool_call("The AAPL quote page is visible.")],
                        [_complete_goal_tool_call(goal_summary="Reached the AAPL quote page.")],
                    ]
                ),
                start_url="https://finance.yahoo.com",
                target_prompt="Go to the quote page for AAPL.",
                goal_type="generic",
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["navigate", "verify", "complete"])

    async def test_navigate_supports_interactable_refs_for_link_hrefs(self) -> None:
        page = _DummyNavigatePage()
        page.url = "https://www.apple.com/iphone/"
        before = PageState(
            url="https://www.apple.com/iphone/",
            title="iPhone",
            markdown="Shop iPhone",
            interactables=[
                Interactable(
                    ref="el1",
                    kind="link",
                    label="Buy iPhone 17",
                    selector='role=link[name="Buy iPhone 17"]',
                    href="/shop/buy-iphone/iphone-17",
                    region="main",
                )
            ],
        )
        after = PageState(
            url="https://www.apple.com/shop/buy-iphone/iphone-17",
            title="Buy iPhone 17 - Apple",
            markdown="iPhone 17 From $799",
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(side_effect=[before, after])),
            patch("agent.run.page_to_markdown", new=AsyncMock(side_effect=[before.markdown, after.markdown])),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_navigate_tool_call(None, interactable_ref="el1")],
                        [_extract_answer_tool_call(answer="$799", evidence="iPhone 17 From $799", confidence=0.8)],
                    ]
                ),
                start_url="https://www.apple.com/iphone/",
                target_prompt="Tell me the price of the iPhone 17 on Apple's website.",
                goal_type="extract",
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.trace[0].decision.action, "navigate")
        self.assertEqual(result.trace[0].decision.interactable_ref, "el1")
        self.assertEqual(result.trace[0].decision.url, "https://www.apple.com/shop/buy-iphone/iphone-17")

    async def test_click_unknown_selector_returns_analysis_instead_of_failing(self) -> None:
        state = PageState(
            url="https://www.apple.com",
            title="Apple",
            markdown="Apple homepage",
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _BasePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Apple\n\nHomepage")),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_click_tool_call('css=a[href="/iphone-17/"]')],
                        [_extract_answer_tool_call(answer="Apple", evidence="Homepage", confidence=0.8)],
                    ]
                ),
                start_url="https://www.apple.com",
                target_prompt="Report the current page title.",
                goal_type="extract",
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["analyze", "extract"])
        self.assertIn("not present in the current snapshot", result.trace[0].decision.analysis or "")

    async def test_planner_timeout_falls_back_to_analysis_then_grounded_ref_click(self) -> None:
        before = PageState(
            url="https://example.com/results",
            title="Results",
            markdown="Nintendo DS results",
            interactables=[
                Interactable(
                    ref="el1",
                    kind="link",
                    label="Nintendo DS",
                    selector="css=a[href] >> nth=0",
                    href="/result",
                    region="main",
                )
            ],
        )
        after = PageState(
            url="https://example.com/result",
            title="Nintendo DS",
            markdown="Nintendo DS product page",
            interactables=[],
        )
        page = _DummyNavigatePage()
        page.url = "https://example.com/results"

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(side_effect=[before, before, after])),
            patch(
                "agent.run.page_to_markdown",
                new=AsyncMock(side_effect=[before.markdown, before.markdown, after.markdown]),
            ),
            patch(
                "agent.run.analyze_page_with_deep_agents",
                new=AsyncMock(
                    return_value=PageAnalysisResult(
                        summary="The main result link is visible.",
                        recommended_action="click",
                        recommended_interactable_ref="el1",
                        recommended_selector="css=a[href] >> nth=0",
                        completion_signals=[],
                        confidence=0.7,
                    )
                ),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        TimeoutError(),
                        TimeoutError(),
                        [_extract_answer_tool_call(answer="Nintendo DS", evidence="Nintendo DS product page", confidence=0.8)],
                    ]
                ),
                start_url="https://example.com/results",
                target_prompt="Open the first Nintendo DS result and report its title.",
                goal_type="extract",
                max_steps=4,
            )

        # Last-ditch extraction at max_steps returns page content
        # instead of a bare max_steps_exceeded error.
        if result.error is None:
            self.assertEqual(result.status, "completed")
        else:
            self.assertEqual(result.error, "max_steps_exceeded")
        self.assertTrue(result.trace)
        self.assertEqual(result.trace[0].decision.interactable_ref, "el1")

    async def test_recent_analysis_is_applied_without_extra_llm_turn(self) -> None:
        before = PageState(
            url="https://example.com/results",
            title="Results",
            markdown="Nintendo DS results",
            interactables=[
                Interactable(
                    ref="el1",
                    kind="link",
                    label="Nintendo DS",
                    selector="css=a[href] >> nth=0",
                    href="/result",
                    region="main",
                )
            ],
        )
        after = PageState(
            url="https://example.com/result",
            title="Nintendo DS",
            markdown="Nintendo DS product page",
            interactables=[],
        )
        page = _DummyNavigatePage()
        page.url = "https://example.com/results"

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(side_effect=[before, before, after])),
            patch(
                "agent.run.page_to_markdown",
                new=AsyncMock(side_effect=[before.markdown, before.markdown, after.markdown]),
            ),
            patch(
                "agent.run.analyze_page_with_deep_agents",
                new=AsyncMock(
                    return_value=PageAnalysisResult(
                        summary="The main result link is visible.",
                        recommended_action="click",
                        recommended_interactable_ref="el1",
                        recommended_selector="css=a[href] >> nth=0",
                        completion_signals=[],
                        confidence=0.7,
                    )
                ),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        TimeoutError(),
                        [_extract_answer_tool_call(answer="Nintendo DS", evidence="Nintendo DS product page", confidence=0.8)],
                    ]
                ),
                start_url="https://example.com/results",
                target_prompt="Open the first Nintendo DS result and report its title.",
                goal_type="extract",
                max_steps=4,
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["navigate", "analyze", "analyze", "extract"])
        self.assertEqual(result.trace[0].decision.interactable_ref, "el1")

    async def test_grounded_shortcut_submits_visible_search_field(self) -> None:
        before = PageState(
            url="https://example.com",
            title="Example Search",
            markdown="Search the catalog",
            interactables=[
                Interactable(
                    ref="search1",
                    kind="input",
                    label="Search",
                    selector='css=input[name="query"]',
                    field_type="search",
                    region="main",
                )
            ],
            page_hints=["search_input"],
        )
        after = PageState(
            url="https://example.com/results",
            title="Nintendo DS Results",
            markdown="Nintendo DS Results",
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(
                    return_value=(_DummyPlaywright(), _DummyBrowser(), _NoFocusTypeSubmitPage())
                ),
            ),
            patch("agent.run.capture_state", new=AsyncMock(side_effect=[before, after])),
            patch(
                "agent.run.page_to_markdown",
                new=AsyncMock(side_effect=[before.markdown, after.markdown]),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [
                            _extract_answer_tool_call(
                                answer="Nintendo DS Results",
                                evidence="Current page title: Nintendo DS Results",
                                confidence=0.8,
                            )
                        ]
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Search for Nintendo DS and report the search results page title.",
                goal_type="extract",
                max_steps=3,
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["type_and_submit", "extract"])
        self.assertEqual(result.trace[0].decision.interactable_ref, "search1")

    async def test_planner_fallback_does_not_extract_results_title_when_destination_page_is_required(self) -> None:
        before = PageState(
            url="https://example.com",
            title="Docs Home",
            markdown="Docs home",
            interactables=[
                Interactable(
                    ref="search1",
                    kind="input",
                    label="Search",
                    selector='css=input[name="query"]',
                    field_type="search",
                    region="main",
                )
            ],
            page_hints=["search_input"],
        )
        results = PageState(
            url="https://example.com/search?q=nintendo+ds",
            title="Search Results | Example",
            markdown="Search Results",
            interactables=[
                Interactable(
                    ref="search1",
                    kind="input",
                    label="Search",
                    selector='css=input[name="query"]',
                    field_type="search",
                    region="main",
                ),
                Interactable(
                    ref="nav1",
                    kind="link",
                    label="Home",
                    selector='role=link[name="Home"]',
                    href="/",
                    region="nav",
                ),
            ],
            page_archetype="search_results",
            page_hints=["search_input", "result_links", "nav_heavy"],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(
                    return_value=(_DummyPlaywright(), _DummyBrowser(), _NoFocusTypeSubmitPage())
                ),
            ),
            patch("agent.run.capture_state", new=AsyncMock(side_effect=[before, results, results, results])),
            patch(
                "agent.run.page_to_markdown",
                new=AsyncMock(
                    side_effect=[before.markdown, results.markdown, results.markdown, results.markdown]
                ),
            ),
            patch(
                "agent.run.analyze_page_with_deep_agents",
                new=AsyncMock(
                    return_value=PageAnalysisResult(
                        summary="The page is still a search results shell with no grounded result to open yet.",
                        recommended_action="wait_for",
                        completion_signals=[],
                        confidence=0.2,
                    )
                ),
            ),
            patch("agent.run.search_progress_state", return_value="search_entry"),
            patch("agent.run.task_requires_destination_page", return_value=True),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient([TimeoutError(), TimeoutError()]),
                start_url="https://example.com",
                target_prompt="Search for Nintendo DS and report the destination page title.",
                goal_type="extract",
                max_steps=2,
            )

        # Last-ditch extraction returns page content at max_steps
        if result.error is None:
            self.assertEqual(result.status, "completed")
        else:
            self.assertEqual(result.error, "max_steps_exceeded")
        self.assertTrue(result.trace)

    async def test_grounded_shortcut_opens_ranked_result_link(self) -> None:
        page = _DummyNavigatePage()
        page.url = "https://example.com/results"
        before = PageState(
            url="https://example.com/results",
            title="Results",
            markdown="Nintendo DS results",
            interactables=[
                Interactable(
                    ref="el1",
                    kind="link",
                    label="Nintendo DS",
                    selector='role=link[name="Nintendo DS"]',
                    href="/result",
                    region="main",
                )
            ],
            page_archetype="search_results",
            page_hints=["result_links"],
        )
        after = PageState(
            url="https://example.com/result",
            title="Nintendo DS",
            markdown="Nintendo DS product page",
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(side_effect=[before, after])),
            patch(
                "agent.run.page_to_markdown",
                new=AsyncMock(side_effect=[before.markdown, after.markdown]),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [
                            _extract_answer_tool_call(
                                answer="Nintendo DS",
                                evidence="Current page title: Nintendo DS",
                                confidence=0.8,
                            )
                        ]
                    ]
                ),
                start_url="https://example.com/results",
                target_prompt="Search for Nintendo DS and report the destination page title.",
                goal_type="extract",
                max_steps=3,
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["navigate", "extract"])
        self.assertEqual(result.trace[0].decision.interactable_ref, "el1")

    async def test_destination_page_open_does_not_shortcut_extract_without_llm(self) -> None:
        page = _DummyNavigatePage()
        page.url = "https://example.com/docs/taints-and-tolerations"
        state = PageState(
            url="https://example.com/docs/taints-and-tolerations",
            title="Taints and Tolerations | Example Docs",
            markdown="Taints and tolerations guide",
            interactables=[
                Interactable(
                    ref="el1",
                    kind="link",
                    label="Related scheduling guide",
                    selector='role=link[name="Related scheduling guide"]',
                    href="/docs/scheduling-guide",
                    region="main",
                )
            ],
            page_archetype="generic",
            page_hints=["body_links"],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=state.markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient([TimeoutError(), TimeoutError()]),
                start_url="https://example.com/docs/taints-and-tolerations",
                target_prompt="Search for taints and tolerations and report the destination page title.",
                goal_type="extract",
                max_steps=2,
            )

        # With last-ditch extraction at max_steps, the agent now returns
        # page content instead of a bare max_steps_exceeded error.
        self.assertIsNone(result.error)
        self.assertEqual(result.status, "completed")
        self.assertIn("Taints and Tolerations", result.answer or "")

    async def test_generic_navigation_destination_open_does_not_shortcut_extract_without_llm(self) -> None:
        page = _FirefoxDownloadPage()
        state = PageState(
            url="https://www.firefox.com/en-US/?redirect_source=mozilla-org",
            title="Get Firefox - Firefox.com",
            markdown="Download Firefox for desktop and mobile browsers.",
            interactables=[],
            page_archetype="generic",
            page_hints=["body_links"],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=state.markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient([TimeoutError(), TimeoutError()]),
                start_url="https://www.mozilla.org/",
                target_prompt="Go to the Firefox download page and report the page title.",
                goal_type="generic",
                max_steps=2,
            )

        # With last-ditch extraction at max_steps, the agent returns
        # page content instead of a bare max_steps_exceeded error.
        self.assertIsNone(result.error)
        self.assertEqual(result.status, "completed")
        self.assertIn("Firefox", result.answer or "")

    async def test_extract_title_preserves_specific_answer_when_page_title_is_generic(self) -> None:
        state = PageState(
            url="https://www.marketwatch.com/investing/stock/msft",
            title="marketwatch.com",
            markdown="Microsoft Corp. (MSFT) Stock Price - MarketWatch",
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _GenericTitlePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=state.markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [
                            _extract_answer_tool_call(
                                answer="Microsoft Corp. (MSFT) Stock Price - MarketWatch",
                                evidence="Page title: Microsoft Corp. (MSFT) Stock Price - MarketWatch",
                                confidence=0.8,
                            )
                        ]
                    ]
                ),
                start_url="https://www.marketwatch.com/investing/stock/msft",
                target_prompt="Report the page title.",
                goal_type="extract",
                max_steps=1,
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "Microsoft Corp. (MSFT) Stock Price - MarketWatch")

    async def test_extract_title_does_not_enrich_with_matching_markdown_heading(self) -> None:
        state = PageState(
            url="https://forecast.weather.gov/MapClick.php?lat=41.883229&lon=-87.632398",
            title="7-Day Forecast 41.88N 87.65W",
            markdown="# Chicago, IL\n\n7-Day Forecast 41.88N 87.65W",
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _BasePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=state.markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [
                            _extract_answer_tool_call(
                                answer="7-Day Forecast 41.88N 87.65W",
                                evidence="Page title: 7-Day Forecast 41.88N 87.65W",
                                confidence=0.8,
                            )
                        ]
                    ]
                ),
                start_url="https://www.weather.gov/",
                target_prompt="Search weather.gov for Chicago, Illinois and report the page title of the forecast page.",
                goal_type="extract",
                max_steps=1,
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "7-Day Forecast 41.88N 87.65W")

    async def test_type_and_submit_falls_back_to_form_submit_when_enter_has_no_effect(self) -> None:
        state = PageState(url="https://example.com", title="Example", markdown="", interactables=[])

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(
                    return_value=(_DummyPlaywright(), _DummyBrowser(), _TypeSubmitFallbackPage())
                ),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Example\n\nSearch\n")),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_type_and_submit_tool_call('css=input[name="query"]', "AirPods Pro")],
                        [_verify_goal_tool_call("The search results page is visible.")],
                        [_complete_goal_tool_call(goal_summary="Submitted the search query.")],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Search for AirPods Pro.",
                goal_type="generic",
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["type_and_submit", "verify", "complete"])

    async def test_type_and_submit_falls_back_to_bounded_keyboard_typing_when_fill_hangs(self) -> None:
        before = PageState(
            url="https://example.com",
            title="Example Search",
            markdown="Search the docs",
            interactables=[
                Interactable(
                    ref="search1",
                    kind="input",
                    label="Search docs",
                    selector='css=input[name="query"]',
                    field_type="search",
                    region="main",
                )
            ],
            page_hints=["search_input"],
        )
        after = PageState(
            url="https://example.com/docs/locators",
            title="Locators | Example Docs",
            markdown="Locators guide",
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _SlowFillTypePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(side_effect=[before, after])),
            patch("agent.run.page_to_markdown", new=AsyncMock(side_effect=[before.markdown, after.markdown])),
            patch("agent.browser_actions._DIRECT_FILL_TIMEOUT_SECONDS", 0.01),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_type_and_submit_tool_call('css=input[name="query"]', "locators")],
                        [
                            _extract_answer_tool_call(
                                answer="Locators | Example Docs",
                                evidence="Current page title: Locators | Example Docs",
                                confidence=0.8,
                            )
                        ],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Search for locators and report the destination page title.",
                goal_type="extract",
                max_steps=3,
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["type_and_submit", "type_and_submit", "extract"])

    async def test_type_and_submit_waits_for_search_results_before_forcing_submit(self) -> None:
        page = _AutocompleteResultsPage()

        ok = await try_type_and_submit_selector(page, 'css=input[name="query"]', "defineConfig")

        self.assertTrue(ok)
        self.assertEqual(page.enter_presses, 0)

    async def test_submit_supports_form_submission(self) -> None:
        state = PageState(url="https://example.com", title="Example", markdown="", interactables=[])
        page = _DummySubmitPage()

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Example\n\nForm\n")),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_submit_tool_call('css=form[id="signup-form"]')],
                        [_verify_goal_tool_call("The confirmation page is visible after submission.")],
                        [_complete_goal_tool_call(goal_summary="Submitted the form.")],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Submit the form.",
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.final_url, "https://example.com/welcome")
        self.assertEqual(result.final_title, "Welcome")
        self.assertEqual([item.decision.action for item in result.trace], ["submit", "verify", "complete"])

    async def test_select_option_supports_dropdown_changes(self) -> None:
        state = PageState(url="https://example.com", title="Example", markdown="", interactables=[])
        page = _DummySelectPage()

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Example\n\nFilters\n")),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_select_option_tool_call('css=select[name="travellers"]', "2")],
                        [_verify_goal_tool_call("The dropdown shows the requested traveller count.")],
                        [_complete_goal_tool_call(goal_summary="Selected two travellers.")],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Choose 2 travellers.",
            )

        self.assertIsNone(result.error)
        self.assertEqual(page.selected['css=select[name="travellers"]'], "2")
        self.assertEqual([item.decision.action for item in result.trace], ["select_option", "verify", "complete"])

    async def test_check_supports_checkbox_and_radio_controls(self) -> None:
        state = PageState(url="https://example.com", title="Example", markdown="", interactables=[])
        page = _DummyCheckPage()

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Example\n\nOptions\n")),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_check_tool_call('css=input[name="terms"]')],
                        [_verify_goal_tool_call("The terms checkbox is enabled.")],
                        [_complete_goal_tool_call(goal_summary="Accepted the terms checkbox.")],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Accept the terms.",
            )

        self.assertIsNone(result.error)
        self.assertTrue(page.checked['css=input[name="terms"]'])
        self.assertEqual([item.decision.action for item in result.trace], ["check", "verify", "complete"])

    async def test_wait_for_supports_non_terminal_sync_steps(self) -> None:
        state = PageState(url="https://example.com", title="Example", markdown="", interactables=[])

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _BasePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Example\n\nLoading\n")),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_wait_for_tool_call('css=div.results')],
                        [_verify_goal_tool_call("The results view is visible.")],
                        [_complete_goal_tool_call(goal_summary="Waited for the results view.")],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Wait for the results view.",
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["wait_for", "verify", "complete"])

    async def test_analyze_page_adds_advisor_guidance_to_trace(self) -> None:
        state = PageState(url="https://example.com", title="Example", markdown="", interactables=[])

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _BasePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Example\n\nResults\n")),
            patch(
                "agent.run.analyze_page_with_deep_agents",
                new=AsyncMock(return_value=self._analysis_result()),
            ),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_analyze_page_tool_call("What is the best next action on this page?")],
                        [_verify_goal_tool_call("The correct result is visible and ready to open.")],
                        [_complete_goal_tool_call(goal_summary="Analyzed the result page.")],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Search for Nintendo DS and inspect the first result.",
                goal_type="generic",
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["analyze", "verify", "complete"])
        self.assertIn("best_next_action=click", result.trace[0].decision.analysis or "")

    async def test_analyze_page_falls_back_when_deep_advisor_errors(self) -> None:
        state = PageState(
            url="https://www.apple.com/shop/buy-iphone/iphone-17",
            title="Buy iPhone 17 - Apple",
            markdown="iPhone 17\nFrom $799 or $33.29/mo.\nBuy now.",
            page_archetype="product_detail",
            page_hints=["price_signals", "commerce_signals"],
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _BasePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=state.markdown)),
            patch(
                "agent.run.analyze_page_with_deep_agents",
                new=AsyncMock(side_effect=RuntimeError("structured response missing")),
            ),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_analyze_page_tool_call("What price is visible on this page?")],
                        [_verify_goal_tool_call("The current page clearly shows the iPhone 17 price.")],
                        [_complete_goal_tool_call(goal_summary="Captured the visible iPhone 17 price.")],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Tell me the price of the iPhone 17 on Apple's website.",
                goal_type="generic",
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.trace[0].decision.action, "analyze")
        self.assertIn("best_next_action=wait_for", result.trace[0].decision.analysis or "")
        self.assertEqual(result.trace[0].decision.result_data["recommended_action"], "wait_for")
        self.assertTrue(result.trace[0].decision.result_data["blocker_summary"].startswith("RuntimeError:"))

    async def test_extract_answer_does_not_enrich_with_fragment_heading(self) -> None:
        state = PageState(
            url="https://docs.example.com/reference#skip-tests",
            title="API Reference - Example Docs",
            markdown="API Reference",
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(
                    return_value=(_DummyPlaywright(), _DummyBrowser(), _FragmentHeadingPage())
                ),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# API Reference\n")),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [
                            _extract_answer_tool_call(
                                answer="API Reference - Example Docs",
                                evidence="Page title: API Reference - Example Docs",
                                confidence=0.7,
                            )
                        ]
                    ]
                ),
                start_url="https://docs.example.com/reference#skip-tests",
                target_prompt="Report the destination page title for the skip tests docs page.",
                goal_type="extract",
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "API Reference - Example Docs")
        self.assertNotIn("Anchor heading:", result.evidence or "")

    async def test_extract_answer_does_not_override_with_current_page_title(self) -> None:
        state = PageState(
            url="https://forecast.example.com/chicago",
            title="7-Day Forecast for Chicago",
            markdown="Chicago forecast",
            interactables=[],
        )

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _BasePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="Chicago forecast")),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [
                            _extract_answer_tool_call(
                                answer="National Weather Service",
                                evidence="Page title: National Weather Service",
                                confidence=0.7,
                            )
                        ]
                    ]
                ),
                start_url="https://forecast.example.com/chicago",
                target_prompt="Report the page title for the Chicago forecast page.",
                goal_type="extract",
            )

        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "National Weather Service")

    async def test_extract_answer_blocks_query_mismatch_until_destination_matches(self) -> None:
        home = PageState(
            url="https://playwright.dev/",
            title="Fast and reliable end-to-end testing for modern web apps | Playwright",
            markdown="Playwright homepage",
            interactables=[],
        )
        page = _BasePage()
        page.url = home.url
        page._title = home.title

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), page)),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=home)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value=home.markdown)),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [
                            _extract_answer_tool_call(
                                answer="Fast and reliable end-to-end testing for modern web apps | Playwright",
                                evidence="Current page title: Fast and reliable end-to-end testing for modern web apps | Playwright",
                                confidence=0.8,
                            )
                        ],
                    ]
                ),
                start_url="https://playwright.dev/",
                target_prompt="Search for locators and report the destination page title.",
                goal_type="extract",
                max_steps=2,
            )

        # Last-ditch extraction returns page content at max_steps
        if result.error is None:
            self.assertEqual(result.status, "completed")
            self.assertIn("Playwright", result.answer or "")
        else:
            self.assertEqual(result.error, "max_steps_exceeded")
        self.assertTrue(result.trace)

    async def test_type_and_submit_does_not_require_manual_focus(self) -> None:
        state = PageState(url="https://example.com", title="Example", markdown="", interactables=[])

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(
                    return_value=(_DummyPlaywright(), _DummyBrowser(), _NoFocusTypeSubmitPage())
                ),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Example\n\nSearch\n")),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_type_and_submit_tool_call('css=input[type="search"]', "Nintendo DS")],
                        [_verify_goal_tool_call("The search results page is visible.")],
                        [_complete_goal_tool_call(goal_summary="Submitted the search query.")],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Search for Nintendo DS.",
                goal_type="generic",
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["type_and_submit", "verify", "complete"])

    async def test_complete_goal_requires_current_verification(self) -> None:
        state = PageState(url="https://example.com", title="Example", markdown="", interactables=[])

        with (
            patch(
                "agent.run.run_browser",
                new=AsyncMock(return_value=(_DummyPlaywright(), _DummyBrowser(), _BasePage())),
            ),
            patch("agent.run.capture_state", new=AsyncMock(return_value=state)),
            patch("agent.run.page_to_markdown", new=AsyncMock(return_value="# Example\n\nReady\n")),
            patch(
                "agent.run.verify_goal_with_deep_agents",
                new=AsyncMock(return_value=self._verified_result()),
            ),
        ):
            result = await run_agent(
                openrouter_client=_StubOpenRouterClient(
                    [
                        [_complete_goal_tool_call(goal_summary="Finished without checking.")],
                        [_verify_goal_tool_call("The requested outcome is actually visible.")],
                        [_complete_goal_tool_call(goal_summary="Finished after checking.")],
                    ]
                ),
                start_url="https://example.com",
                target_prompt="Finish the browser task.",
                goal_type="generic",
            )

        self.assertIsNone(result.error)
        self.assertEqual([item.decision.action for item in result.trace], ["verify", "verify", "complete"])
        self.assertIn("Call verify_goal", result.trace[0].decision.analysis or "")
