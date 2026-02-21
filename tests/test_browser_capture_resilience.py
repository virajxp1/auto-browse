from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from agent.browser import capture_state
from agent.models import Interactable


class _FakePage:
    url = "https://example.com"

    async def title(self) -> str:
        return "Example"


class BrowserCaptureResilienceTest(unittest.IsolatedAsyncioTestCase):
    async def test_builder_failure_does_not_skip_other_interactables(self) -> None:
        button = Interactable(kind="button", label="Submit", selector="css=button >> nth=0")
        link = Interactable(kind="link", label="Home", selector="css=a[href] >> nth=0", href="/")

        with (
            patch("agent.browser._build_input_interactables", new=AsyncMock(side_effect=RuntimeError("boom"))),
            patch("agent.browser._build_button_interactables", new=AsyncMock(return_value=[button])),
            patch("agent.browser._build_link_interactables", new=AsyncMock(return_value=[link])),
        ):
            state = await capture_state(_FakePage())

        self.assertEqual(state.url, "https://example.com")
        self.assertEqual(state.title, "Example")
        self.assertEqual([item.kind for item in state.interactables], ["button", "link"])
