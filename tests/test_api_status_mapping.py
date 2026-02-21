from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.models import AgentResult
from auto_browse.api import app


class ApiStatusMappingTest(unittest.TestCase):
    def test_run_returns_422_when_agent_returns_error(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch(
                "auto_browse.api.run_agent",
                new=AsyncMock(return_value=AgentResult(error="max_steps_exceeded", trace=[])),
            ),
        ):
            client = TestClient(app)
            response = client.post(
                "/run",
                json={
                    "start_url": "https://example.com",
                    "target_prompt": "test",
                    "max_steps": 3,
                    "headed": False,
                },
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["error"], "max_steps_exceeded")

    def test_run_returns_200_when_agent_succeeds(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch(
                "auto_browse.api.run_agent",
                new=AsyncMock(
                    return_value=AgentResult(
                        answer="ok",
                        source_url="https://example.com",
                        evidence="example evidence",
                        confidence=0.9,
                        trace=[],
                    )
                ),
            ),
        ):
            client = TestClient(app)
            response = client.post(
                "/run",
                json={
                    "start_url": "https://example.com",
                    "target_prompt": "test",
                    "max_steps": 3,
                    "headed": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "ok")
