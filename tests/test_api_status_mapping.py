from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.models import AgentResult
from auto_browse.api import create_app
from auto_browse.security import DEFAULT_API_TOKEN_HEADER, SecuritySettings

TEST_SHARED_API_TOKEN = "test-shared-token"


def _client() -> TestClient:
    return TestClient(
        create_app(
            security=SecuritySettings(
                api_token=TEST_SHARED_API_TOKEN,
                api_token_header=DEFAULT_API_TOKEN_HEADER,
                rate_limit_max_requests=1000,
                rate_limit_window_seconds=60,
                max_concurrent_requests_per_ip=100,
                max_request_body_bytes=64 * 1024,
            )
        )
    )


def _auth_headers() -> dict[str, str]:
    return {DEFAULT_API_TOKEN_HEADER: TEST_SHARED_API_TOKEN}


class ApiStatusMappingTest(unittest.TestCase):
    def test_run_returns_422_when_agent_returns_error(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch(
                "auto_browse.api.run_agent",
                new=AsyncMock(return_value=AgentResult(error="max_steps_exceeded", trace=[])),
            ),
        ):
            client = _client()
            response = client.post(
                "/run",
                headers=_auth_headers(),
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
            client = _client()
            response = client.post(
                "/run",
                headers=_auth_headers(),
                json={
                    "start_url": "https://example.com",
                    "target_prompt": "test",
                    "max_steps": 3,
                    "headed": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "ok")
