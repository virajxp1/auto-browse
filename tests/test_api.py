from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from playwright.async_api import Error as PlaywrightError

from agent.models import AgentResult
from auto_browse.api import app


class ApiTest(unittest.TestCase):
    def test_run_uses_env_client_when_inline_credentials_are_missing(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()) as mock_from_env,
            patch(
                "auto_browse.api.run_agent",
                new=AsyncMock(
                    return_value=AgentResult(
                        answer="May 25, 1977",
                        source_url="https://example.com",
                        evidence="Release date May 25, 1977",
                        confidence=0.86,
                        trace=[],
                    )
                ),
            ) as mock_run_agent,
        ):
            response = TestClient(app).post(
                "/run",
                json={
                    "start_url": "https://example.com",
                    "target_prompt": "release date",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "May 25, 1977")
        mock_from_env.assert_called_once()
        mock_run_agent.assert_awaited_once()
        self.assertTrue(callable(mock_run_agent.await_args.kwargs["on_step"]))

    def test_run_rejects_partial_inline_credentials(self) -> None:
        response = TestClient(app).post(
            "/run",
            json={
                "start_url": "https://example.com",
                "target_prompt": "release date",
                "api_key": "abc",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("api_key and model_name", response.json()["detail"])

    def test_run_normalizes_start_url_without_scheme(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch(
                "auto_browse.api.run_agent",
                new=AsyncMock(
                    return_value=AgentResult(
                        answer="ok",
                        source_url="https://www.google.com",
                        evidence="ok",
                        confidence=0.8,
                        trace=[],
                    )
                ),
            ) as mock_run_agent,
        ):
            response = TestClient(app).post(
                "/run",
                json={
                    "start_url": "www.google.com",
                    "target_prompt": "release date of Star Wars",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_run_agent.await_args.kwargs["start_url"], "https://www.google.com")

    def test_run_maps_playwright_navigation_error_to_400(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch(
                "auto_browse.api.run_agent",
                new=AsyncMock(side_effect=PlaywrightError("Cannot navigate to invalid URL")),
            ),
        ):
            response = TestClient(app).post(
                "/run",
                json={
                    "start_url": "https://example.com",
                    "target_prompt": "release date",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Browser navigation failed", response.json()["detail"])

    def test_run_can_disable_step_logging_callback(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch(
                "auto_browse.api.run_agent",
                new=AsyncMock(
                    return_value=AgentResult(
                        answer="ok",
                        source_url="https://example.com",
                        evidence="ok",
                        confidence=0.8,
                        trace=[],
                    )
                ),
            ) as mock_run_agent,
        ):
            response = TestClient(app).post(
                "/run",
                json={
                    "start_url": "https://example.com",
                    "target_prompt": "release date",
                    "log_steps": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(mock_run_agent.await_args.kwargs["on_step"])
