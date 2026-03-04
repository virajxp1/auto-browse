from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from playwright.async_api import Error as PlaywrightError

from agent.models import AgentResult
from auto_browse.api import create_app
from auto_browse.security import DEFAULT_API_TOKEN_HEADER, SecuritySettings

TEST_SHARED_API_TOKEN = "test-shared-token"


def _build_client(
    *,
    rate_limit_max_requests: int = 1000,
    rate_limit_window_seconds: int = 60,
    max_concurrent_requests_per_ip: int = 100,
    max_request_body_bytes: int = 64 * 1024,
) -> TestClient:
    return TestClient(
        create_app(
            security=SecuritySettings(
                api_token=TEST_SHARED_API_TOKEN,
                api_token_header=DEFAULT_API_TOKEN_HEADER,
                rate_limit_max_requests=rate_limit_max_requests,
                rate_limit_window_seconds=rate_limit_window_seconds,
                max_concurrent_requests_per_ip=max_concurrent_requests_per_ip,
                max_request_body_bytes=max_request_body_bytes,
            )
        )
    )


def _auth_headers(token: str = TEST_SHARED_API_TOKEN) -> dict[str, str]:
    return {DEFAULT_API_TOKEN_HEADER: token}


def _run_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "start_url": "https://example.com",
        "target_prompt": "release date",
    }
    payload.update(overrides)
    return payload


class ApiTest(unittest.TestCase):
    def test_run_rejects_requests_without_api_token_header(self) -> None:
        with patch("auto_browse.api.run_agent", new=AsyncMock()) as mock_run_agent:
            response = _build_client().post("/run", json=_run_payload())

        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing or invalid API token", response.json()["detail"])
        mock_run_agent.assert_not_awaited()

    def test_run_rejects_requests_with_wrong_api_token(self) -> None:
        with patch("auto_browse.api.run_agent", new=AsyncMock()) as mock_run_agent:
            response = _build_client().post(
                "/run",
                headers=_auth_headers(token="wrong-token"),
                json=_run_payload(),
            )

        self.assertEqual(response.status_code, 401)
        mock_run_agent.assert_not_awaited()

    def test_run_rejects_when_another_run_is_in_progress(self) -> None:
        with (
            patch("auto_browse.api._RunConcurrencyGate.try_acquire", return_value=False),
            patch("auto_browse.api.logger.warning") as mock_logger_warning,
            patch("auto_browse.api.run_agent", new=AsyncMock()) as mock_run_agent,
        ):
            response = _build_client().post("/run", headers=_auth_headers(), json=_run_payload())

        self.assertEqual(response.status_code, 429)
        self.assertIn("already in progress", response.json()["detail"])
        blocked_logs = [
            call
            for call in mock_logger_warning.call_args_list
            if call.args and call.args[0] == "[run:%s trace:%s] blocked_by_gate active_runs=%s max_active_runs=%s"
        ]
        self.assertEqual(len(blocked_logs), 1)
        mock_run_agent.assert_not_awaited()

    def test_run_logs_input_and_output_payloads(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch("auto_browse.api.logger.info") as mock_logger_info,
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
            ),
        ):
            response = _build_client().post("/run", headers=_auth_headers(), json=_run_payload())

        self.assertEqual(response.status_code, 200)

        input_logs = [
            call
            for call in mock_logger_info.call_args_list
            if call.args and call.args[0] == "[run:%s trace:%s] input_payload=%s"
        ]
        self.assertEqual(len(input_logs), 1)
        self.assertEqual(input_logs[0].args[3]["start_url"], "https://example.com")
        self.assertEqual(input_logs[0].args[3]["target_prompt"], "release date")

        output_logs = [
            call
            for call in mock_logger_info.call_args_list
            if call.args and call.args[0] == "[run:%s trace:%s] output_payload=%s"
        ]
        self.assertEqual(len(output_logs), 1)
        self.assertEqual(output_logs[0].args[3]["answer"], "May 25, 1977")

    def test_middleware_logs_rejected_request(self) -> None:
        with (
            patch("auto_browse.security.logger.info") as mock_logger_info,
            patch("auto_browse.api.run_agent", new=AsyncMock()) as mock_run_agent,
        ):
            response = _build_client().post("/run", json=_run_payload())

        self.assertEqual(response.status_code, 401)
        mock_run_agent.assert_not_awaited()
        rejection_logs = [
            call
            for call in mock_logger_info.call_args_list
            if call.args and call.args[0] == "[security] rejected method=%s path=%s client_ip=%s status=%s reason=%s detail=%s"
        ]
        self.assertEqual(len(rejection_logs), 1)
        self.assertEqual(rejection_logs[0].args[1], "POST")
        self.assertEqual(rejection_logs[0].args[2], "/run")
        self.assertEqual(rejection_logs[0].args[4], 401)
        self.assertEqual(rejection_logs[0].args[5], "invalid_api_token")

    def test_run_applies_rate_limit_before_route_logic(self) -> None:
        client = _build_client(rate_limit_max_requests=1, rate_limit_window_seconds=60)
        first = client.get("/health", headers=_auth_headers())
        second = client.get("/health", headers=_auth_headers())

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.headers.get("Retry-After"), "60")

    def test_run_rejects_request_body_that_is_too_large(self) -> None:
        with patch("auto_browse.api.run_agent", new=AsyncMock()) as mock_run_agent:
            response = _build_client(max_request_body_bytes=10).post(
                "/run",
                headers=_auth_headers(),
                json=_run_payload(),
            )

        self.assertEqual(response.status_code, 413)
        self.assertIn("Request body too large", response.json()["detail"])
        mock_run_agent.assert_not_awaited()

    def test_run_rejects_oversized_body_even_with_small_content_length_header(self) -> None:
        with patch("auto_browse.api.run_agent", new=AsyncMock()) as mock_run_agent:
            response = _build_client(max_request_body_bytes=10).post(
                "/run",
                headers={**_auth_headers(), "content-length": "1"},
                content=b"{" + b"a" * 200,
            )

        self.assertEqual(response.status_code, 413)
        self.assertIn("Request body too large", response.json()["detail"])
        mock_run_agent.assert_not_awaited()

    def test_run_uses_env_client(self) -> None:
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
            response = _build_client().post("/run", headers=_auth_headers(), json=_run_payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "May 25, 1977")
        mock_from_env.assert_called_once()
        mock_run_agent.assert_awaited_once()
        self.assertTrue(callable(mock_run_agent.await_args.kwargs["on_step"]))
        self.assertIsInstance(mock_run_agent.await_args.kwargs["trace_id"], str)

    def test_run_rejects_request_level_api_key(self) -> None:
        response = _build_client().post(
            "/run",
            headers=_auth_headers(),
            json=_run_payload(api_key="abc"),
        )

        self.assertEqual(response.status_code, 422)

    def test_run_rejects_request_level_model_name(self) -> None:
        response = _build_client().post(
            "/run",
            headers=_auth_headers(),
            json=_run_payload(model_name="openai/gpt-4.1-mini"),
        )

        self.assertEqual(response.status_code, 422)

    def test_run_rejects_request_level_log_steps(self) -> None:
        response = _build_client().post(
            "/run",
            headers=_auth_headers(),
            json=_run_payload(log_steps=False),
        )

        self.assertEqual(response.status_code, 422)

    def test_run_rejects_request_level_trace_id(self) -> None:
        response = _build_client().post(
            "/run",
            headers=_auth_headers(),
            json=_run_payload(trace_id="trace-123"),
        )

        self.assertEqual(response.status_code, 422)

    def test_run_rejects_request_level_session_id(self) -> None:
        response = _build_client().post(
            "/run",
            headers=_auth_headers(),
            json=_run_payload(session_id="session-456"),
        )

        self.assertEqual(response.status_code, 422)

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
            response = _build_client().post(
                "/run",
                headers=_auth_headers(),
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
            response = _build_client().post("/run", headers=_auth_headers(), json=_run_payload())

        self.assertEqual(response.status_code, 400)
        self.assertIn("Browser navigation failed", response.json()["detail"])

    def test_run_releases_gate_after_value_error(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch(
                "auto_browse.api.run_agent",
                new=AsyncMock(
                    side_effect=[
                        ValueError("bad input"),
                        AgentResult(
                            answer="ok",
                            source_url="https://example.com",
                            evidence="ok",
                            confidence=0.8,
                            trace=[],
                        ),
                    ]
                ),
            ),
        ):
            client = _build_client()
            first_response = client.post("/run", headers=_auth_headers(), json=_run_payload())
            second_response = client.post("/run", headers=_auth_headers(), json=_run_payload())

        self.assertEqual(first_response.status_code, 400)
        self.assertEqual(first_response.json()["detail"], "bad input")
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(second_response.json()["answer"], "ok")

    def test_run_releases_gate_after_timeout_error(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch(
                "auto_browse.api.run_agent",
                new=AsyncMock(
                    side_effect=[
                        asyncio.TimeoutError(),
                        AgentResult(
                            answer="ok",
                            source_url="https://example.com",
                            evidence="ok",
                            confidence=0.8,
                            trace=[],
                        ),
                    ]
                ),
            ),
        ):
            client = _build_client()
            first_response = client.post("/run", headers=_auth_headers(), json=_run_payload())
            second_response = client.post("/run", headers=_auth_headers(), json=_run_payload())

        self.assertEqual(first_response.status_code, 504)
        self.assertIn("timed out", first_response.json()["detail"])
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(second_response.json()["answer"], "ok")

    def test_run_always_sets_step_logging_callback(self) -> None:
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
            response = _build_client().post("/run", headers=_auth_headers(), json=_run_payload())

        self.assertEqual(response.status_code, 200)
        self.assertTrue(callable(mock_run_agent.await_args.kwargs["on_step"]))

    def test_run_uses_generated_trace_id_when_missing(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch("auto_browse.api._new_trace_id", return_value="trace-generated"),
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
            response = _build_client().post("/run", headers=_auth_headers(), json=_run_payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_run_agent.await_args.kwargs["trace_id"], "trace-generated")

    def test_run_passes_extraction_and_action_budget_options(self) -> None:
        with (
            patch("auto_browse.api.OpenRouterClient.from_env", return_value=object()),
            patch(
                "auto_browse.api.run_agent",
                new=AsyncMock(
                    return_value=AgentResult(
                        answer=None,
                        structured_data={
                            "release_date": "May 25, 1977",
                            "director": "George Lucas",
                        },
                        source_url="https://example.com",
                        evidence="Infobox data",
                        confidence=0.8,
                        trace=[],
                    )
                ),
            ) as mock_run_agent,
        ):
            response = _build_client().post(
                "/run",
                headers=_auth_headers(),
                json={
                    "start_url": "https://example.com",
                    "target_prompt": "Extract release date and director",
                    "max_actions_per_step": 3,
                    "extraction_selector": "css=table.infobox",
                    "extraction_schema": {
                        "release_date": "The theatrical release date",
                        "director": "The director name",
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_run_agent.await_args.kwargs["max_actions_per_step"], 3)
        self.assertEqual(
            mock_run_agent.await_args.kwargs["extraction_selector"],
            "css=table.infobox",
        )
        self.assertEqual(
            mock_run_agent.await_args.kwargs["extraction_schema"],
            {
                "release_date": "The theatrical release date",
                "director": "The director name",
            },
        )
