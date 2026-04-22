from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from playwright.async_api import Error as PlaywrightError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.models import AgentResult, AgentStepTrace
from agent.openrouter_client import OpenRouterClient
from agent.run import run_agent
from auto_browse.security import ApiSecurityMiddleware, SecuritySettings

# Use uvicorn's error logger so step logs show up in normal server output.
logger = logging.getLogger("uvicorn.error")
_REQUEST_ID_LENGTH = 8
_RUN_COOLDOWN_SECONDS = 20
_RUN_COOLDOWN_MESSAGE = "Run requests are limited to 1 request every 20 seconds"
_UNHANDLED_ERROR_MESSAGE = "Unhandled internal error occurred"
_RUN_TIMEOUT_MESSAGE = "Agent run timed out"


class _RunCooldownLimiter:
    def __init__(self, *, min_interval_seconds: float = float(_RUN_COOLDOWN_SECONDS)) -> None:
        self._min_interval_seconds = min_interval_seconds
        self._last_allowed_request_time: float | None = None
        self._lock = threading.Lock()

    def try_acquire(self, *, now: float) -> tuple[bool, int]:
        with self._lock:
            if self._last_allowed_request_time is None:
                self._last_allowed_request_time = now
                return True, 0

            elapsed = now - self._last_allowed_request_time
            remaining = self._min_interval_seconds - elapsed
            if remaining > 0:
                return False, max(1, math.ceil(remaining))

            self._last_allowed_request_time = now
            return True, 0


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start_url: str
    target_prompt: str
    goal_type: str | None = None
    task_data: dict[str, str] | None = None
    sensitive_data: dict[str, str] | None = None
    max_steps: int = Field(default=10, ge=1, le=50)
    max_actions_per_step: int = Field(default=1, ge=1, le=3)
    max_runtime_seconds: int = Field(default=90, ge=1, le=600)
    extraction_schema: dict[str, str] | None = None
    extraction_selector: str | None = None
    headed: bool = False

    @field_validator("start_url")
    @classmethod
    def normalize_start_url(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("start_url cannot be empty")
        if "://" not in trimmed:
            return f"https://{trimmed}"
        return trimmed

    @field_validator("goal_type")
    @classmethod
    def normalize_goal_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("goal_type cannot be empty")
        return trimmed

    @field_validator("task_data", "sensitive_data")
    @classmethod
    def validate_task_maps(
        cls,
        value: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("data maps cannot be empty")
        normalized: dict[str, str] = {}
        for key, raw in value.items():
            normalized_key = key.strip()
            normalized_value = raw.strip()
            if not normalized_key:
                raise ValueError("data map keys must be non-empty")
            if not normalized_value:
                raise ValueError("data map values must be non-empty")
            normalized[normalized_key] = normalized_value
        return normalized

    @field_validator("extraction_schema")
    @classmethod
    def validate_extraction_schema(
        cls,
        value: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("extraction_schema cannot be empty")
        normalized: dict[str, str] = {}
        for key, description in value.items():
            normalized_key = key.strip()
            normalized_description = description.strip()
            if not normalized_key:
                raise ValueError("extraction_schema keys must be non-empty")
            if not normalized_description:
                raise ValueError("extraction_schema values must be non-empty")
            normalized[normalized_key] = normalized_description
        return normalized

    @field_validator("extraction_selector")
    @classmethod
    def normalize_extraction_selector(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("extraction_selector cannot be empty")
        return trimmed


def _new_trace_id() -> str:
    uuid7_factory = getattr(uuid, "uuid7", None)
    if callable(uuid7_factory):
        return str(uuid7_factory())
    return str(uuid.uuid4())


def _client_from_env() -> OpenRouterClient:
    try:
        return OpenRouterClient.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _redacted_payload(payload: RunRequest) -> dict[str, object]:
    redacted = payload.model_dump(mode="json")
    sensitive_data = redacted.get("sensitive_data")
    if isinstance(sensitive_data, dict):
        redacted["sensitive_data"] = {key: "[REDACTED]" for key in sensitive_data}
    return redacted


def _log_output_payload(request_id: str, trace_id: str, payload: dict[str, object]) -> None:
    logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, payload)


def _result_has_user_payload(result: AgentResult) -> bool:
    return bool(result.answer or result.goal_summary or result.result_data)


def _log_run_finished(request_id: str, trace_id: str, result: AgentResult) -> None:
    logger.info(
        "[run:%s trace:%s] finished error=%s result_present=%s trace_steps=%s",
        request_id,
        trace_id,
        result.error,
        _result_has_user_payload(result),
        len(result.trace),
    )


def _logged_http_exception(
    request_id: str,
    trace_id: str,
    *,
    status_code: int,
    detail: str | dict[str, object],
    headers: dict[str, str] | None = None,
) -> HTTPException:
    payload = {"detail": detail}
    _log_output_payload(request_id, trace_id, payload)
    return HTTPException(status_code=status_code, detail=detail, headers=headers)


def _logged_unhandled_http_exception(
    request_id: str,
    trace_id: str,
    exc: Exception,
) -> HTTPException:
    logger.exception(
        "[run:%s trace:%s] unhandled_exception error=%s",
        request_id,
        trace_id,
        str(exc),
    )
    return _logged_http_exception(
        request_id,
        trace_id,
        status_code=500,
        detail=_UNHANDLED_ERROR_MESSAGE,
    )


def create_app(security: SecuritySettings | None = None) -> FastAPI:
    app = FastAPI(title="auto-browse API", version="0.1.0")
    security_settings = security or SecuritySettings.from_env()
    app.add_middleware(ApiSecurityMiddleware, settings=security_settings)
    run_rate_limiter = _RunCooldownLimiter(min_interval_seconds=float(_RUN_COOLDOWN_SECONDS))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run", response_model=AgentResult)
    async def run(payload: RunRequest, request: Request) -> AgentResult:
        request_id = uuid.uuid4().hex[:_REQUEST_ID_LENGTH]
        trace_id = _new_trace_id()
        trace_parent = request.headers.get("x-bt-parent", "").strip() or None

        def _log_step(trace_item: AgentStepTrace) -> None:
            logger.info(
                "[run:%s trace:%s step:%s] summary=%s",
                request_id,
                trace_id,
                trace_item.step,
                trace_item.decision.step_summary,
            )
            logger.info(
                "[run:%s trace:%s step:%s] next=%s",
                request_id,
                trace_id,
                trace_item.step,
                trace_item.decision.next_step,
            )

        logger.info(
            "[run:%s trace:%s] start url=%s max_steps=%s max_runtime_seconds=%s headed=%s",
            request_id,
            trace_id,
            payload.start_url,
            payload.max_steps,
            payload.max_runtime_seconds,
            payload.headed,
        )
        logger.info(
            "[run:%s trace:%s] input_payload=%s",
            request_id,
            trace_id,
            _redacted_payload(payload),
        )

        is_allowed, retry_after_seconds = run_rate_limiter.try_acquire(now=time.monotonic())
        if not is_allowed:
            logger.warning(
                "[run:%s trace:%s] blocked_by_rate_limit retry_after=%s cooldown_seconds=%s",
                request_id,
                trace_id,
                retry_after_seconds,
                _RUN_COOLDOWN_SECONDS,
            )
            raise _logged_http_exception(
                request_id,
                trace_id,
                status_code=429,
                detail=_RUN_COOLDOWN_MESSAGE,
                headers={"Retry-After": str(retry_after_seconds)},
            )

        try:
            client = _client_from_env()
        except HTTPException as exc:
            raise _logged_http_exception(
                request_id,
                trace_id,
                status_code=exc.status_code,
                detail=exc.detail,
                headers=exc.headers,
            ) from exc
        except Exception as exc:
            raise _logged_unhandled_http_exception(request_id, trace_id, exc) from exc

        try:
            result = await run_agent(
                client,
                start_url=payload.start_url,
                target_prompt=payload.target_prompt,
                goal_type=payload.goal_type,
                task_data=payload.task_data,
                sensitive_data=payload.sensitive_data,
                max_steps=payload.max_steps,
                max_actions_per_step=payload.max_actions_per_step,
                max_runtime_seconds=payload.max_runtime_seconds,
                extraction_schema=payload.extraction_schema,
                extraction_selector=payload.extraction_selector,
                headless=not payload.headed,
                on_step=_log_step,
                trace_id=trace_id,
                trace_parent=trace_parent,
            )
        except PlaywrightError as exc:
            raise _logged_http_exception(
                request_id,
                trace_id,
                status_code=400,
                detail=f"Browser navigation failed: {exc}",
            ) from exc
        except asyncio.TimeoutError as exc:
            raise _logged_http_exception(
                request_id,
                trace_id,
                status_code=504,
                detail=_RUN_TIMEOUT_MESSAGE,
            ) from exc
        except ValueError as exc:
            raise _logged_http_exception(
                request_id,
                trace_id,
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise _logged_unhandled_http_exception(request_id, trace_id, exc) from exc

        _log_run_finished(request_id, trace_id, result)
        if result.error:
            detail = result.model_dump()
            raise _logged_http_exception(
                request_id,
                trace_id,
                status_code=422,
                detail=detail,
            )

        _log_output_payload(request_id, trace_id, result.model_dump(mode="json"))
        return result

    return app
