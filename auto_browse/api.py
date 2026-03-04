from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from playwright.async_api import Error as PlaywrightError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.models import AgentResult, AgentStepTrace
from agent.openrouter_client import OpenRouterClient
from agent.run import run_agent
from auto_browse.security import ApiSecurityMiddleware, SecuritySettings

# Use uvicorn's error logger so step logs show up in normal server output.
logger = logging.getLogger("uvicorn.error")


class _RunCooldownLimiter:
    def __init__(self, *, min_interval_seconds: float = 20.0) -> None:
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
    max_steps: int = Field(default=10, ge=1, le=50)
    max_actions_per_step: int = Field(default=1, ge=1, le=4)
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


def create_app(security: SecuritySettings | None = None) -> FastAPI:
    app = FastAPI(title="auto-browse API", version="0.1.0")
    security_settings = security or SecuritySettings.from_env()
    app.add_middleware(ApiSecurityMiddleware, settings=security_settings)
    run_rate_limiter = _RunCooldownLimiter(min_interval_seconds=20.0)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run", response_model=AgentResult)
    async def run(payload: RunRequest) -> AgentResult:
        request_id = uuid.uuid4().hex[:8]
        trace_id = _new_trace_id()

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
            "[run:%s trace:%s] start url=%s max_steps=%s headed=%s",
            request_id,
            trace_id,
            payload.start_url,
            payload.max_steps,
            payload.headed,
        )
        logger.info(
            "[run:%s trace:%s] input_payload=%s",
            request_id,
            trace_id,
            payload.model_dump(mode="json"),
        )
        is_allowed, retry_after_seconds = run_rate_limiter.try_acquire(now=time.monotonic())
        if not is_allowed:
            response_payload = {
                "detail": "Run requests are limited to 1 request every 20 seconds",
            }
            logger.warning(
                "[run:%s trace:%s] blocked_by_rate_limit retry_after=%s cooldown_seconds=%s",
                request_id,
                trace_id,
                retry_after_seconds,
                20,
            )
            logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
            raise HTTPException(
                status_code=429,
                detail=response_payload["detail"],
                headers={"Retry-After": str(retry_after_seconds)},
            )

        try:
            client = _client_from_env()
        except HTTPException as exc:
            response_payload = {"detail": exc.detail}
            logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
            raise
        except Exception as exc:
            response_payload = {"detail": "Unhandled internal error occurred"}
            logger.exception(
                "[run:%s trace:%s] unhandled_exception error=%s",
                request_id,
                trace_id,
                str(exc),
            )
            logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
            raise HTTPException(status_code=500, detail=response_payload["detail"]) from exc

        try:
            result = await run_agent(
                client,
                start_url=payload.start_url,
                target_prompt=payload.target_prompt,
                max_steps=payload.max_steps,
                max_actions_per_step=payload.max_actions_per_step,
                extraction_schema=payload.extraction_schema,
                extraction_selector=payload.extraction_selector,
                headless=not payload.headed,
                on_step=_log_step,
                trace_id=trace_id,
            )
        except PlaywrightError as exc:
            response_payload = {"detail": f"Browser navigation failed: {exc}"}
            logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
            raise HTTPException(status_code=400, detail=response_payload["detail"]) from exc
        except asyncio.TimeoutError as exc:
            response_payload = {"detail": "Agent run timed out"}
            logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
            raise HTTPException(status_code=504, detail=response_payload["detail"]) from exc
        except ValueError as exc:
            response_payload = {"detail": str(exc)}
            logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
            raise HTTPException(status_code=400, detail=response_payload["detail"]) from exc
        except Exception as exc:
            response_payload = {"detail": "Unhandled internal error occurred"}
            logger.exception(
                "[run:%s trace:%s] unhandled_exception error=%s",
                request_id,
                trace_id,
                str(exc),
            )
            logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
            raise HTTPException(status_code=500, detail=response_payload["detail"]) from exc

        if result.error:
            logger.info(
                "[run:%s trace:%s] finished error=%s answer_present=%s trace_steps=%s",
                request_id,
                trace_id,
                result.error,
                bool(result.answer),
                len(result.trace),
            )
            detail = result.model_dump()
            response_payload = {"detail": detail}
            logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
            raise HTTPException(status_code=422, detail=detail)

        logger.info(
            "[run:%s trace:%s] finished error=%s answer_present=%s trace_steps=%s",
            request_id,
            trace_id,
            result.error,
            bool(result.answer),
            len(result.trace),
        )
        logger.info(
            "[run:%s trace:%s] output_payload=%s",
            request_id,
            trace_id,
            result.model_dump(mode="json"),
        )
        return result

    return app
