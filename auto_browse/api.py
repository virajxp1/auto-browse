from __future__ import annotations

import asyncio
import logging
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


class _RunConcurrencyGate:
    def __init__(self, *, max_active_runs: int = 1) -> None:
        self._max_active_runs = max_active_runs
        self._active_runs = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._active_runs >= self._max_active_runs:
                return False
            self._active_runs += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active_runs > 0:
                self._active_runs -= 1


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
    run_gate = _RunConcurrencyGate(max_active_runs=1)

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
        if not await run_gate.try_acquire():
            response_payload = {"detail": "Another agent run is already in progress"}
            logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
            raise HTTPException(status_code=429, detail=response_payload["detail"])

        try:
            try:
                client = _client_from_env()
            except HTTPException as exc:
                response_payload = {"detail": exc.detail}
                logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
                raise

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
            except ValueError as exc:
                response_payload = {"detail": str(exc)}
                logger.info("[run:%s trace:%s] output_payload=%s", request_id, trace_id, response_payload)
                raise HTTPException(status_code=400, detail=response_payload["detail"]) from exc

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
        finally:
            await run_gate.release()

    return app
