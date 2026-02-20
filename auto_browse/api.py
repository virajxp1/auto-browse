from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException
from playwright.async_api import Error as PlaywrightError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.models import AgentResult, AgentStepTrace
from agent.openrouter_client import OpenRouterClient
from agent.run import run_agent

# Use uvicorn's error logger so step logs show up in normal server output.
logger = logging.getLogger("uvicorn.error")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start_url: str
    target_prompt: str
    max_steps: int = Field(default=10, ge=1, le=50)
    headed: bool = False
    log_steps: bool = True

    trace_id: str | None = None
    session_id: str | None = None

    @field_validator("start_url")
    @classmethod
    def normalize_start_url(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("start_url cannot be empty")
        if "://" not in trimmed:
            return f"https://{trimmed}"
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


def create_app() -> FastAPI:
    app = FastAPI(title="Auto Browse API", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run", response_model=AgentResult)
    async def run(payload: RunRequest) -> AgentResult:
        request_id = uuid.uuid4().hex[:8]
        trace_id = payload.trace_id or _new_trace_id()
        session_id = payload.session_id or trace_id
        client = _client_from_env()

        on_step = None
        if payload.log_steps:
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

            on_step = _log_step

        logger.info(
            "[run:%s trace:%s session:%s] start url=%s max_steps=%s headed=%s",
            request_id,
            trace_id,
            session_id,
            payload.start_url,
            payload.max_steps,
            payload.headed,
        )
        try:
            result = await run_agent(
                client,
                start_url=payload.start_url,
                target_prompt=payload.target_prompt,
                max_steps=payload.max_steps,
                headless=not payload.headed,
                on_step=on_step,
                trace_id=trace_id,
                session_id=session_id,
            )
            logger.info(
                "[run:%s trace:%s session:%s] finished error=%s answer_present=%s trace_steps=%s",
                request_id,
                trace_id,
                session_id,
                result.error,
                bool(result.answer),
                len(result.trace),
            )
            return result
        except PlaywrightError as exc:
            raise HTTPException(status_code=400, detail=f"Browser navigation failed: {exc}") from exc

    return app


app = create_app()
