from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException
from playwright.async_api import Error as PlaywrightError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.models import AgentResult, AgentStepTrace
from agent.openrouter_client import OpenRouterClient
from agent.run import run_agent

logger = logging.getLogger("auto_browse.api")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start_url: str
    target_prompt: str
    max_steps: int = Field(default=10, ge=1, le=50)
    headed: bool = False
    log_steps: bool = True

    api_key: str | None = None
    model_name: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"

    @field_validator("start_url")
    @classmethod
    def normalize_start_url(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("start_url cannot be empty")
        if "://" not in trimmed:
            return f"https://{trimmed}"
        return trimmed


def _client_from_request(request: RunRequest) -> OpenRouterClient:
    has_inline_key = bool(request.api_key)
    has_inline_model = bool(request.model_name)
    if has_inline_key != has_inline_model:
        raise HTTPException(
            status_code=400,
            detail="Provide both api_key and model_name together, or omit both.",
        )
    if has_inline_key and has_inline_model:
        return OpenRouterClient(
            api_key=request.api_key or "",
            model_name=request.model_name or "",
            base_url=request.base_url,
        )
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
        client = _client_from_request(payload)

        on_step = None
        if payload.log_steps:
            def _log_step(trace_item: AgentStepTrace) -> None:
                logger.info(
                    "[run:%s step:%s] summary=%s",
                    request_id,
                    trace_item.step,
                    trace_item.decision.step_summary,
                )
                logger.info(
                    "[run:%s step:%s] next=%s",
                    request_id,
                    trace_item.step,
                    trace_item.decision.next_step,
                )

            on_step = _log_step

        logger.info(
            "[run:%s] start url=%s max_steps=%s headed=%s",
            request_id,
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
            )
            logger.info(
                "[run:%s] finished error=%s answer_present=%s trace_steps=%s",
                request_id,
                result.error,
                bool(result.answer),
                len(result.trace),
            )
            return result
        except PlaywrightError as exc:
            raise HTTPException(status_code=400, detail=f"Browser navigation failed: {exc}") from exc

    return app


app = create_app()
