from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from auto_browse_v2.graph import run_query
from auto_browse_v2.models import SubTaskResult

logger = logging.getLogger("uvicorn.error")

_REQUEST_ID_LEN = 8


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(description="Complex multi-part research query", min_length=1, max_length=2000)


class RunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    query: str
    final_answer: str
    subtask_results: list[SubTaskResult]


def create_app() -> FastAPI:
    app = FastAPI(
        title="auto-browse-v2",
        version="0.2.0",
        description="Parallel multi-site research orchestrator.",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run", response_model=RunResponse)
    async def run(payload: RunRequest) -> RunResponse:
        request_id = uuid.uuid4().hex[:_REQUEST_ID_LEN]
        logger.info("[%s] query=%r", request_id, payload.query)

        try:
            state = await run_query(payload.query)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("[%s] Unhandled error: %s", request_id, exc)
            raise HTTPException(status_code=500, detail="Internal orchestration error") from exc

        final_answer: str = state.get("final_answer", "")
        results: list[SubTaskResult] = state.get("results", [])

        logger.info("[%s] done subtasks=%d answer_len=%d", request_id, len(results), len(final_answer))
        return RunResponse(
            request_id=request_id,
            query=payload.query,
            final_answer=final_answer,
            subtask_results=results,
        )

    return app


app = create_app()
