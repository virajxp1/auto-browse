from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from auto_browse import OpenRouterClient, run_agent

app = FastAPI(title="auto-browse API", version="0.1.0")


class RunRequest(BaseModel):
    start_url: str
    target_prompt: str
    max_steps: int = Field(default=10, ge=1, le=100)
    headed: bool = False


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
async def run(request: RunRequest) -> dict:
    try:
        client = OpenRouterClient.from_env()
        result = await run_agent(
            client,
            start_url=request.start_url,
            target_prompt=request.target_prompt,
            max_steps=request.max_steps,
            headless=not request.headed,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 500 if message.startswith("Missing required environment variable") else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"agent_failed: {exc}") from exc

    if result.error:
        raise HTTPException(status_code=422, detail=result.model_dump())

    return result.model_dump()
