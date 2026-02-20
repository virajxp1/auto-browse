# Auto Browse MVP

Minimal constrained browser agent:

1. Load page with Playwright.
2. Convert HTML to readable markdown (readability + markdownify).
3. Orchestrate the loop as a LangGraph state machine.
4. Ask an LLM for exactly one tool call each turn.
5. Execute that tool with LangGraph `ToolNode` (`extract_answer`, `type_and_submit`, `click`, `navigate`, `fail`).
6. Repeat until answer or failure.

At every step, the API logs:

- what the LLM understood on the current page
- what it will do next

## Setup

1. Install dependencies:
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   playwright install chromium
   ```
2. Set environment variables:
   ```bash
   export OPENROUTER_API_KEY=...
   export OPENROUTER_MODEL=openai/gpt-4.1-mini
   ```
   The runtime also auto-loads these keys from a local `.env` file if present.

## Run As API

Start the server:

```bash
./scripts/run_api.sh
```

or run `uvicorn` directly:

```bash
python -m uvicorn auto_browse.api:app --host 127.0.0.1 --port 8000
```

Call it:

```bash
curl -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "start_url": "https://www.google.com",
    "target_prompt": "release date of Star Wars",
    "max_steps": 10
  }'
```

Notes:

- The API uses env credentials/model only (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`).
- `start_url` accepts either a full URL (`https://...`) or a hostname (`www.google.com`), which is auto-normalized to `https://...`.
- The API always logs intermediary step summaries and next actions.
- Each run sends OpenRouter tracing metadata on every LLM step:
  - `trace.trace_id` is generated automatically (UUIDv7 fallback to UUID4).
  - `trace.generation_name` is set per step as `planner.1`, `planner.2`, ...
  - `session_id` is generated internally to match `trace_id`.

## Deploy (Render Example)

Build command:

```bash
pip install -e . && playwright install chromium
```

Start command:

```bash
./scripts/run_api.sh
```

Required env vars:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`

Optional runtime env vars:

- `PORT` (provided by Render automatically)
- `AUTO_BROWSE_API_HOST` (default `0.0.0.0`)
- `AUTO_BROWSE_API_PORT` (default `8000` if `PORT` is not set)
- `AUTO_BROWSE_INSTALL_PLAYWRIGHT=1` (installs Chromium at startup)

## Use In Other Projects

Install as a dependency:

```bash
pip install -e /path/to/auto-browse
```

Import from the public package namespace:

```python
from auto_browse import OpenRouterClient, run_agent
```

Run programmatically:

```python
import asyncio

from auto_browse import OpenRouterClient, run_agent


async def main() -> None:
    client = OpenRouterClient.from_env()
    result = await run_agent(
        client,
        start_url="https://www.google.com",
        target_prompt="release date of Star Wars",
        max_steps=10,
        headless=True,
    )
    print(result.model_dump())


asyncio.run(main())
```

## Output

The `/run` response body contains:

- `answer`
- `source_url`
- `evidence`
- `confidence`
- `trace` (step-by-step decisions)

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Notes

- This MVP intentionally constrains the action space to reduce hallucinated browser operations.
- Orchestration is implemented with LangGraph and native tool-calling (`ToolNode`) instead of a manual `for` loop.
- The LLM makes navigation decisions directly; the runtime does not auto-rewrite actions.
- Use a Python version within the configured range (`>=3.11,<3.14`).
