# Auto Browse MVP

Minimal constrained browser agent:

1. Load page with Playwright.
2. Convert HTML to readable markdown (readability + markdownify).
3. Orchestrate the loop as a LangGraph state machine.
4. Ask an LLM for one or more tool calls each turn (configurable budget).
5. Execute those tools with a LangGraph tool-execution node (`extract_answer`, `type_and_submit`, `click`, `navigate`, `fail`).
6. Repeat until answer or failure.

At every step, the API logs:

- what the LLM understood on the current page
- what it will do next

## Setup

1. Install dependencies:
   ```bash
   python3.14 -m venv .venv  # python3.13 also supported
   source .venv/bin/activate
   pip install -e .
   playwright install chromium
   ```
2. Set environment variables:
   ```bash
   cp .env.example .env
   export OPENROUTER_API_KEY=...
   export OPENROUTER_MODEL=openai/gpt-4.1-mini
   export AUTO_BROWSE_API_TOKEN=replace-with-shared-token
   ```
   The runtime also auto-loads these keys from a local `.env` file if present.

## Run As API

Start the server:

```bash
./scripts/run_api.sh
```

or run `uvicorn` directly:

```bash
python -m uvicorn auto_browse.api:create_app --factory --host 127.0.0.1 --port 8000
```

Call it:

```bash
curl -X POST "http://127.0.0.1:8000/run" \
  -H "X-API-Token: replace-with-shared-token" \
  -H "Content-Type: application/json" \
  -d '{
    "start_url": "https://www.google.com",
    "target_prompt": "release date of Star Wars",
    "max_steps": 10,
    "max_actions_per_step": 2
  }'
```

Notes:

- The API uses env credentials/model only (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`).
- Startup requires `AUTO_BROWSE_API_TOKEN` (recommended in `.env`).
- Every request must include the shared token header (`X-API-Token`) matching `AUTO_BROWSE_API_TOKEN`.
- Built-in middleware adds basic DDoS controls configured in `config/security.toml`:
  - `max_request_body_bytes` (default `65536`)
  - `rate_limit_max_requests` per `rate_limit_window_seconds` (defaults `30` per `60s`)
  - `max_concurrent_requests_per_ip` (default `4`)
  - `api_token_header` (default `x-api-token`)
- `X-Forwarded-For` is ignored by default. To trust it behind a known proxy, set:
  - `trust_x_forwarded_for = true`
  - `trusted_proxy_cidrs = ["<proxy-cidr>"]`
- `AUTO_BROWSE_SECURITY_CONFIG_PATH` can point to a different config file path if needed.
- `start_url` accepts either a full URL (`https://...`) or a hostname (`www.google.com`), which is auto-normalized to `https://...`.
- The API always logs intermediary step summaries and next actions.
- `max_actions_per_step` controls how many tool calls the model may emit in one turn (`1..4`).
- Optional schema extraction:
  - `extraction_schema`: `{ "field_name": "field description" }`
  - `extraction_selector`: optional Playwright selector to scope extraction to a DOM subtree.
- Each run sends OpenRouter tracing metadata on every LLM step:
  - `trace.trace_id` is generated automatically (UUIDv7 fallback to UUID4).
  - `trace.generation_name` is set per step as `planner.1`, `planner.2`, ...
  - `session_id` is generated internally to match `trace_id`.

## Deploy (Render)

This repo includes a Render Blueprint at [`render.yaml`](render.yaml) and a Playwright-ready [`Dockerfile`](Dockerfile).

Steps:

1. Push this repo to GitHub.
2. In Render, create a new Blueprint service from the repo.
3. Set `OPENROUTER_API_KEY` in the Render dashboard before first deploy.
4. Deploy.

Render setup details:

- Runtime: Docker (`Dockerfile`)
- Start command: `./scripts/run_api.sh` (from container `CMD`)
- Health endpoint: `/health`
- Configured env vars:
  - `OPENROUTER_API_KEY` (secret, required)
  - `OPENROUTER_MODEL` (default `openai/gpt-4.1-mini`)
  - `AUTO_BROWSE_API_TOKEN` (auto-generated in `render.yaml`)

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
        max_actions_per_step=2,
        headless=True,
    )
    print(result.model_dump())


asyncio.run(main())
```

## Output

The `/run` response body contains:

- `answer`
- `structured_data` (present when schema extraction is used)
- `source_url`
- `evidence`
- `confidence`
- `trace` (step-by-step decisions)

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Evals

Run repeated benchmark evals with per-task metrics:

```bash
python scripts/run_eval.py --tasks evals/tasks.json --repeats 5 --output .context/eval_report.json
```

Run the complex workflow suite (multi-step + required action assertions):

```bash
python scripts/run_eval.py --tasks evals/tasks_complex.json --repeats 3 --output .context/eval_report_complex.json
```

Options:

- `--limit N` to run only the first `N` tasks.
- `--headed` to run with a visible browser.
- `--api-base-url http://127.0.0.1:8000` to run evals through the running API server.

Task schema supports optional behavior assertions:

- `min_trace_steps`: minimum required action count in the resulting trace.
- `required_actions`: list of required actions that must appear in the trace.

## Documented Successes and Failures (Observed)

The following results were observed in API eval runs on **February 23, 2026**.
They are run logs from this repo's current implementation, not permanent guarantees.

### Complex Suite (`evals/tasks_complex.json`, Google-start)

- Full run report: `.context/eval_report_google_start_complex_api_full.json`
- Result: **8/10 passed** (`80%` success, `repeats=1`, API mode).
- Step behavior in successful runs:
  - `median_steps_success = 2`
  - Multi-step examples include `type_and_submit + click + extract` and `navigate + extract`.
- Successful examples:
  - `ddg_search_openai_docs_title` -> `type_and_submit + click + extract` -> `OpenAI API Platform Documentation`
  - `ddg_search_python_wiki_title` -> `type_and_submit + click + extract` -> `Python (programming language)`
  - `navigate_then_schema_star_wars` -> `navigate + extract` with structured infobox fields
  - `schema_python_infobox` -> structured extraction (`designer`, `first_appeared`)
- Failures in this run:
  - `example_click_to_iana_title` -> `click_failed`
  - `wikipedia_search_un_title` -> `max_steps_exceeded` after repeated navigation/click attempts

### Fresh-State Fix Sanity Slice (first 5 complex tasks)

- Report: `.context/eval_report_google_start_complex_limit5_after_fresh_state_fix.json`
- Result: **4/5 passed** (`80%` success, API mode).
- Shows stable passes on:
  - direct navigation tasks
  - Google search + click workflows
- Remaining miss in this slice:
  - `example_click_to_iana_title` (still unreliable on click-driven transition)

### What This Means

- The agent now handles substantially more multi-step tasks than earlier runs, including search-driven flows.
- The remaining primary gap is robust click-driven progression on some pages, which is the highest-value next reliability target.

## Notes

- This MVP intentionally constrains the action space to reduce hallucinated browser operations.
- Orchestration is implemented with LangGraph state nodes instead of a manual `for` loop.
- The LLM makes navigation decisions directly; the runtime does not auto-rewrite actions.
- Use a Python version within the configured range (`>=3.11,<3.15`).

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file.
