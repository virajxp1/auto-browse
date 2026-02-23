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
    "max_steps": 10,
    "max_actions_per_step": 2
  }'
```

Notes:

- The API uses env credentials/model only (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`).
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

The following results were observed in local API eval runs on **February 21, 2026**.
They are run logs from this repo's current implementation, not permanent guarantees.

### Baseline Suite (`evals/tasks.json`)

- Result: **18/20 passed** (`90%` success, `repeats=1`, API mode).
- Typical behavior:
  - Most successful tasks complete in **1 step** (`extract`) on already-correct pages.
  - Examples:
    - `example_domain_heading` -> `Example Domain`
    - `wiki_star_wars_title` -> `Star Wars (film)`
    - `python_infobox_schema` -> structured extraction from infobox
- Failures in this run:
  - `wiki_nasa_title` -> `422`, `tool_execution_failed`
  - `star_wars_infobox_schema` -> `422`, `Infobox content not present in provided page text`

### Complex Suite (`evals/tasks_complex.json`)

- Result: **4/10 passed** (`40%` success, `repeats=1`, API mode).
- This suite enforces multi-step/behavior constraints (`min_trace_steps`, `required_actions`), and does trigger deeper traces:
  - Step distribution in one run: `{1: 1, 2: 7, 3: 1, 8: 1}`
  - Maximum observed trace depth: `8` steps
- Successful complex examples:
  - `example_navigate_to_iana_title` -> `navigate + extract` (2 steps)
  - `iana_navigate_to_example_heading` -> `navigate + extract` (2 steps)
  - `wikipedia_search_un_title` -> `type_and_submit + extract` (2 steps)
- Where it currently falls short:
  - `example_click_to_iana_title` -> `click` followed by `navigate_failed`
  - `ddg_search_openai_docs_title` and `ddg_search_python_wiki_title` -> `type_and_submit_failed`
  - `wikipedia_search_nasa_title` -> `max_steps_exceeded` after repeated actions
  - `schema_star_wars_infobox` and `navigate_then_schema_star_wars` returned HTTP `200` but failed eval expectations (`expected_contains_mismatch`)

### What This Means

- The agent is strong on straightforward extraction and some navigation flows.
- Complex, search-driven, and selector-sensitive workflows are still the main reliability gap.
- The complex suite now measures these gaps directly and is the better indicator for progress on multi-step capability.

## Notes

- This MVP intentionally constrains the action space to reduce hallucinated browser operations.
- Orchestration is implemented with LangGraph state nodes instead of a manual `for` loop.
- The LLM makes navigation decisions directly; the runtime does not auto-rewrite actions.
- Use a Python version within the configured range (`>=3.11,<3.15`).
