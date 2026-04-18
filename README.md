# Auto Browse MVP

Constrained browser agent for discrete web tasks:

1. Start a Playwright browser session and capture a page snapshot.
2. Build a cached page state with URL, title, markdown, interactables, and grounded refs.
3. Orchestrate the loop as a LangGraph state machine.
4. Take exactly one typed browser action per step.
5. Use grounded shortcuts plus a planner LLM to choose the next action.
6. Use Deep Agents as an analysis/verifier layer for ambiguous pages and goal completion checks.
7. Repeat until answer, verified completion, or failure.

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
   export AUTO_BROWSE_API_TOKEN=replace-with-shared-token
   # Optional runtime tracing:
   export BRAINTRUST_API_KEY=...
   ```
   The runtime also auto-loads these keys from a local `.env` file if present.
3. Set the model in `config/config.ini`:
   ```ini
   [openrouter]
   model = gpt-oss-20b

   [braintrust]
   project_id = 11726530-f1d2-4490-b9ba-7d8996e37880
   ```

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
    "max_actions_per_step": 1
  }'
```

Notes:

- The API reads OpenRouter credentials from env (`OPENROUTER_API_KEY`) and model from config with this precedence:
  - `AUTO_BROWSE_OPENROUTER_CONFIG_PATH`, if set
  - local `config/config.ini`, if present
  - packaged default bundled with `agent` (wheel-friendly fallback)
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
- `/run` enforces a cooldown of 1 accepted request every 20 seconds (HTTP `429` with `Retry-After` when exceeded).
- `max_actions_per_step` is currently fixed to `1`.
- `max_runtime_seconds` caps total wall-clock runtime for the whole agent run (default `90`).
- Optional goal-oriented fields:
  - `goal_type`: high-level mode such as `generic`, `search`, `signup`, or `extract`
  - `task_data`: non-secret task inputs like `{ "city": "Madrid", "guests": "2" }`
  - `sensitive_data`: secret inputs like passwords; these are redacted in request logs
  - Generic goal completion now supports a verifier step via `verify_goal`, and `complete_goal` is expected only after success has been checked on the current page.
- Optional schema extraction:
  - `extraction_schema`: `{ "field_name": "field description" }`
  - `extraction_selector`: optional Playwright selector to scope extraction to a DOM subtree.
- Optional distributed tracing parent:
  - `X-BT-Parent`: parent span token from `current_span().export()` in an upstream Braintrust-instrumented service.
- Braintrust runtime traces are enabled when:
  - `BRAINTRUST_API_KEY` is set, and
  - `[braintrust].project_id` is set in `config/config.ini`.
  - Each run creates one root span: `agent.run`.
  - Child spans are emitted for `startup.browser`, `capture.N`, `llm.N`, `execute_tools.N`/`tool.*`, and `post_tool.N`.
  - Root span metadata includes `run_id`, matching the internal OpenRouter `trace_id`.
- Each run sends OpenRouter tracing metadata on every LLM step:
  - `trace.trace_id` is generated automatically (UUIDv7 fallback to UUID4).
  - `trace.generation_name` is set per step as `planner.1`, `planner.2`, ...
  - `session_id` is generated internally to match `trace_id`.

## Deploy (Render)

Deploy this repo as a standard Render **Web Service** using the included [`Dockerfile`](Dockerfile).

1. Push this repo to GitHub.
2. In Render, click **New +** -> **Web Service**.
3. Connect your repo and choose the branch to deploy.
4. In service settings:
   - Environment: `Docker`
   - Health Check Path: `/health`
   - Auto-Deploy: your preference
5. Add environment variables:
   - `OPENROUTER_API_KEY` (required)
   - `AUTO_BROWSE_API_TOKEN` (required shared token for API requests)
   - `BRAINTRUST_API_KEY` (optional, for run tracing)
6. Ensure your deployed config sets `config/config.ini`:
   ```ini
   [openrouter]
   model = gpt-oss-20b

   [braintrust]
   project_id = 11726530-f1d2-4490-b9ba-7d8996e37880
   ```
   You can override this path with `AUTO_BROWSE_OPENROUTER_CONFIG_PATH`.
7. Click **Create Web Service** and wait for deploy to complete.

Render uses the container `CMD` from the Dockerfile, which runs `./scripts/run_api.sh`.

Quick verification after deploy:

```bash
curl https://<your-render-url>/health
```

Example API call:

```bash
curl -X POST "https://<your-render-url>/run" \
  -H "X-API-Token: <AUTO_BROWSE_API_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "start_url": "https://www.google.com",
    "target_prompt": "release date of Star Wars",
    "max_steps": 10,
    "max_actions_per_step": 1
  }'
```
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
        goal_type="extract",
        max_steps=10,
        max_actions_per_step=1,
        max_runtime_seconds=90,
        headless=True,
    )
    print(result.model_dump())


asyncio.run(main())
```

## Output

The `/run` response body contains:

- `status`
- `goal_summary`
- `result_data`
- `answer`
- `structured_data` (present when schema extraction is used)
- `source_url`
- `final_url`
- `final_title`
- `evidence`
- `confidence`
- `trace` (step-by-step decisions)

## Runtime Architecture

The current runtime is organized around a few core pieces:

- `agent/run.py`: LangGraph execution loop, typed tool handlers, grounded shortcuts, planner fallback, and Braintrust span emission.
- `agent/snapshot.py`: page snapshot capture, caching, invalidation, and fallback capture paths.
- `agent/browser.py`: Playwright startup plus interactable extraction and ref assignment.
- `agent/planner.py`: planner prompt + tool-call message construction.
- `agent/deep_advisor.py`: Deep Agents-backed page analysis and goal verification.
- `agent/task_intent.py`: generic search/result progression hints used before planner invocation.

Behavior notes:

- The planner no longer executes multiple tools in one turn. Each step resolves to one tool call.
- Interactables are exposed with stable `interactable_ref` values, and planner/runtime prefer refs over invented selectors.
- The runtime can skip the main planner LLM call when a grounded shortcut is obvious from the current snapshot, for example:
  - submitting a visible site search field
  - opening the strongest visible result link
  - applying the latest page analysis result directly
- Snapshot capture is cached per page state and invalidated after actions that change the page.
- `analyze_page` and `verify_goal` are part of the runtime loop, not external manual utilities.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Evals

The eval runner uses Braintrust `EvalAsync`, pulls eval cases from a Braintrust dataset, executes the local library code, and writes results back to Braintrust.

Run evals from the Braintrust dataset:

```bash
python scripts/run_eval.py \
  --repeats 1 \
  --output .context/eval_report.json
```

Auto-evals scorers used in each run:

- `coherence` (custom `autoevals.LLMClassifier`)
- `factuality` (`autoevals.Factuality`)

The run fails automatically when `overall_score` is below `min_overall_score` from `config/config.ini`.

All test cases are managed directly in Braintrust dataset `dataset_name` from `config/config.ini`.

Options:

- `--repeats N` to run each dataset record `N` times.
- `--limit N` to run only the first `N` dataset records.
- `--max-concurrency N` to control concurrent task execution in Braintrust (default: `1`).
- `--output <path>` to write the JSON report.

By default, Braintrust eval settings are read from `config/config.ini` under `[braintrust]`
(`project_name`, `project_id`, `dataset_name`, `min_overall_score`, `autoevals_model`).

Auto-evals scorers use `OPENROUTER_API_KEY`.

Task schema supports optional behavior assertions:

- `min_trace_steps`: minimum required action count in the resulting trace.
- `required_actions`: list of required actions that must appear in the trace.

## Documented Successes and Failures (Observed)

The following are point-in-time eval results from this repo's current implementation, not permanent guarantees.

### Braintrust Dataset Runs

Representative Braintrust-rooted reports:

- `.context/eval_report_full_expanded_suite_braintrust_2026-03-30.json`
- `.context/eval_report_full_expanded_suite_braintrust_2026-04-04_after_anchor_enrichment.json`

Recent observed broad-suite behavior on **April 4, 2026**:

- Best full Braintrust run in the current iteration: `15/21`, `overall_score=0.7048`
- Results are still stochastic across heavier search/docs/retail sites
- Current main gaps are search-driven progression and first-turn reliability on some JS-heavy pages

### What This Means

- Eval source of truth is the Braintrust dataset, not committed JSON fixtures.
- The main quality gap is no longer generic extraction; it is reliable multi-step progression on search-heavy and JS-heavy sites.

## Notes

- This MVP intentionally constrains the action space to reduce hallucinated browser operations.
- Orchestration is implemented with LangGraph state nodes instead of a manual `for` loop.
- The runtime may short-circuit the planner with grounded next-step actions when the current snapshot makes the move obvious.
- Use a Python version within the configured range (`>=3.11,<3.15`).

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file.
