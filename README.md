# Auto Browse MVP

Minimal constrained browser agent:

1. Load page with Playwright.
2. Convert HTML to readable markdown (readability + markdownify).
3. Orchestrate the loop as a LangGraph state machine.
4. Ask an LLM for exactly one tool call each turn.
5. Execute that tool with LangGraph `ToolNode` (`extract_answer`, `type_and_submit`, `click`, `navigate`, `fail`).
6. Repeat until answer or failure.

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

## Run API Server

Start FastAPI server:

```bash
./scripts/run_api.sh
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Run one agent job:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"start_url":"https://www.google.com","target_prompt":"release date of Star Wars","max_steps":10}'
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
        max_steps=10,
        headless=True,
    )
    print(result.model_dump())


asyncio.run(main())
```

## Output

The API `/run` returns JSON:

- `answer`
- `source_url`
- `evidence`
- `confidence`
- `trace` (step-by-step decisions)

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Documented Successes and Failures (Observed)

The following results were observed in local API eval runs on **February 21, 2026**.
They are run logs from this repo's current implementation, not permanent guarantees.

### Successes

- Baseline expected-success matrix: **8/8 passed** (all HTTP `200`).
  - Examples:
    - Wikipedia extraction: Star Wars theatrical release date -> `May 25, 1977`
    - Example -> navigate -> IANA title extraction -> `IANA-managed Reserved Domains`
    - DuckDuckGo search flow -> OpenAI docs title extraction -> `OpenAI API Platform Documentation`
    - IMDb moviemeter top title extraction -> `Wuthering Heights` (as observed during run)
- Multi-step complexity demonstrated:
  - The most complex passing case in this set was DuckDuckGo -> docs extraction with:
    - `6` steps
    - `4` unique tool actions (`type_and_submit`, `click`, `navigate`, `extract`)

### Failures

- Before the MVP action-outcome verifier change, mixed evals included expected reliability misses:
  - BooksToScrape "find cheapest Travel book" -> failed (`422`, insufficient extraction/comparison evidence)
  - GitHub Trending "first repo" -> failed (`422`, `click_failed`)
- In an earlier 7-test matrix, result was **6/7 passed** with the same BooksToScrape cheapest-book task failing.
- In a separate mixed 6-test matrix, result was **4/6 passed** (main failures were GitHub Trending click reliability and BooksToScrape comparison extraction).

### What improved

- Added MVP action-outcome verification for `click`, `type_and_submit`, and `navigate`.
- Post-change test suite status:
  - `tests/test_star_wars_example.py`: `8` passed
  - full suite: `32` passed
- Post-change expected-success eval matrix improved to **8/8 passed**.

## Notes

- This MVP intentionally constrains the action space to reduce hallucinated browser operations.
- Orchestration is implemented with LangGraph and native tool-calling (`ToolNode`) instead of a manual `for` loop.
- The LLM makes navigation decisions directly; the runtime does not auto-rewrite actions.
- Use a Python version within the configured range (`>=3.11,<3.15`).
