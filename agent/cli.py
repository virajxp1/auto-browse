from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agent.models import AgentStepTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run constrained browser-agent MVP.")
    parser.add_argument("--start-url", required=True, help="Starting URL.")
    parser.add_argument("--target-prompt", required=True, help="Answer target.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Max agent loop iterations.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode (default is headless).",
    )
    return parser.parse_args()


async def _run() -> int:
    args = parse_args()

    from agent.openrouter_client import OpenRouterClient
    from agent.run import run_agent

    def _print_step(trace_item: AgentStepTrace) -> None:
        print(
            f"[step {trace_item.step}] summary: {trace_item.decision.step_summary}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[step {trace_item.step}] next: {trace_item.decision.next_step}",
            file=sys.stderr,
            flush=True,
        )

    client = OpenRouterClient.from_env()
    result = await run_agent(
        client,
        start_url=args.start_url,
        target_prompt=args.target_prompt,
        max_steps=args.max_steps,
        headless=not args.headed,
        on_step=_print_step,
    )
    print(json.dumps(result.model_dump(), indent=2))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
