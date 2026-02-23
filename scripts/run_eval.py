#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent.openrouter_client import OpenRouterClient
from agent.run import run_agent

ALLOWED_ACTIONS = {"extract", "type_and_submit", "click", "navigate", "fail"}


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    start_url: str
    target_prompt: str
    max_steps: int = 10
    max_actions_per_step: int = 1
    extraction_schema: dict[str, str] | None = None
    extraction_selector: str | None = None
    expected_contains: list[str] | None = None
    min_trace_steps: int | None = None
    required_actions: list[str] | None = None

    def is_complex(self) -> bool:
        if self.max_actions_per_step > 1:
            return True
        if self.required_actions:
            return True
        if self.min_trace_steps is not None and self.min_trace_steps > 1:
            return True
        return False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalTask":
        task_id = str(payload.get("id", "")).strip()
        if not task_id:
            raise ValueError("Task is missing non-empty 'id'")

        start_url = str(payload.get("start_url", "")).strip()
        if not start_url:
            raise ValueError(f"Task '{task_id}' is missing non-empty 'start_url'")

        target_prompt = str(payload.get("target_prompt", "")).strip()
        if not target_prompt:
            raise ValueError(f"Task '{task_id}' is missing non-empty 'target_prompt'")

        max_steps = int(payload.get("max_steps", 10))
        max_actions_per_step = int(payload.get("max_actions_per_step", 1))

        extraction_schema_raw = payload.get("extraction_schema")
        extraction_schema: dict[str, str] | None = None
        if extraction_schema_raw is not None:
            if not isinstance(extraction_schema_raw, dict):
                raise ValueError(f"Task '{task_id}' extraction_schema must be an object")
            extraction_schema = {}
            for key, value in extraction_schema_raw.items():
                normalized_key = str(key).strip()
                normalized_value = str(value).strip()
                if not normalized_key:
                    raise ValueError(f"Task '{task_id}' has empty extraction_schema key")
                if not normalized_value:
                    raise ValueError(
                        f"Task '{task_id}' has empty extraction_schema description for '{normalized_key}'"
                    )
                extraction_schema[normalized_key] = normalized_value

        extraction_selector_raw = payload.get("extraction_selector")
        extraction_selector = None
        if extraction_selector_raw is not None:
            extraction_selector = str(extraction_selector_raw).strip()
            if not extraction_selector:
                raise ValueError(f"Task '{task_id}' has empty extraction_selector")

        expected_contains_raw = payload.get("expected_contains")
        expected_contains = None
        if expected_contains_raw is not None:
            if not isinstance(expected_contains_raw, list):
                raise ValueError(f"Task '{task_id}' expected_contains must be a list")
            expected_contains = [str(item).strip() for item in expected_contains_raw if str(item).strip()]

        min_trace_steps_raw = payload.get("min_trace_steps")
        min_trace_steps: int | None = None
        if min_trace_steps_raw is not None:
            min_trace_steps = int(min_trace_steps_raw)
            if min_trace_steps < 1:
                raise ValueError(f"Task '{task_id}' min_trace_steps must be >= 1")

        required_actions_raw = payload.get("required_actions")
        required_actions: list[str] | None = None
        if required_actions_raw is not None:
            if not isinstance(required_actions_raw, list):
                raise ValueError(f"Task '{task_id}' required_actions must be a list")
            required_actions = []
            for item in required_actions_raw:
                action = str(item).strip()
                if not action:
                    continue
                if action not in ALLOWED_ACTIONS:
                    raise ValueError(
                        f"Task '{task_id}' required_actions contains invalid action '{action}'"
                    )
                required_actions.append(action)
            if not required_actions:
                required_actions = None

        return cls(
            task_id=task_id,
            start_url=start_url,
            target_prompt=target_prompt,
            max_steps=max_steps,
            max_actions_per_step=max_actions_per_step,
            extraction_schema=extraction_schema,
            extraction_selector=extraction_selector,
            expected_contains=expected_contains,
            min_trace_steps=min_trace_steps,
            required_actions=required_actions,
        )


def _load_tasks(path: Path) -> list[EvalTask]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Task file must contain a JSON array")

    tasks = [EvalTask.from_dict(item) for item in raw]
    if not tasks:
        raise ValueError("Task file must contain at least one task")
    return tasks


def _result_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("answer") or "",
        record.get("evidence") or "",
        json.dumps(record.get("structured_data") or {}, sort_keys=True),
    ]
    return "\n".join(parts).lower()


def _matches_expectations(record: dict[str, Any], expected_contains: list[str] | None) -> bool:
    if not expected_contains:
        return True
    blob = _result_text(record)
    return all(expected.lower() in blob for expected in expected_contains)


def _extract_actions_from_trace_payload(trace_payload: Any) -> list[str]:
    if not isinstance(trace_payload, list):
        return []

    actions: list[str] = []
    for item in trace_payload:
        if not isinstance(item, dict):
            continue
        decision = item.get("decision")
        if not isinstance(decision, dict):
            continue
        action = decision.get("action")
        if isinstance(action, str) and action:
            actions.append(action)
    return actions


def _first_expectation_failure(record: dict[str, Any], task: EvalTask) -> str | None:
    if not _matches_expectations(record, task.expected_contains):
        return "expected_contains_mismatch"

    if task.min_trace_steps is not None and int(record["trace_steps"]) < task.min_trace_steps:
        return (
            f"min_trace_steps_not_met:"
            f"expected>={task.min_trace_steps},actual={int(record['trace_steps'])}"
        )

    if task.required_actions:
        seen_actions = {str(action) for action in record.get("actions", [])}
        missing = [action for action in task.required_actions if action not in seen_actions]
        if missing:
            return f"required_actions_missing:{','.join(missing)}"

    return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, math.ceil((percentile / 100) * len(sorted_values)) - 1))
    return sorted_values[index]


async def _run_single(
    client: OpenRouterClient,
    task: EvalTask,
    repeat_index: int,
    total_repeats: int,
    *,
    headed: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = await run_agent(
        client,
        start_url=task.start_url,
        target_prompt=task.target_prompt,
        max_steps=task.max_steps,
        max_actions_per_step=task.max_actions_per_step,
        extraction_schema=task.extraction_schema,
        extraction_selector=task.extraction_selector,
        headless=not headed,
    )
    duration_s = time.perf_counter() - started

    record = {
        "task_id": task.task_id,
        "repeat": repeat_index + 1,
        "total_repeats": total_repeats,
        "is_complex": task.is_complex(),
        "success": result.error is None,
        "error": result.error,
        "duration_s": round(duration_s, 4),
        "trace_steps": len(result.trace),
        "actions": [item.decision.action for item in result.trace],
        "answer": result.answer,
        "structured_data": result.structured_data,
        "evidence": result.evidence,
        "source_url": result.source_url,
    }

    expectation_failure = _first_expectation_failure(record, task) if record["success"] else None
    if expectation_failure is not None:
        record["success"] = False
        record["error"] = expectation_failure

    status = "PASS" if record["success"] else "FAIL"
    print(
        f"[{repeat_index + 1}/{total_repeats}] {task.task_id}: {status} "
        f"steps={record['trace_steps']} duration={record['duration_s']}s"
    )
    return record


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> tuple[int, dict[str, Any] | str]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            status = int(resp.getcode())
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8") if exc.fp is not None else ""
    except URLError as exc:
        raise RuntimeError(f"request_failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"request_failed: {exc}") from exc

    try:
        return status, json.loads(raw)
    except Exception:
        return status, raw


async def _run_single_via_api(
    base_url: str,
    task: EvalTask,
    repeat_index: int,
    total_repeats: int,
    *,
    headed: bool,
    request_timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        status, response_body = await asyncio.to_thread(
            _post_json,
            f"{base_url}/run",
            {
                "start_url": task.start_url,
                "target_prompt": task.target_prompt,
                "max_steps": task.max_steps,
                "max_actions_per_step": task.max_actions_per_step,
                "extraction_schema": task.extraction_schema,
                "extraction_selector": task.extraction_selector,
                "headed": headed,
            },
            timeout_s=request_timeout_s,
        )
    except RuntimeError as exc:
        duration_s = time.perf_counter() - started
        record = {
            "task_id": task.task_id,
            "repeat": repeat_index + 1,
            "total_repeats": total_repeats,
            "is_complex": task.is_complex(),
            "success": False,
            "error": str(exc),
            "duration_s": round(duration_s, 4),
            "trace_steps": 0,
            "actions": [],
            "answer": None,
            "structured_data": None,
            "evidence": None,
            "source_url": None,
            "http_status": None,
        }
        print(
            f"[{repeat_index + 1}/{total_repeats}] {task.task_id}: FAIL "
            f"http=none steps=0 duration={record['duration_s']}s"
        )
        return record

    duration_s = time.perf_counter() - started

    result_payload: dict[str, Any] = {}
    error: str | None = None
    success = False

    if status == 200 and isinstance(response_body, dict):
        result_payload = response_body
        success = not bool(result_payload.get("error"))
        error = result_payload.get("error")
    elif status == 422 and isinstance(response_body, dict):
        detail = response_body.get("detail")
        if isinstance(detail, dict):
            result_payload = detail
            error = str(detail.get("error") or "http_422")
        else:
            error = f"http_422:{detail}"
    else:
        error = f"http_{status}"
        if isinstance(response_body, dict):
            detail = response_body.get("detail")
            if detail:
                error = f"{error}:{detail}"
        elif isinstance(response_body, str) and response_body.strip():
            error = f"{error}:{response_body.strip()[:160]}"

    record = {
        "task_id": task.task_id,
        "repeat": repeat_index + 1,
        "total_repeats": total_repeats,
        "is_complex": task.is_complex(),
        "success": success,
        "error": error,
        "duration_s": round(duration_s, 4),
        "trace_steps": int(len(result_payload.get("trace") or [])),
        "actions": _extract_actions_from_trace_payload(result_payload.get("trace")),
        "answer": result_payload.get("answer"),
        "structured_data": result_payload.get("structured_data"),
        "evidence": result_payload.get("evidence"),
        "source_url": result_payload.get("source_url"),
        "http_status": status,
    }

    expectation_failure = _first_expectation_failure(record, task) if record["success"] else None
    if expectation_failure is not None:
        record["success"] = False
        record["error"] = expectation_failure

    status_text = "PASS" if record["success"] else "FAIL"
    print(
        f"[{repeat_index + 1}/{total_repeats}] {task.task_id}: {status_text} "
        f"http={status} steps={record['trace_steps']} duration={record['duration_s']}s"
    )
    return record


async def _run_eval(
    tasks: list[EvalTask],
    repeats: int,
    *,
    headed: bool,
    api_base_url: str | None,
    request_timeout_s: float,
) -> list[dict[str, Any]]:
    client = None
    if api_base_url is None:
        client = OpenRouterClient.from_env()
    records: list[dict[str, Any]] = []

    for repeat_index in range(repeats):
        for task in tasks:
            if api_base_url is None:
                if client is None:
                    raise RuntimeError("missing_openrouter_client")
                record = await _run_single(
                    client,
                    task,
                    repeat_index,
                    repeats,
                    headed=headed,
                )
            else:
                record = await _run_single_via_api(
                    api_base_url,
                    task,
                    repeat_index,
                    repeats,
                    headed=headed,
                    request_timeout_s=request_timeout_s,
                )
            records.append(record)

    return records


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    success_records = [record for record in records if record["success"]]
    success_rate = (len(success_records) / len(records)) if records else 0.0

    durations = [float(record["duration_s"]) for record in records]
    successful_steps = [int(record["trace_steps"]) for record in success_records]
    complex_records = [record for record in records if bool(record.get("is_complex"))]
    complex_success_records = [record for record in complex_records if record["success"]]

    per_task: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = str(record["task_id"])
        bucket = per_task.setdefault(
            task_id,
            {"runs": 0, "successes": 0, "durations": [], "steps": [], "actions": []},
        )
        bucket["runs"] += 1
        bucket["durations"].append(float(record["duration_s"]))
        bucket["actions"].extend([str(action) for action in record.get("actions", [])])
        if record["success"]:
            bucket["successes"] += 1
            bucket["steps"].append(int(record["trace_steps"]))

    per_task_summary: dict[str, Any] = {}
    for task_id, bucket in sorted(per_task.items()):
        runs = int(bucket["runs"])
        successes = int(bucket["successes"])
        task_durations: list[float] = bucket["durations"]
        task_steps: list[int] = bucket["steps"]
        task_actions: list[str] = bucket["actions"]
        per_task_summary[task_id] = {
            "runs": runs,
            "successes": successes,
            "success_rate": round((successes / runs) if runs else 0.0, 4),
            "median_duration_s": round(statistics.median(task_durations), 4) if task_durations else None,
            "median_steps_success": statistics.median(task_steps) if task_steps else None,
            "unique_actions_seen": sorted(set(task_actions)),
        }

    return {
        "total_runs": len(records),
        "successful_runs": len(success_records),
        "success_rate": round(success_rate, 4),
        "complex_total_runs": len(complex_records),
        "complex_successful_runs": len(complex_success_records),
        "complex_success_rate": round(
            (len(complex_success_records) / len(complex_records)) if complex_records else 0.0,
            4,
        ),
        "median_duration_s": round(statistics.median(durations), 4) if durations else None,
        "p95_duration_s": round(_percentile(durations, 95) or 0.0, 4) if durations else None,
        "median_steps_success": statistics.median(successful_steps) if successful_steps else None,
        "per_task": per_task_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated evals for auto-browse agent")
    parser.add_argument(
        "--tasks",
        default="evals/tasks.json",
        help="Path to JSON task list (default: evals/tasks.json)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of runs per task (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for first N tasks (default: all)",
    )
    parser.add_argument(
        "--output",
        default=".context/eval_report.json",
        help="Where to write JSON report (default: .context/eval_report.json)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run in headed mode (default: headless)",
    )
    parser.add_argument(
        "--api-base-url",
        default="",
        help="Optional API base URL (e.g. http://127.0.0.1:8000). If set, evals run via /run.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds for API mode (default: 120)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    task_path = Path(args.tasks)
    output_path = Path(args.output)
    tasks = _load_tasks(task_path)
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    api_base_url = args.api_base_url.strip() or None
    if api_base_url is not None:
        api_base_url = api_base_url.rstrip("/")

    records = asyncio.run(
        _run_eval(
            tasks,
            args.repeats,
            headed=args.headed,
            api_base_url=api_base_url,
            request_timeout_s=args.request_timeout_s,
        )
    )
    summary = _build_summary(records)

    report = {
        "generated_at_unix": time.time(),
        "tasks_file": str(task_path),
        "repeats": args.repeats,
        "task_count": len(tasks),
        "mode": "api" if api_base_url else "local",
        "api_base_url": api_base_url,
        "summary": summary,
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print("")
    print("Eval summary:")
    print(json.dumps(summary, indent=2))
    print("")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
