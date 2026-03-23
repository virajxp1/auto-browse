#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import configparser
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_ACTIONS = {"extract", "type_and_submit", "click", "navigate", "fail"}
DEFAULT_EVAL_CONFIG_PATH = "config/config.ini"
DEFAULT_DATASET_NAME = "auto-browse-eval-cases"
DEFAULT_MIN_OVERALL_SCORE = 0.75
DEFAULT_AUTOEVALS_MODEL = "openai/gpt-4o-mini"

COHERENCE_PROMPT = """\
You are evaluating the coherence of an autonomous browser agent result.
Judge whether the answer is internally consistent, understandable, and not self-contradictory.
Ignore factual correctness.

Task:
{{input}}

Agent Output:
{{output}}

Choose exactly one option:
- coherent
- incoherent
"""


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


def _resolve_eval_config_path() -> Path:
    return Path(DEFAULT_EVAL_CONFIG_PATH)


def _load_env_file_if_present(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _read_braintrust_settings(path: Path) -> dict[str, str | None]:
    if not path.is_file():
        raise ValueError(f"Missing config file: {path}")

    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error as exc:
        raise ValueError(f"Failed to parse eval config file: {path}") from exc

    if "braintrust" not in parser:
        raise ValueError(f"Config file must include [braintrust]: {path}")

    project_name = parser.get("braintrust", "project_name", fallback="").strip()
    project_id = parser.get("braintrust", "project_id", fallback="").strip()
    dataset_name = parser.get("braintrust", "dataset_name", fallback=DEFAULT_DATASET_NAME).strip()
    min_overall_score = parser.get("braintrust", "min_overall_score", fallback=str(DEFAULT_MIN_OVERALL_SCORE)).strip()
    autoevals_model = parser.get("braintrust", "autoevals_model", fallback=DEFAULT_AUTOEVALS_MODEL).strip()

    if not project_name:
        raise ValueError("Config missing [braintrust].project_name")
    if not project_id:
        raise ValueError("Config missing [braintrust].project_id")

    return {
        "project_name": project_name,
        "project_id": project_id,
        "dataset_name": dataset_name,
        "min_overall_score": min_overall_score,
        "autoevals_model": autoevals_model,
    }


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


def _task_id_from_eval_case(case: dict[str, Any]) -> str:
    metadata = case.get("metadata")
    if isinstance(metadata, dict):
        task_id = str(metadata.get("task_id") or "").strip()
        if task_id:
            return task_id

    payload = case.get("input")
    if isinstance(payload, dict):
        task_id = str(payload.get("id") or "").strip()
        if task_id:
            return task_id

    task_id = str(case.get("id") or "").strip()
    return task_id or "unknown_task"


def _normalize_dataset_case(raw: dict[str, Any]) -> dict[str, Any] | None:
    task_payload = raw.get("input")
    if not isinstance(task_payload, dict):
        return None

    task_id = str(task_payload.get("id") or raw.get("id") or "").strip()
    if task_id and not str(task_payload.get("id") or "").strip():
        task_payload = dict(task_payload)
        task_payload["id"] = task_id

    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    if task_id:
        metadata.setdefault("task_id", task_id)

    expected = raw.get("expected")
    if expected is None and isinstance(raw.get("output"), (str, dict, list)):
        expected = raw.get("output")

    tags = raw.get("tags")
    if not isinstance(tags, list):
        tags = []

    return {
        "input": task_payload,
        "expected": expected,
        "metadata": metadata,
        "tags": [str(tag) for tag in tags if str(tag).strip()],
    }


def _fetch_eval_cases_from_dataset(dataset: Any, *, limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for raw in dataset.fetch():
        if not isinstance(raw, dict):
            continue
        normalized = _normalize_dataset_case(raw)
        if normalized is None:
            continue
        cases.append(normalized)

    cases.sort(key=_task_id_from_eval_case)
    if limit > 0:
        cases = cases[:limit]
    return cases


def _reference_text_from_expected(input_payload: Any, expected_payload: Any) -> str | None:
    if isinstance(expected_payload, str) and expected_payload.strip():
        return expected_payload.strip()

    if isinstance(expected_payload, dict):
        reference_text = str(expected_payload.get("reference_text") or "").strip()
        if reference_text:
            return reference_text
        expected_contains = expected_payload.get("expected_contains")
        if isinstance(expected_contains, list):
            joined = "\n".join([str(item).strip() for item in expected_contains if str(item).strip()]).strip()
            if joined:
                return joined

    if isinstance(input_payload, dict):
        expected_contains = input_payload.get("expected_contains")
        if isinstance(expected_contains, list):
            joined = "\n".join([str(item).strip() for item in expected_contains if str(item).strip()]).strip()
            if joined:
                return joined

    return None


def _resolve_autoevals_settings(*, configured_model: str) -> dict[str, str | None]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or os.getenv("OPEN_ROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY for eval scorers.")

    return {
        "api_key": api_key,
        "base_url": "https://openrouter.ai/api/v1",
        "model": configured_model or DEFAULT_AUTOEVALS_MODEL,
    }


def _compute_overall_score(records: list[dict[str, Any]]) -> dict[str, Any]:
    tracked = ["score_success", "coherence", "factuality"]
    per_metric: dict[str, float | None] = {}
    for metric in tracked:
        values: list[float] = []
        for record in records:
            score_map = record.get("scores")
            if not isinstance(score_map, dict):
                continue
            value = score_map.get(metric)
            if value is None:
                continue
            values.append(float(value))
        per_metric[metric] = round(statistics.mean(values), 4) if values else None

    available = [value for value in per_metric.values() if value is not None]
    overall = round(statistics.mean(available), 4) if available else None
    return {
        "overall_score": overall,
        "overall_components": per_metric,
    }


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


def _normalize_record(record: dict[str, Any], *, total_repeats: int) -> dict[str, Any]:
    normalized = {
        "task_id": str(record.get("task_id") or "unknown_task"),
        "repeat": int(record.get("repeat") or 1),
        "total_repeats": int(record.get("total_repeats") or total_repeats),
        "is_complex": bool(record.get("is_complex")),
        "success": bool(record.get("success")),
        "error": record.get("error"),
        "duration_s": round(float(record.get("duration_s") or 0.0), 4),
        "trace_steps": int(record.get("trace_steps") or 0),
        "actions": [str(action) for action in (record.get("actions") or [])],
        "answer": record.get("answer"),
        "structured_data": record.get("structured_data"),
        "evidence": record.get("evidence"),
        "source_url": record.get("source_url"),
    }
    if "http_status" in record:
        normalized["http_status"] = record.get("http_status")
    if "scores" in record and isinstance(record.get("scores"), dict):
        normalized["scores"] = {str(name): score for name, score in record["scores"].items()}
    return normalized


def _set_hook_metadata(hooks: Any, task: EvalTask, record: dict[str, Any]) -> None:
    metadata = getattr(hooks, "metadata", None)
    if isinstance(metadata, dict):
        metadata.update(
            {
                "task_id": task.task_id,
                "repeat": int(record.get("repeat") or 1),
                "total_repeats": int(record.get("total_repeats") or 1),
                "is_complex": bool(record.get("is_complex")),
                "success": bool(record.get("success")),
                "error": record.get("error"),
                "duration_s": float(record.get("duration_s") or 0.0),
                "trace_steps": int(record.get("trace_steps") or 0),
                "actions": list(record.get("actions") or []),
            }
        )

    tags = getattr(hooks, "tags", None)
    if isinstance(tags, list) and task.is_complex() and "complex" not in tags:
        tags.append("complex")


def _initial_task_record(task: EvalTask, *, repeat_index: int, total_repeats: int) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "repeat": repeat_index + 1,
        "total_repeats": total_repeats,
        "is_complex": task.is_complex(),
        "success": False,
        "error": None,
        "duration_s": 0.0,
        "trace_steps": 0,
        "actions": [],
        "answer": None,
        "structured_data": None,
        "evidence": None,
        "source_url": None,
    }


def _normalize_repeat_from_metadata(metadata: dict[str, Any]) -> int:
    explicit_repeat = metadata.get("repeat")
    if explicit_repeat not in (None, ""):
        try:
            parsed_repeat = int(explicit_repeat)
            if parsed_repeat >= 1:
                return parsed_repeat
        except Exception:
            pass

    try:
        trial_index = int(metadata.get("trial_index") or 0)
    except Exception:
        trial_index = 0
    return max(1, trial_index + 1)


def _zero_score_error_handler(_span: Any, _datum: Any, unhandled_scores: list[str]) -> dict[str, float]:
    return {str(score_name): 0.0 for score_name in unhandled_scores}


def _record_from_eval_result(result: Any, *, total_repeats: int) -> dict[str, Any]:
    output = getattr(result, "output", None)
    if isinstance(output, dict):
        record = dict(output)
    else:
        input_payload = getattr(result, "input", {})
        metadata = getattr(result, "metadata", {}) or {}
        if not isinstance(input_payload, dict):
            input_payload = {}
        if not isinstance(metadata, dict):
            metadata = {}

        repeat = _normalize_repeat_from_metadata(metadata)
        task_id = str(metadata.get("task_id") or input_payload.get("id") or "unknown_task")
        record = {
            "task_id": task_id,
            "repeat": repeat,
            "total_repeats": int(metadata.get("total_repeats") or total_repeats),
            "is_complex": bool(metadata.get("is_complex")),
            "success": False,
            "error": "task_failed_without_output",
            "duration_s": 0.0,
            "trace_steps": 0,
            "actions": [],
            "answer": None,
            "structured_data": None,
            "evidence": None,
            "source_url": None,
        }

    err = getattr(result, "error", None)
    if err is not None:
        record["success"] = False
        if not record.get("error"):
            record["error"] = str(err)

    scores = getattr(result, "scores", None)
    if isinstance(scores, dict):
        record["scores"] = {str(name): score for name, score in scores.items()}

    return _normalize_record(record, total_repeats=total_repeats)


def _records_from_eval_results(results: list[Any], *, total_repeats: int) -> list[dict[str, Any]]:
    records = [_record_from_eval_result(result, total_repeats=total_repeats) for result in results]
    records.sort(key=lambda record: (int(record["repeat"]), str(record["task_id"])))
    return records


async def _run_single(
    client: Any,
    task: EvalTask,
    repeat_index: int,
    total_repeats: int,
) -> dict[str, Any]:
    from agent.run import run_agent

    started = time.perf_counter()
    result = await run_agent(
        client,
        start_url=task.start_url,
        target_prompt=task.target_prompt,
        max_steps=task.max_steps,
        max_actions_per_step=task.max_actions_per_step,
        extraction_schema=task.extraction_schema,
        extraction_selector=task.extraction_selector,
        headless=True,
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
    return _normalize_record(record, total_repeats=total_repeats)


async def _run_eval(
    eval_cases: list[dict[str, Any]],
    repeats: int,
    *,
    braintrust_project: str,
    braintrust_project_id: str | None,
    max_concurrency: int,
    autoeval_settings: dict[str, str | None],
) -> list[dict[str, Any]]:
    try:
        from braintrust import EvalAsync
    except Exception as exc:
        raise RuntimeError(
            "Braintrust SDK is required to run evals. Install it with `pip install braintrust`."
        ) from exc
    try:
        from autoevals import Factuality, LLMClassifier
    except Exception as exc:
        raise RuntimeError(
            "autoevals is required to run coherence/factuality scorers. Install it with `pip install autoevals`."
        ) from exc

    from agent.openrouter_client import OpenRouterClient

    client = OpenRouterClient.from_env()

    autoeval_api_key = str(autoeval_settings.get("api_key") or "").strip()
    if not autoeval_api_key:
        raise ValueError("Missing autoeval API key")
    autoeval_model = str(autoeval_settings.get("model") or DEFAULT_AUTOEVALS_MODEL)
    autoeval_base_url = autoeval_settings.get("base_url")

    coherence_scorer = LLMClassifier(
        name="coherence",
        prompt_template=COHERENCE_PROMPT,
        choice_scores={"coherent": 1.0, "incoherent": 0.0},
        model=autoeval_model,
        use_cot=False,
        api_key=autoeval_api_key,
        base_url=str(autoeval_base_url) if autoeval_base_url else None,
    )
    factuality_scorer = Factuality(
        model=autoeval_model,
        use_cot=False,
        api_key=autoeval_api_key,
        base_url=str(autoeval_base_url) if autoeval_base_url else None,
    )

    async def run_eval_task(task_payload: dict[str, Any], hooks: Any) -> dict[str, Any]:
        task = EvalTask.from_dict(task_payload)
        trial_index = int(getattr(hooks, "trial_index", 0))
        _set_hook_metadata(
            hooks,
            task,
            _initial_task_record(task, repeat_index=trial_index, total_repeats=repeats),
        )

        record = await _run_single(
            client,
            task,
            trial_index,
            repeats,
        )

        _set_hook_metadata(hooks, task, record)
        return record

    def score_success(_input: Any, output: Any, expected: Any = None, **kwargs: Any) -> float:
        _ = expected
        _ = kwargs
        return 1.0 if isinstance(output, dict) and bool(output.get("success")) else 0.0

    async def score_coherence(input: Any, output: Any, expected: Any = None, **kwargs: Any) -> dict[str, Any]:
        _ = expected
        _ = kwargs
        output_text = _result_text(output) if isinstance(output, dict) else str(output or "")
        if not output_text.strip():
            return {"name": "coherence", "score": 0.0, "metadata": {"reason": "empty_output"}}

        input_text = ""
        if isinstance(input, dict):
            input_text = str(input.get("target_prompt") or input.get("id") or "")

        score = await coherence_scorer.eval_async(output=output_text, input=input_text)
        return {"name": "coherence", "score": score.score, "metadata": score.metadata}

    async def score_factuality(input: Any, output: Any, expected: Any = None, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        output_text = _result_text(output) if isinstance(output, dict) else str(output or "")
        if not output_text.strip():
            return {"name": "factuality", "score": 0.0, "metadata": {"reason": "empty_output"}}

        reference_text = _reference_text_from_expected(input, expected)
        if not reference_text:
            return {"name": "factuality", "score": None, "metadata": {"reason": "missing_reference"}}

        score = await factuality_scorer.eval_async(output=output_text, expected=reference_text)
        return {"name": "factuality", "score": score.score, "metadata": score.metadata}

    result = await EvalAsync(
        name=braintrust_project,
        project_id=braintrust_project_id,
        experiment_name=None,
        data=lambda: eval_cases,
        task=run_eval_task,
        scores=[score_success, score_coherence, score_factuality],
        trial_count=repeats,
        no_send_logs=False,
        max_concurrency=max_concurrency,
        error_score_handler=_zero_score_error_handler,
        metadata={
            "repeats": repeats,
            "data_source": "dataset",
            "autoevals_model": autoeval_model,
        },
    )

    raw_results = list(getattr(result, "results", []))
    return _records_from_eval_results(raw_results, total_repeats=repeats)


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
    parser = argparse.ArgumentParser(description="Run Braintrust dataset evals for auto-browse.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of runs per task (default: 1)",
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
        "--max-concurrency",
        type=int,
        default=1,
        help="Max concurrent eval task executions in Braintrust (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")

    _load_env_file_if_present()
    eval_config_path = _resolve_eval_config_path()
    braintrust_settings = _read_braintrust_settings(eval_config_path)
    braintrust_project = str(braintrust_settings.get("project_name") or "")
    braintrust_project_id = str(braintrust_settings.get("project_id") or "")
    dataset_name = str(braintrust_settings.get("dataset_name") or DEFAULT_DATASET_NAME)
    min_overall_score = float(str(braintrust_settings.get("min_overall_score") or DEFAULT_MIN_OVERALL_SCORE))
    autoeval_model = str(braintrust_settings.get("autoevals_model") or DEFAULT_AUTOEVALS_MODEL)
    if min_overall_score < 0 or min_overall_score > 1:
        raise ValueError("Config [braintrust].min_overall_score must be between 0 and 1")

    output_path = Path(args.output)

    try:
        from braintrust import init_dataset
    except Exception as exc:
        raise RuntimeError(
            "Braintrust SDK is required to initialize datasets. Install it with `pip install braintrust`."
        ) from exc

    dataset = init_dataset(
        project=braintrust_project,
        project_id=braintrust_project_id,
        name=dataset_name,
        use_output=False,
    )

    eval_cases = _fetch_eval_cases_from_dataset(dataset, limit=args.limit if args.limit > 0 else 0)
    if not eval_cases:
        raise ValueError(
            f"No eval records found in Braintrust dataset '{dataset_name}'. "
            "Seed/manage test cases directly in Braintrust."
        )

    autoeval_settings = _resolve_autoevals_settings(
        configured_model=autoeval_model,
    )

    records = asyncio.run(
        _run_eval(
            eval_cases,
            args.repeats,
            braintrust_project=braintrust_project,
            braintrust_project_id=braintrust_project_id,
            max_concurrency=args.max_concurrency,
            autoeval_settings=autoeval_settings,
        )
    )
    summary = _build_summary(records)
    summary.update(_compute_overall_score(records))
    overall_score = summary.get("overall_score")

    report = {
        "generated_at_unix": time.time(),
        "source": "dataset",
        "dataset_name": dataset_name,
        "repeats": args.repeats,
        "task_count": len(eval_cases),
        "mode": "local",
        "braintrust": {
            "config_path": str(eval_config_path),
            "project": braintrust_project,
            "project_id": braintrust_project_id,
            "max_concurrency": args.max_concurrency,
            "autoevals_model": autoeval_settings.get("model"),
        },
        "thresholds": {"min_overall_score": min_overall_score},
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

    if overall_score is None:
        print("Overall score is unavailable; failing run.")
        raise SystemExit(1)
    if float(overall_score) < min_overall_score:
        print(
            f"Overall score below threshold: overall={float(overall_score):.4f}, "
            f"required={float(min_overall_score):.4f}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
