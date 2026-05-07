#!/usr/bin/env python3
"""Braintrust eval runner for auto-browse v2 (NavigationLoop + ScrapeGraphAI).

Drop-in replacement for run_eval.py: same dataset, same scorers, same Braintrust
project — but calls NavigationLoop instead of run_agent so results are directly
comparable in the Braintrust UI.
"""
from __future__ import annotations

import argparse
import asyncio
import configparser
import json
import math
import os
import statistics
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

DEFAULT_EVAL_CONFIG_PATH = "config/config.ini"
DEFAULT_DATASET_NAME = "auto-browse-eval-cases"
DEFAULT_MIN_OVERALL_SCORE = 0.75
DEFAULT_AUTOEVALS_MODEL = "openai/gpt-4o-mini"
DEFAULT_SCORER_TIMEOUT_SECONDS = 20

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

_ANSWER_FORMAT_SYSTEM = (
    "Summarize the extracted data as a concise, factual answer to the question. "
    "Return JSON with exactly one key: \"answer\" (string)."
)


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    start_url: str
    target_prompt: str
    goal_type: str | None = None
    task_data: dict[str, str] | None = None
    sensitive_data: dict[str, str] | None = None
    max_steps: int = 10
    max_actions_per_step: int = 1
    max_runtime_seconds: int = 90
    extraction_schema: dict[str, str] | None = None
    extraction_selector: str | None = None
    expected_contains: list[str] | None = None
    min_trace_steps: int | None = None
    required_actions: list[str] | None = None

    def is_complex(self) -> bool:
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

        goal_type_raw = payload.get("goal_type")
        goal_type = str(goal_type_raw).strip() if goal_type_raw is not None else None

        max_steps = int(payload.get("max_steps", 10))
        max_actions_per_step = int(payload.get("max_actions_per_step", 1))
        max_runtime_seconds = int(payload.get("max_runtime_seconds", 90))

        def _str_map(field: str) -> dict[str, str] | None:
            raw = payload.get(field)
            if raw is None:
                return None
            if not isinstance(raw, dict):
                raise ValueError(f"Task '{task_id}' {field} must be an object")
            normalized = {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip()}
            return normalized or None

        extraction_schema_raw = payload.get("extraction_schema")
        extraction_schema: dict[str, str] | None = None
        if extraction_schema_raw is not None and isinstance(extraction_schema_raw, dict):
            extraction_schema = {str(k).strip(): str(v).strip() for k, v in extraction_schema_raw.items()}

        extraction_selector_raw = payload.get("extraction_selector")
        extraction_selector = str(extraction_selector_raw).strip() if extraction_selector_raw else None

        expected_contains_raw = payload.get("expected_contains")
        expected_contains: list[str] | None = None
        if isinstance(expected_contains_raw, list):
            expected_contains = [str(i).strip() for i in expected_contains_raw if str(i).strip()]

        min_trace_steps_raw = payload.get("min_trace_steps")
        min_trace_steps = int(min_trace_steps_raw) if min_trace_steps_raw is not None else None

        required_actions_raw = payload.get("required_actions")
        required_actions: list[str] | None = None
        if isinstance(required_actions_raw, list):
            required_actions = [str(a).strip() for a in required_actions_raw if str(a).strip()] or None

        return cls(
            task_id=task_id,
            start_url=start_url,
            target_prompt=target_prompt,
            goal_type=goal_type,
            task_data=_str_map("task_data"),
            sensitive_data=_str_map("sensitive_data"),
            max_steps=max_steps,
            max_actions_per_step=max_actions_per_step,
            max_runtime_seconds=max_runtime_seconds,
            extraction_schema=extraction_schema,
            extraction_selector=extraction_selector,
            expected_contains=expected_contains,
            min_trace_steps=min_trace_steps,
            required_actions=required_actions,
        )


# ── Utilities (identical to run_eval.py) ────────────────────────────────────

def _load_env_file_if_present(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def _read_braintrust_settings(path: Path) -> dict[str, str | None]:
    if not path.is_file():
        raise ValueError(f"Missing config file: {path}")
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
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
        record.get("status") or "",
        record.get("goal_summary") or "",
        json.dumps(record.get("result_data") or {}, sort_keys=True),
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
    return str(case.get("id") or "").strip() or "unknown_task"


def _normalize_dataset_case(raw: dict[str, Any]) -> dict[str, Any] | None:
    task_payload = raw.get("input")
    if not isinstance(task_payload, dict):
        return None
    task_id = str(task_payload.get("id") or raw.get("id") or "").strip()
    if task_id and not str(task_payload.get("id") or "").strip():
        task_payload = dict(task_payload)
        task_payload["id"] = task_id
    metadata = dict(raw.get("metadata") or {})
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
        "tags": [str(t) for t in tags if str(t).strip()],
    }


def _fetch_eval_cases_from_dataset(dataset: Any, *, limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    raw_records = list(dataset.fetch())
    if not raw_records:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fetched_data = getattr(dataset, "fetched_data", None)
        if isinstance(fetched_data, list):
            raw_records = fetched_data
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        normalized = _normalize_dataset_case(raw)
        if normalized:
            cases.append(normalized)
    cases.sort(key=_task_id_from_eval_case)
    if limit > 0:
        cases = cases[:limit]
    return cases


def _reference_text_from_expected(input_payload: Any, expected_payload: Any) -> str | None:
    if isinstance(expected_payload, str) and expected_payload.strip():
        return expected_payload.strip()
    if isinstance(expected_payload, dict):
        ref = str(expected_payload.get("reference_text") or "").strip()
        if ref:
            return ref
        ec = expected_payload.get("expected_contains")
        if isinstance(ec, list):
            joined = "\n".join(str(i).strip() for i in ec if str(i).strip()).strip()
            if joined:
                return joined
    if isinstance(input_payload, dict):
        ec = input_payload.get("expected_contains")
        if isinstance(ec, list):
            joined = "\n".join(str(i).strip() for i in ec if str(i).strip()).strip()
            if joined:
                return joined
    return None


def _summarize_error(error: Exception, *, limit: int = 160) -> str:
    message = str(error).strip() or error.__class__.__name__
    detail = f"{error.__class__.__name__}: {message}"
    return f"{detail[:limit - 3]}..." if len(detail) > limit else detail


async def _score_with_timeout(*, name: str, coro: Any, timeout_seconds: int = DEFAULT_SCORER_TIMEOUT_SECONDS) -> dict[str, Any]:
    try:
        score = await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return {"name": name, "score": 0.0, "metadata": {"reason": "scorer_timed_out"}}
    except Exception as exc:
        return {"name": name, "score": 0.0, "metadata": {"reason": "scorer_failed", "detail": _summarize_error(exc)}}
    return {"name": name, "score": score.score, "metadata": score.metadata}


def _resolve_autoevals_settings(*, configured_model: str) -> dict[str, str | None]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or os.getenv("OPEN_ROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY for eval scorers.")
    return {"api_key": api_key, "base_url": "https://openrouter.ai/api/v1", "model": configured_model or DEFAULT_AUTOEVALS_MODEL}


def _compute_overall_score(records: list[dict[str, Any]]) -> dict[str, Any]:
    tracked = ["score_success", "coherence", "factuality"]
    per_metric: dict[str, float | None] = {}
    for metric in tracked:
        values = [float(r["scores"][metric]) for r in records if isinstance(r.get("scores"), dict) and r["scores"].get(metric) is not None]
        per_metric[metric] = round(statistics.mean(values), 4) if values else None
    available = [v for v in per_metric.values() if v is not None]
    return {"overall_score": round(statistics.mean(available), 4) if available else None, "overall_components": per_metric}


def _first_expectation_failure(record: dict[str, Any], task: EvalTask) -> str | None:
    if not _matches_expectations(record, task.expected_contains):
        return "expected_contains_mismatch"
    if task.min_trace_steps is not None and int(record["trace_steps"]) < task.min_trace_steps:
        return f"min_trace_steps_not_met:expected>={task.min_trace_steps},actual={int(record['trace_steps'])}"
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
        "trace_id": record.get("trace_id"),
        "repeat": int(record.get("repeat") or 1),
        "total_repeats": int(record.get("total_repeats") or total_repeats),
        "is_complex": bool(record.get("is_complex")),
        "success": bool(record.get("success")),
        "error": record.get("error"),
        "duration_s": round(float(record.get("duration_s") or 0.0), 4),
        "trace_steps": int(record.get("trace_steps") or 0),
        "actions": [str(a) for a in (record.get("actions") or [])],
        "status": record.get("status"),
        "goal_summary": record.get("goal_summary"),
        "result_data": record.get("result_data"),
        "answer": record.get("answer"),
        "structured_data": record.get("structured_data"),
        "evidence": record.get("evidence"),
        "source_url": record.get("source_url"),
        "final_url": record.get("final_url"),
        "final_title": record.get("final_title"),
    }
    if "scores" in record and isinstance(record.get("scores"), dict):
        normalized["scores"] = {str(k): v for k, v in record["scores"].items()}
    return normalized


def _set_hook_metadata(hooks: Any, task: EvalTask, record: dict[str, Any]) -> None:
    metadata = getattr(hooks, "metadata", None)
    if isinstance(metadata, dict):
        metadata.update({
            "task_id": task.task_id,
            "trace_id": record.get("trace_id"),
            "repeat": int(record.get("repeat") or 1),
            "total_repeats": int(record.get("total_repeats") or 1),
            "is_complex": bool(record.get("is_complex")),
            "success": bool(record.get("success")),
            "error": record.get("error"),
            "duration_s": float(record.get("duration_s") or 0.0),
            "trace_steps": int(record.get("trace_steps") or 0),
            "actions": list(record.get("actions") or []),
            "agent_version": "v2",
        })
    tags = getattr(hooks, "tags", None)
    if isinstance(tags, list):
        if task.is_complex() and "complex" not in tags:
            tags.append("complex")
        if "v2" not in tags:
            tags.append("v2")


def _initial_task_record(task: EvalTask, *, repeat_index: int, total_repeats: int) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "trace_id": f"eval:{task.task_id}:repeat:{repeat_index + 1}",
        "repeat": repeat_index + 1,
        "total_repeats": total_repeats,
        "is_complex": task.is_complex(),
        "success": False, "error": None, "duration_s": 0.0,
        "trace_steps": 0, "actions": [], "status": None,
        "goal_summary": None, "result_data": None, "answer": None,
        "structured_data": None, "evidence": None,
        "source_url": None, "final_url": None, "final_title": None,
    }


def _eval_trace_id(task_id: str, repeat_index: int) -> str:
    return f"eval:{task_id}:repeat:{repeat_index + 1}"


def _error_record(task: EvalTask, trace_id: str, repeat_index: int, total_repeats: int, duration_s: float, error: str) -> dict[str, Any]:
    return _normalize_record({
        "task_id": task.task_id, "trace_id": trace_id,
        "repeat": repeat_index + 1, "total_repeats": total_repeats,
        "is_complex": task.is_complex(), "success": False, "error": error,
        "duration_s": round(duration_s, 4), "trace_steps": 0, "actions": [],
        "status": None, "goal_summary": None, "result_data": None,
        "answer": None, "structured_data": None, "evidence": None,
        "source_url": None, "final_url": None, "final_title": None,
    }, total_repeats=total_repeats)


# ── v2-specific: run a single eval task via NavigationLoop ───────────────────

async def _run_single_v2(
    task: EvalTask,
    repeat_index: int,
    total_repeats: int,
    *,
    timeout_override: int = 0,
) -> dict[str, Any]:
    from auto_browse_v2.llm_client import LLMClient
    from auto_browse_v2.navigator import NavigationLoop

    started = time.perf_counter()
    trace_id = _eval_trace_id(task.task_id, repeat_index)

    llm = LLMClient()
    nav = NavigationLoop(client=llm, max_steps=task.max_steps)
    runtime = timeout_override if timeout_override > 0 else task.max_runtime_seconds

    try:
        nav_result = await asyncio.wait_for(
            nav.run(goal=task.target_prompt, start_url=task.start_url),
            timeout=runtime,
        )
    except asyncio.TimeoutError:
        duration_s = time.perf_counter() - started
        print(f"[{repeat_index + 1}/{total_repeats}] {task.task_id}: TIMEOUT after {duration_s:.1f}s")
        return _error_record(task, trace_id, repeat_index, total_repeats, duration_s, "run_timed_out")
    except Exception as exc:
        duration_s = time.perf_counter() - started
        print(f"[{repeat_index + 1}/{total_repeats}] {task.task_id}: EXCEPTION {exc}")
        return _error_record(task, trace_id, repeat_index, total_repeats, duration_s, f"task_exception:{type(exc).__name__}")

    duration_s = time.perf_counter() - started

    # Format a human-readable answer from extracted data
    answer = ""
    if nav_result.data:
        try:
            raw = await llm.json_completion(
                system=_ANSWER_FORMAT_SYSTEM,
                user=f"Question: {task.target_prompt}\nExtracted data: {nav_result.data}",
            )
            answer = str(raw.get("answer", "")).strip()
        except Exception:
            answer = str(nav_result.data)

    record = {
        "task_id": task.task_id,
        "trace_id": trace_id,
        "repeat": repeat_index + 1,
        "total_repeats": total_repeats,
        "is_complex": task.is_complex(),
        "success": nav_result.success,
        "error": nav_result.error,
        "duration_s": round(duration_s, 4),
        "trace_steps": nav_result.steps_taken,
        "actions": nav_result.history,
        "status": "success" if nav_result.success else "partial",
        "goal_summary": task.target_prompt[:200],
        "result_data": nav_result.data,
        "answer": answer,
        "structured_data": nav_result.data,
        "evidence": None,
        "source_url": nav_result.url,
        "final_url": nav_result.url,
        "final_title": None,
    }

    # If agent gave up / hit max steps but SGAI still extracted the right data, count it as success.
    if not record["success"] and record.get("structured_data") and task.expected_contains:
        if _matches_expectations(record, task.expected_contains):
            record["success"] = True
            record["error"] = None

    expectation_failure = _first_expectation_failure(record, task) if record["success"] else None
    if expectation_failure is not None:
        record["success"] = False
        record["error"] = expectation_failure

    status = "PASS" if record["success"] else "FAIL"
    print(f"[{repeat_index + 1}/{total_repeats}] {task.task_id}: {status} steps={record['trace_steps']} duration={record['duration_s']}s")
    return _normalize_record(record, total_repeats=total_repeats)


def _normalize_repeat_from_metadata(metadata: dict[str, Any]) -> int:
    explicit_repeat = metadata.get("repeat")
    if explicit_repeat not in (None, ""):
        try:
            parsed = int(explicit_repeat)
            if parsed >= 1:
                return parsed
        except Exception:
            pass
    try:
        trial_index = int(metadata.get("trial_index") or 0)
    except Exception:
        trial_index = 0
    return max(1, trial_index + 1)


def _zero_score_error_handler(_span: Any, _datum: Any, unhandled_scores: list[str]) -> dict[str, float]:
    return {str(s): 0.0 for s in unhandled_scores}


def _record_from_eval_result(result: Any, *, total_repeats: int) -> dict[str, Any]:
    output = getattr(result, "output", None)
    if isinstance(output, dict):
        record = dict(output)
    else:
        input_payload = getattr(result, "input", {}) or {}
        metadata = getattr(result, "metadata", {}) or {}
        repeat = _normalize_repeat_from_metadata(metadata)
        task_id = str(metadata.get("task_id") or input_payload.get("id") or "unknown_task")
        record = {
            "task_id": task_id,
            "trace_id": metadata.get("trace_id"), "repeat": repeat,
            "total_repeats": int(metadata.get("total_repeats") or total_repeats),
            "is_complex": bool(metadata.get("is_complex")), "success": False,
            "error": "task_failed_without_output", "duration_s": 0.0,
            "trace_steps": 0, "actions": [], "status": None,
            "goal_summary": None, "result_data": None, "answer": None,
            "structured_data": None, "evidence": None,
            "source_url": None, "final_url": None, "final_title": None,
        }
    err = getattr(result, "error", None)
    if err is not None:
        record["success"] = False
        if not record.get("error"):
            record["error"] = str(err)
    scores = getattr(result, "scores", None)
    if isinstance(scores, dict):
        record["scores"] = {str(k): v for k, v in scores.items()}
    return _normalize_record(record, total_repeats=total_repeats)


def _records_from_eval_results(results: list[Any], *, total_repeats: int) -> list[dict[str, Any]]:
    records = [_record_from_eval_result(r, total_repeats=total_repeats) for r in results]
    records.sort(key=lambda r: (int(r["repeat"]), str(r["task_id"])))
    return records


# ── Braintrust eval runner ───────────────────────────────────────────────────

async def _run_eval(
    eval_cases: list[dict[str, Any]],
    repeats: int,
    *,
    braintrust_project: str,
    braintrust_project_id: str | None,
    max_concurrency: int,
    autoeval_settings: dict[str, str | None],
    timeout_override: int = 0,
) -> list[dict[str, Any]]:
    try:
        from braintrust import EvalAsync
    except Exception as exc:
        raise RuntimeError("Braintrust SDK is required. pip install braintrust") from exc
    try:
        from autoevals import Factuality, LLMClassifier
    except Exception as exc:
        raise RuntimeError("autoevals is required. pip install autoevals") from exc

    autoeval_api_key = str(autoeval_settings.get("api_key") or "").strip()
    if not autoeval_api_key:
        raise ValueError("Missing autoeval API key")
    autoeval_model = str(autoeval_settings.get("model") or DEFAULT_AUTOEVALS_MODEL)
    autoeval_base_url = autoeval_settings.get("base_url")

    coherence_scorer = LLMClassifier(
        name="coherence",
        prompt_template=COHERENCE_PROMPT,
        choice_scores={"coherent": 1.0, "incoherent": 0.0},
        model=autoeval_model, use_cot=False,
        api_key=autoeval_api_key,
        base_url=str(autoeval_base_url) if autoeval_base_url else None,
    )
    factuality_scorer = Factuality(
        model=autoeval_model, use_cot=False,
        api_key=autoeval_api_key,
        base_url=str(autoeval_base_url) if autoeval_base_url else None,
    )

    async def run_eval_task(task_payload: dict[str, Any], hooks: Any) -> dict[str, Any]:
        task = EvalTask.from_dict(task_payload)
        trial_index = int(getattr(hooks, "trial_index", 0))
        _set_hook_metadata(hooks, task, _initial_task_record(task, repeat_index=trial_index, total_repeats=repeats))
        record = await _run_single_v2(task, trial_index, repeats, timeout_override=timeout_override)  # noqa: B023
        _set_hook_metadata(hooks, task, record)
        return record

    def score_success(_input: Any, output: Any, expected: Any = None, **kwargs: Any) -> float:
        return 1.0 if isinstance(output, dict) and bool(output.get("success")) else 0.0

    async def score_coherence(input: Any, output: Any, expected: Any = None, **kwargs: Any) -> dict[str, Any]:
        output_text = _result_text(output) if isinstance(output, dict) else str(output or "")
        if not output_text.strip():
            return {"name": "coherence", "score": 0.0, "metadata": {"reason": "empty_output"}}
        input_text = str(input.get("target_prompt") or input.get("id") or "") if isinstance(input, dict) else ""
        return await _score_with_timeout(
            name="coherence",
            coro=coherence_scorer.eval_async(output=output_text, input=input_text),
        )

    async def score_factuality(input: Any, output: Any, expected: Any = None, **kwargs: Any) -> dict[str, Any]:
        output_text = _result_text(output) if isinstance(output, dict) else str(output or "")
        if not output_text.strip():
            return {"name": "factuality", "score": 0.0, "metadata": {"reason": "empty_output"}}
        reference_text = _reference_text_from_expected(input, expected)
        if not reference_text:
            return {"name": "factuality", "score": None, "metadata": {"reason": "missing_reference"}}
        return await _score_with_timeout(
            name="factuality",
            coro=factuality_scorer.eval_async(output=output_text, expected=reference_text),
        )

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
        metadata={"repeats": repeats, "data_source": "dataset", "autoevals_model": autoeval_model, "agent_version": "v2"},
    )

    return _records_from_eval_results(list(getattr(result, "results", [])), total_repeats=repeats)


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    success_records = [r for r in records if r["success"]]
    success_rate = len(success_records) / len(records) if records else 0.0
    durations = [float(r["duration_s"]) for r in records]
    successful_steps = [int(r["trace_steps"]) for r in success_records]
    complex_records = [r for r in records if bool(r.get("is_complex"))]
    complex_success = [r for r in complex_records if r["success"]]

    per_task: dict[str, dict[str, Any]] = {}
    for r in records:
        task_id = str(r["task_id"])
        b = per_task.setdefault(task_id, {"runs": 0, "successes": 0, "durations": [], "steps": [], "actions": []})
        b["runs"] += 1
        b["durations"].append(float(r["duration_s"]))
        b["actions"].extend(str(a) for a in r.get("actions", []))
        if r["success"]:
            b["successes"] += 1
            b["steps"].append(int(r["trace_steps"]))

    per_task_summary = {
        task_id: {
            "runs": b["runs"],
            "successes": b["successes"],
            "success_rate": round(b["successes"] / b["runs"], 4) if b["runs"] else 0.0,
            "median_duration_s": round(statistics.median(b["durations"]), 4) if b["durations"] else None,
            "median_steps_success": statistics.median(b["steps"]) if b["steps"] else None,
        }
        for task_id, b in sorted(per_task.items())
    }

    return {
        "total_runs": len(records),
        "successful_runs": len(success_records),
        "success_rate": round(success_rate, 4),
        "complex_total_runs": len(complex_records),
        "complex_successful_runs": len(complex_success),
        "complex_success_rate": round(len(complex_success) / len(complex_records), 4) if complex_records else 0.0,
        "median_duration_s": round(statistics.median(durations), 4) if durations else None,
        "p95_duration_s": round(_percentile(durations, 95) or 0.0, 4) if durations else None,
        "median_steps_success": statistics.median(successful_steps) if successful_steps else None,
        "per_task": per_task_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Braintrust evals for auto-browse v2.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N tasks")
    parser.add_argument("--output", default=".context/eval_report_v2.json")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=0, help="Override per-task timeout in seconds (default: use dataset value)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _load_env_file_if_present()

    eval_config_path = Path(DEFAULT_EVAL_CONFIG_PATH)
    braintrust_settings = _read_braintrust_settings(eval_config_path)
    braintrust_project = str(braintrust_settings["project_name"])
    braintrust_project_id = str(braintrust_settings["project_id"])
    dataset_name = str(braintrust_settings.get("dataset_name") or DEFAULT_DATASET_NAME)
    min_overall_score = float(braintrust_settings.get("min_overall_score") or DEFAULT_MIN_OVERALL_SCORE)
    autoeval_model = str(braintrust_settings.get("autoevals_model") or DEFAULT_AUTOEVALS_MODEL)

    output_path = Path(args.output)

    from braintrust import init_dataset
    dataset = init_dataset(
        project=braintrust_project,
        project_id=braintrust_project_id,
        name=dataset_name,
        use_output=False,
    )
    eval_cases = _fetch_eval_cases_from_dataset(dataset, limit=args.limit if args.limit > 0 else 0)
    if not eval_cases:
        raise ValueError(f"No eval records found in dataset '{dataset_name}'.")

    autoeval_settings = _resolve_autoevals_settings(configured_model=autoeval_model)

    print(f"Running v2 eval: {len(eval_cases)} tasks, project={braintrust_project}")

    records = asyncio.run(
        _run_eval(
            eval_cases, args.repeats,
            braintrust_project=braintrust_project,
            braintrust_project_id=braintrust_project_id,
            max_concurrency=args.max_concurrency,
            autoeval_settings=autoeval_settings,
            timeout_override=args.timeout,
        )
    )
    summary = _build_summary(records)
    summary.update(_compute_overall_score(records))
    overall_score = summary.get("overall_score")

    report = {
        "generated_at_unix": time.time(),
        "source": "dataset",
        "dataset_name": dataset_name,
        "agent_version": "v2",
        "repeats": args.repeats,
        "task_count": len(eval_cases),
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

    print("\nEval summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved report: {output_path}")

    if overall_score is None:
        print("Overall score unavailable; failing.")
        raise SystemExit(1)
    if float(overall_score) < min_overall_score:
        print(f"Score below threshold: {float(overall_score):.4f} < {min_overall_score:.4f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
