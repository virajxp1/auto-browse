from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts.run_eval import (
    EvalTask,
    _fetch_eval_cases_from_dataset,
    _initial_task_record,
    _normalize_repeat_from_metadata,
    _record_from_eval_result,
    _run_single,
    _score_with_timeout,
    _set_hook_metadata,
    _zero_score_error_handler,
)


class _Hooks:
    def __init__(self, *, trial_index: int = 0) -> None:
        self.trial_index = trial_index
        self.metadata: dict[str, object] = {}
        self.tags: list[str] = []


class RunEvalHelpersTest(unittest.TestCase):
    def test_fetch_eval_cases_from_dataset_falls_back_to_fetched_data(self) -> None:
        dataset = SimpleNamespace(
            fetch=lambda: iter(()),
            fetched_data=[
                {
                    "id": "task-1",
                    "input": {
                        "id": "task-1",
                        "start_url": "https://example.com",
                        "target_prompt": "Inspect the page.",
                    },
                    "expected": {"reference_text": "Example"},
                    "metadata": {"task_id": "task-1"},
                    "tags": ["goal-based"],
                }
            ],
        )

        cases = _fetch_eval_cases_from_dataset(dataset, limit=0)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["input"]["id"], "task-1")

    def test_zero_score_error_handler_sets_zero_for_unhandled_scores(self) -> None:
        result = _zero_score_error_handler(None, None, ["score_success", "coherence"])
        self.assertEqual(result, {"score_success": 0.0, "coherence": 0.0})

    def test_set_hook_metadata_prepopulates_complex_task_fields(self) -> None:
        task = EvalTask.from_dict(
            {
                "id": "complex-task",
                "start_url": "https://example.com",
                "target_prompt": "Find the answer.",
                "min_trace_steps": 2,
            }
        )
        hooks = _Hooks(trial_index=1)
        record = _initial_task_record(task, repeat_index=hooks.trial_index, total_repeats=3)

        _set_hook_metadata(hooks, task, record)

        self.assertEqual(hooks.metadata["task_id"], "complex-task")
        self.assertEqual(hooks.metadata["trace_id"], "eval:complex-task:repeat:2")
        self.assertEqual(hooks.metadata["repeat"], 2)
        self.assertEqual(hooks.metadata["total_repeats"], 3)
        self.assertEqual(hooks.metadata["is_complex"], True)
        self.assertIn("complex", hooks.tags)

    def test_normalize_repeat_prefers_explicit_repeat(self) -> None:
        repeat = _normalize_repeat_from_metadata({"repeat": 3, "trial_index": 0})
        self.assertEqual(repeat, 3)

    def test_record_from_eval_result_uses_one_based_trial_index_on_failures(self) -> None:
        result = SimpleNamespace(
            output=None,
            input={"id": "task-1"},
            metadata={"task_id": "task-1", "trial_index": 1, "total_repeats": 4, "is_complex": True},
            error=RuntimeError("task crashed"),
            scores={},
        )

        record = _record_from_eval_result(result, total_repeats=4)

        self.assertEqual(record["task_id"], "task-1")
        self.assertEqual(record["repeat"], 2)
        self.assertEqual(record["total_repeats"], 4)
        self.assertEqual(record["is_complex"], True)
        self.assertEqual(record["success"], False)
        self.assertEqual(record["error"], "task_failed_without_output")

    def test_record_from_eval_result_preserves_trace_id_from_metadata(self) -> None:
        result = SimpleNamespace(
            output=None,
            input={"id": "task-2"},
            metadata={"task_id": "task-2", "trace_id": "eval:task-2:repeat:1"},
            error=None,
            scores={},
        )

        record = _record_from_eval_result(result, total_repeats=1)

        self.assertEqual(record["trace_id"], "eval:task-2:repeat:1")


class RunEvalExecutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_score_with_timeout_returns_zero_on_timeout(self) -> None:
        async def slow_score():
            await asyncio.sleep(0.05)

        result = await _score_with_timeout(name="coherence", coro=slow_score(), timeout_seconds=0)

        self.assertEqual(result["name"], "coherence")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["metadata"]["reason"], "scorer_timed_out")

    async def test_score_with_timeout_returns_zero_on_exception(self) -> None:
        async def boom():
            raise RuntimeError("network exploded")

        result = await _score_with_timeout(name="factuality", coro=boom(), timeout_seconds=1)

        self.assertEqual(result["name"], "factuality")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["metadata"]["reason"], "scorer_failed")
        self.assertIn("RuntimeError", result["metadata"]["detail"])

    async def test_run_single_reports_timeout_as_structured_failure(self) -> None:
        task = EvalTask.from_dict(
            {
                "id": "timeout-task",
                "start_url": "https://example.com",
                "target_prompt": "Do something slow.",
                "max_runtime_seconds": 1,
            }
        )

        with patch("agent.run.run_agent", new=AsyncMock(side_effect=asyncio.TimeoutError())):
            record = await _run_single(object(), task, repeat_index=0, total_repeats=1)

        self.assertEqual(record["task_id"], "timeout-task")
        self.assertFalse(record["success"])
        self.assertEqual(record["error"], "run_timed_out")
        self.assertEqual(record["trace_steps"], 0)
        self.assertEqual(record["trace_id"], "eval:timeout-task:repeat:1")

    async def test_run_single_forwards_trace_context_and_records_trace_id(self) -> None:
        task = EvalTask.from_dict(
            {
                "id": "trace-task",
                "start_url": "https://example.com",
                "target_prompt": "Inspect the page.",
            }
        )
        result = SimpleNamespace(
            error=None,
            trace=[],
            status="completed",
            goal_summary="Done",
            result_data=None,
            answer="ok",
            structured_data=None,
            evidence="ok",
            source_url="https://example.com",
            final_url="https://example.com",
            final_title="Example",
        )

        with (
            patch("scripts.run_eval.export_current_span_parent", return_value="parent-span"),
            patch("agent.run.run_agent", new=AsyncMock(return_value=result)) as mock_run_agent,
        ):
            record = await _run_single(object(), task, repeat_index=1, total_repeats=3)

        self.assertEqual(record["trace_id"], "eval:trace-task:repeat:2")
        self.assertTrue(record["success"])
        self.assertEqual(mock_run_agent.await_args.kwargs["trace_id"], "eval:trace-task:repeat:2")
        self.assertEqual(mock_run_agent.await_args.kwargs["trace_parent"], "parent-span")


if __name__ == "__main__":
    unittest.main()
