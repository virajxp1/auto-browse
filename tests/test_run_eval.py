from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.run_eval import (
    EvalTask,
    _initial_task_record,
    _normalize_repeat_from_metadata,
    _record_from_eval_result,
    _set_hook_metadata,
    _zero_score_error_handler,
)


class _Hooks:
    def __init__(self, *, trial_index: int = 0) -> None:
        self.trial_index = trial_index
        self.metadata: dict[str, object] = {}
        self.tags: list[str] = []


class RunEvalHelpersTest(unittest.TestCase):
    def test_zero_score_error_handler_sets_zero_for_unhandled_scores(self) -> None:
        result = _zero_score_error_handler(None, None, ["score_success", "coherence"])
        self.assertEqual(result, {"score_success": 0.0, "coherence": 0.0})

    def test_set_hook_metadata_prepopulates_complex_task_fields(self) -> None:
        task = EvalTask.from_dict(
            {
                "id": "complex-task",
                "start_url": "https://example.com",
                "target_prompt": "Find the answer.",
                "max_actions_per_step": 2,
            }
        )
        hooks = _Hooks(trial_index=1)
        record = _initial_task_record(task, repeat_index=hooks.trial_index, total_repeats=3)

        _set_hook_metadata(hooks, task, record)

        self.assertEqual(hooks.metadata["task_id"], "complex-task")
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


if __name__ == "__main__":
    unittest.main()
