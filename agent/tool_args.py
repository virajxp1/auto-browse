"""Pydantic argument schemas for the agent's typed tool calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _StepArgs(_StrictArgs):
    step_summary: str
    next_step: str


class _TargetStepArgs(_StepArgs):
    selector: str | None = None
    interactable_ref: str | None = None


class TypeAndSubmitArgs(_TargetStepArgs):
    text: str


class AnalyzePageArgs(_StepArgs):
    question: str


class VerifyGoalArgs(_StepArgs):
    criteria: str


class FillArgs(_TargetStepArgs):
    text: str


class SubmitArgs(_TargetStepArgs):
    pass


class SelectOptionArgs(_TargetStepArgs):
    value: str


class CheckArgs(_TargetStepArgs):
    pass


class WaitForArgs(_TargetStepArgs):
    state: str = "visible"
    timeout_ms: int = Field(default=5000, ge=0, le=30000)


class ClickArgs(_TargetStepArgs):
    pass


class NavigateArgs(_StepArgs):
    url: str | None = None
    interactable_ref: str | None = None
    selector: str | None = None


class ExtractAnswerArgs(_StepArgs):
    answer: str | None = None
    structured_data: dict[str, str | None] | None = None
    evidence: str
    confidence: float | int | str | None = None


class CompleteGoalArgs(_StepArgs):
    status: str = "completed"
    goal_summary: str
    result_data: dict[str, Any] | None = None
    evidence: str
    confidence: float | int | str | None = None


class FailArgs(_StepArgs):
    reason: str
