from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


class SubTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="Short slug, e.g. 'hyatt-ziva-price'")
    description: str = Field(description="Plain-English description of what to find")
    site_hint: str = Field(description="Domain/brand hint for routing, e.g. 'hyatt.com'")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Key/value params extracted from the query (dates, origins, etc.)",
    )


class SubTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: str  # "success" | "error"
    answer: str | None = None
    structured_data: dict[str, Any] | None = None
    source_url: str | None = None
    error: str | None = None


# OrchestratorState is a TypedDict (not Pydantic) because LangGraph's StateGraph
# requires a TypedDict or dataclass.
#
# results uses operator.add as its reducer: each run_subtask node returns
# {"results": [<one result>]}, and operator.add concatenates lists so all
# parallel branch results accumulate correctly.
class OrchestratorState(TypedDict):
    query: str
    subtasks: list[SubTask]
    results: Annotated[list[SubTaskResult], operator.add]
    final_answer: str
