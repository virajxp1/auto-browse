from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from auto_browse_v2.decomposer import decompose_query
from auto_browse_v2.llm_client import LLMClient
from auto_browse_v2.models import OrchestratorState, SubTask, SubTaskResult
from auto_browse_v2.subtask_agent import run_subtask

logger = logging.getLogger(__name__)

_AGGREGATE_SYSTEM = """\
You are a research aggregator. Given answers from parallel research subtasks,
synthesize them into a single coherent response (3-8 sentences) for the user.
Address all parts of the original query. Use bullet points for multiple data points.
Note briefly if any subtask failed.
Return a JSON object with exactly one key: "answer" (string).
"""


async def node_decompose(state: OrchestratorState) -> dict[str, Any]:
    client = LLMClient()
    subtasks = await decompose_query(state["query"], client=client)
    return {"subtasks": subtasks}


def edge_fan_out(state: OrchestratorState) -> list[Send]:
    """Emit one Send per SubTask for parallel fan-out.

    Each Send targets run_subtask with the subtask serialized as a dict.
    LangGraph fires all Sends concurrently; results accumulate via operator.add.
    """
    return [
        Send("run_subtask", {"subtask": task.model_dump(), "query": state["query"]})
        for task in state["subtasks"]
    ]


async def node_run_subtask(state: dict[str, Any]) -> dict[str, Any]:
    client = LLMClient()
    subtask = SubTask.model_validate(state["subtask"])
    result: SubTaskResult = await run_subtask(subtask, client=client)
    return {"results": [result]}


async def node_aggregate(state: OrchestratorState) -> dict[str, Any]:
    client = LLMClient()

    results_summary = "\n\n".join(
        f"[{r.task_id}] status={r.status}\n"
        + (f"answer: {r.answer}" if r.answer else f"error: {r.error}")
        for r in state["results"]
    )

    raw = await client.json_completion(
        system=_AGGREGATE_SYSTEM,
        user=f"Original query: {state['query']}\n\nSubtask results:\n{results_summary}",
    )
    answer = str(raw.get("answer", "")).strip() or results_summary
    return {"final_answer": answer}


def build_graph():
    """Build and compile the orchestrator graph.

    Topology:
        START → decompose → [Send fan-out] → run_subtask (×N, parallel)
              → aggregate → END
    """
    builder = StateGraph(OrchestratorState)

    builder.add_node("decompose", node_decompose)
    builder.add_node("run_subtask", node_run_subtask)
    builder.add_node("aggregate", node_aggregate)

    builder.add_edge(START, "decompose")
    builder.add_conditional_edges("decompose", edge_fan_out, ["run_subtask"])
    builder.add_edge("run_subtask", "aggregate")
    builder.add_edge("aggregate", END)

    return builder.compile()


async def run_query(query: str) -> dict[str, Any]:
    """Run the full pipeline for a query string. Returns the final state dict."""
    graph = build_graph()
    initial: OrchestratorState = {
        "query": query,
        "subtasks": [],
        "results": [],
        "final_answer": "",
    }
    return await graph.ainvoke(initial)
