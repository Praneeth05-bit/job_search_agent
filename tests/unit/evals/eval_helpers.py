"""Builders for eval tests: cases and agent runs with only the parts that matter."""

from typing import Any

from job_search_agent.agent.answer import AgentAnswer, RankedMatch
from job_search_agent.agent.graph import AgentRun
from job_search_agent.evals.dataset import EvalCase


def make_case(**expect: Any) -> EvalCase:
    """A normal case whose expectations are given as keyword arguments."""
    expect.setdefault("should_search", True)
    return EvalCase.model_validate(
        {
            "id": "normal-001",
            "category": "normal",
            "request": "python jobs",
            "profile": {"major": "Computer Science"},
            "reference_query": "python",
            "expect": expect,
            "why": "Labels for a scoring test.",
        }
    )


def make_run(
    *ids: str,
    tool_calls: int = 1,
    stop_reason: str = "completed",
    summary: str = "Here are some matches.",
    question: str | None = None,
    why: str = "Fits the request.",
    caveats: tuple[str, ...] = (),
    tokens: tuple[int, int] = (0, 0),
) -> AgentRun:
    """An agent run that recommended ``ids`` in order."""
    answer = AgentAnswer(
        summary=summary,
        matches=[
            RankedMatch(opportunity_id=i, rank=n, why=why, caveats=list(caveats))
            for n, i in enumerate(ids, 1)
        ],
        clarifying_question=question,
    )
    return AgentRun.model_validate(
        {
            "thread_id": "t-1",
            "status": "awaiting_approval" if ids else "finished",
            "pending_approval": None,
            "answer": answer,
            "final_answer": summary,
            "stop_reason": stop_reason,
            "approved": None,
            "saved_ids": [],
            "tool_results": [],
            "tool_calls_made": tool_calls,
            "model_calls_made": 3,
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "prompt_id": "search_agent@v1",
        }
    )
