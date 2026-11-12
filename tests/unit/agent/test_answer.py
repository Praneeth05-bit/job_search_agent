"""Tests for the structured answer: schema validation and grounding checks."""

import json
from datetime import date
from typing import Any

import pytest

from job_search_agent.agent.answer import (
    AgentAnswer,
    RankedMatch,
    grounding_problems,
    parse_answer,
    retrieved_ids,
)
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.llm.schemas import ToolCall
from job_search_agent.tools.search_tool import ToolResult, run_search_tool

TODAY = date(2026, 11, 4)


@pytest.fixture
def python_results(
    profile: StudentProfile, records: list[Opportunity]
) -> list[ToolResult]:
    """A real 'python' search: matches opp-003/006, excludes opp-001/002."""
    call = ToolCall(id="c1", name="search_opportunities", arguments={"text": "python"})
    return [run_search_tool(call, profile=profile, opportunities=records, today=TODAY)]


def _answer_json(*ids: str, **extra: Any) -> str:
    matches = [
        {"opportunity_id": opp_id, "rank": rank, "why": "Python role."}
        for rank, opp_id in enumerate(ids, start=1)
    ]
    return json.dumps(
        {"summary": "Here are your matches.", "matches": matches, **extra}
    )


def test_retrieved_ids_split_matches_and_exclusions(
    python_results: list[ToolResult],
) -> None:
    matched, excluded = retrieved_ids(python_results)

    assert matched == {"opp-003", "opp-006"}
    assert excluded == {"opp-001", "opp-002"}


def test_grounded_answer_is_accepted(python_results: list[ToolResult]) -> None:
    answer, problems = parse_answer(_answer_json("opp-003", "opp-006"), python_results)

    assert problems == []
    assert answer is not None
    assert [m.opportunity_id for m in answer.matches] == ["opp-003", "opp-006"]


def test_excluded_opportunity_is_rejected(python_results: list[ToolResult]) -> None:
    """Schema-valid JSON recommending an ineligible job is caught by grounding."""
    answer, problems = parse_answer(_answer_json("opp-001"), python_results)

    assert answer is None
    assert problems == [
        "opp-001 was excluded as ineligible and must not be recommended"
    ]


def test_invented_opportunity_is_rejected(python_results: list[ToolResult]) -> None:
    answer, problems = parse_answer(_answer_json("opp-099"), python_results)

    assert answer is None
    assert problems == ["opp-099 was not returned by the search tool"]


def test_invalid_json_shape_is_reported(python_results: list[ToolResult]) -> None:
    """Schema problems come back as readable messages the model can fix."""
    bad = json.dumps({"matches": [{"opportunity_id": "opp-003", "rank": 0}]})

    answer, problems = parse_answer(bad, python_results)

    assert answer is None
    assert any(p.startswith("summary") for p in problems)
    assert any("rank" in p for p in problems)


def test_empty_reply_is_reported(python_results: list[ToolResult]) -> None:
    assert parse_answer(None, python_results) == (
        None,
        ["The reply was empty; return JSON matching the schema"],
    )


def test_duplicates_and_rank_gaps_are_reported(
    python_results: list[ToolResult],
) -> None:
    answer = AgentAnswer(
        summary="x",
        matches=[
            RankedMatch(opportunity_id="opp-003", rank=1, why="a"),
            RankedMatch(opportunity_id="opp-003", rank=3, why="b"),
        ],
    )

    problems = grounding_problems(answer, python_results)

    assert "Duplicate opportunities: opp-003" in problems
    assert "Ranks must be 1, 2, 3, ... with no gaps or repeats" in problems


def test_clarifying_question_with_no_matches_is_valid(
    python_results: list[ToolResult],
) -> None:
    text = json.dumps(
        {"summary": "I need a bit more detail.", "clarifying_question": "Which field?"}
    )

    answer, problems = parse_answer(text, python_results)

    assert problems == []
    assert answer is not None and answer.clarifying_question == "Which field?"
