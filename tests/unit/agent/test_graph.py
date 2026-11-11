"""End-to-end agent runs through the real LangGraph graph, with a mock model.

The model is scripted; everything else is real: tool validation, search,
eligibility, routing, limits, and the conversation flow.
"""

import pytest
from agent_helpers import TODAY, answer, search_request

from job_search_agent.agent.graph import AgentRun, run_agent
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.llm.errors import ModelRateLimitError
from job_search_agent.llm.mock import MockModelClient
from job_search_agent.llm.schemas import ModelResponse, ToolCall


def _run(
    mock: MockModelClient,
    profile: StudentProfile,
    records: list[Opportunity],
    max_tool_calls: int = 5,
) -> AgentRun:
    return run_agent(
        "Find Python internships",
        profile,
        model=mock,
        opportunities=records,
        today=TODAY,
        max_tool_calls=max_tool_calls,
    )


def test_search_then_answer(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """The normal loop: model asks, our code searches, model answers."""
    mock = MockModelClient(
        [
            search_request("c1", "r1", text="python"),
            answer("You are eligible for opp-003 and opp-006."),
        ]
    )

    run = _run(mock, profile, records)

    assert run.stop_reason == "completed"
    assert run.final_answer == "You are eligible for opp-003 and opp-006."
    assert run.tool_calls_made == 1
    assert run.model_calls_made == 2
    assert run.prompt_id == "search_agent@v1"
    report = run.tool_results[0].report
    assert report is not None
    assert [m.search.opportunity_id for m in report.matches] == ["opp-003", "opp-006"]


def test_rules_and_results_reach_the_model_on_turn_two(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """Turn 2 continues turn 1, carries the result, and re-sends the rules."""
    mock = MockModelClient([search_request("c1", "r1", text="python"), answer("done")])

    _run(mock, profile, records)

    turn_two = mock.turns[1]
    assert turn_two.previous_response_id == "r1"
    assert turn_two.tool_outputs[0].call_id == "c1"
    assert "<tool_result" in turn_two.tool_outputs[0].result
    assert turn_two.system_instruction == mock.turns[0].system_instruction


def test_answer_without_tools(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """A clarifying question needs no search."""
    mock = MockModelClient([answer("Which field are you interested in?")])

    run = _run(mock, profile, records)

    assert run.tool_calls_made == 0
    assert run.final_answer == "Which field are you interested in?"


def test_self_correction_after_invalid_arguments(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """The error goes back to the model, which fixes its call."""
    mock = MockModelClient(
        [
            search_request("c1", "r1", text="python", limit=500),
            search_request("c2", "r2", text="python", limit=10),
            answer("Found opp-003 and opp-006."),
        ]
    )

    run = _run(mock, profile, records)

    assert run.tool_calls_made == 2
    assert [result.ok for result in run.tool_results] == [False, True]
    assert mock.turns[1].tool_outputs[0].is_error


def test_unknown_tool_is_refused(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    mock = MockModelClient(
        [
            ModelResponse(
                response_id="r1",
                tool_calls=[ToolCall(id="c1", name="delete_all_jobs", arguments={})],
            ),
            answer("Sorry, I can only search."),
        ]
    )

    run = _run(mock, profile, records)

    assert run.tool_results[0].ok is False
    assert run.tool_results[0].report is None


def test_tool_limit_stops_a_looping_model(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """A model that never stops searching is cut off by code, not by the prompt."""
    script = [search_request(f"c{i}", f"r{i}", text="python") for i in range(1, 10)]
    mock = MockModelClient(script)

    run = _run(mock, profile, records, max_tool_calls=3)

    assert run.stop_reason == "tool_limit"
    assert run.tool_calls_made == 3
    assert (
        run.model_calls_made == 4
    )  # 3 searches answered, then the 4th request is cut off


def test_model_failure_propagates(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """Model errors surface to the caller (the API decides how to show them)."""
    mock = MockModelClient([ModelRateLimitError("429")])

    with pytest.raises(ModelRateLimitError):
        _run(mock, profile, records)
