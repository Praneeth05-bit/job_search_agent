"""Unit tests for the agent's nodes and router, without running a graph."""

from datetime import date

from agent_helpers import TODAY, answer, search_request

from job_search_agent.agent.nodes import (
    LIMIT_MESSAGE,
    AgentDependencies,
    AgentNodes,
)
from job_search_agent.agent.state import AgentState
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.llm.mock import MockModelClient
from job_search_agent.llm.schemas import ModelResponse, ToolCall, ToolOutput
from job_search_agent.prompts.loader import load_prompt


def _nodes(
    records: list[Opportunity],
    mock: MockModelClient | None = None,
    max_tool_calls: int = 5,
    today: date = TODAY,
) -> AgentNodes:
    deps = AgentDependencies(
        model=mock or MockModelClient([]),
        opportunities=records,
        today=today,
        prompt=load_prompt("search_agent", 1),
        max_tool_calls=max_tool_calls,
    )
    return AgentNodes(deps)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def test_router_runs_tools_when_requested_and_under_limit(
    records: list[Opportunity],
) -> None:
    state: AgentState = {
        "last_response": search_request("c1", "r1", text="python"),
        "tool_calls_made": 0,
    }
    assert _nodes(records).route_after_model(state) == "run_tools"


def test_router_finishes_on_text_answer(records: list[Opportunity]) -> None:
    state: AgentState = {"last_response": answer("done"), "tool_calls_made": 1}
    assert _nodes(records).route_after_model(state) == "finish"


def test_router_finishes_when_limit_reached(records: list[Opportunity]) -> None:
    state: AgentState = {
        "last_response": search_request("c6", "r6", text="python"),
        "tool_calls_made": 5,
    }
    assert _nodes(records).route_after_model(state) == "finish"


# ---------------------------------------------------------------------------
# call_model
# ---------------------------------------------------------------------------


def test_first_turn_sends_request_tools_and_rules(records: list[Opportunity]) -> None:
    mock = MockModelClient([answer("hi")])
    state: AgentState = {"request": "Find Python internships", "last_response": None}

    update = _nodes(records, mock).call_model(state)

    turn = mock.turns[0]
    assert "Find Python internships" in turn.prompt
    assert turn.system_instruction == load_prompt("search_agent", 1).text
    assert [tool.name for tool in turn.tools] == ["search_opportunities"]
    assert turn.previous_response_id is None
    assert update["model_calls_made"] == 1


def test_profile_is_not_sent_to_the_model(
    records: list[Opportunity], profile: StudentProfile
) -> None:
    """Personal details stay in our state; only the tool uses them."""
    mock = MockModelClient([answer("hi")])
    state: AgentState = {
        "request": "Find jobs",
        "profile": profile,
        "last_response": None,
    }

    _nodes(records, mock).call_model(state)

    assert "3.4" not in mock.turns[0].prompt
    assert "authorized_to_work_in_us" not in mock.turns[0].prompt


def test_follow_up_turn_continues_and_resends_rules(records: list[Opportunity]) -> None:
    mock = MockModelClient([answer("done")])
    output = ToolOutput(call_id="c1", name="search_opportunities", result="{}")
    state: AgentState = {
        "last_response": search_request("c1", "r1", text="python"),
        "pending_outputs": [output],
        "model_calls_made": 1,
    }

    update = _nodes(records, mock).call_model(state)

    turn = mock.turns[0]
    assert turn.previous_response_id == "r1"
    assert turn.tool_outputs == [output]
    assert turn.system_instruction == load_prompt("search_agent", 1).text
    assert update["pending_outputs"] == []
    assert update["model_calls_made"] == 2


# ---------------------------------------------------------------------------
# run_tools
# ---------------------------------------------------------------------------


def test_run_tools_runs_real_search_and_counts_the_call(
    records: list[Opportunity], profile: StudentProfile
) -> None:
    state: AgentState = {
        "last_response": search_request("c1", "r1", text="python"),
        "profile": profile,
        "tool_calls_made": 0,
    }

    update = _nodes(records).run_tools(state)

    result = update["tool_results"][0]
    assert result.ok
    assert result.report is not None
    assert [m.search.opportunity_id for m in result.report.matches] == [
        "opp-003",
        "opp-006",
    ]
    assert update["tool_calls_made"] == 1
    assert update["pending_outputs"][0].call_id == "c1"


def test_failed_tool_call_still_counts(
    records: list[Opportunity], profile: StudentProfile
) -> None:
    """Bad arguments use up the budget, so mistakes cannot loop forever."""
    state: AgentState = {
        "last_response": search_request("c1", "r1", text="python", limit=500),
        "profile": profile,
        "tool_calls_made": 0,
    }

    update = _nodes(records).run_tools(state)

    assert update["tool_calls_made"] == 1
    assert update["pending_outputs"][0].is_error


def test_calls_beyond_limit_are_not_run(
    records: list[Opportunity], profile: StudentProfile
) -> None:
    """With one call left, the second of two requested calls is refused."""
    response = ModelResponse(
        response_id="r1",
        tool_calls=[
            ToolCall(id="c1", name="search_opportunities", arguments={"text": "a"}),
            ToolCall(id="c2", name="search_opportunities", arguments={"text": "b"}),
        ],
    )
    state: AgentState = {
        "last_response": response,
        "profile": profile,
        "tool_calls_made": 4,
    }

    update = _nodes(records, max_tool_calls=5).run_tools(state)

    assert update["tool_calls_made"] == 5
    assert len(update["tool_results"]) == 1  # only c1 ran
    refused = update["pending_outputs"][1]
    assert (refused.call_id, refused.is_error, refused.result) == (
        "c2",
        True,
        LIMIT_MESSAGE,
    )


def test_missing_call_id_gets_a_fallback(
    records: list[Opportunity], profile: StudentProfile
) -> None:
    """Every output needs a call id, even if the model omitted one."""
    response = ModelResponse(
        tool_calls=[ToolCall(name="search_opportunities", arguments={"text": "x"})]
    )
    state: AgentState = {"last_response": response, "profile": profile}

    update = _nodes(records).run_tools(state)

    assert update["pending_outputs"][0].call_id == "call-1"


# ---------------------------------------------------------------------------
# finish
# ---------------------------------------------------------------------------


def test_finish_records_completed_answer(records: list[Opportunity]) -> None:
    update = _nodes(records).finish(
        {"last_response": answer("You qualify for opp-006.")}
    )

    assert update == {
        "stop_reason": "completed",
        "final_answer": "You qualify for opp-006.",
    }


def test_finish_reports_tool_limit(records: list[Opportunity]) -> None:
    update = _nodes(records).finish(
        {"last_response": search_request("c6", "r6", text="python")}
    )

    assert update["stop_reason"] == "tool_limit"
    assert "limit of 5 searches" in update["final_answer"]
