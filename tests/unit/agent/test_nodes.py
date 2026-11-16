"""Unit tests for the agent's nodes and router, without running a graph."""

from datetime import date

import pytest
from agent_helpers import TODAY, answer, json_answer, search_request

from job_search_agent.agent.answer import AgentAnswer
from job_search_agent.agent.nodes import (
    INVALID_ANSWER_MESSAGE,
    LIMIT_MESSAGE,
    AgentDependencies,
    AgentNodes,
    AskApproval,
)
from job_search_agent.agent.state import AgentState
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.llm.mock import MockModelClient
from job_search_agent.llm.schemas import ModelResponse, ToolCall, ToolOutput
from job_search_agent.prompts.loader import load_prompt
from job_search_agent.storage.tracker import InMemoryTracker
from job_search_agent.tools.search_tool import ToolResult, run_search_tool


def _nodes(
    records: list[Opportunity],
    mock: MockModelClient | None = None,
    max_tool_calls: int = 5,
    today: date = TODAY,
    tracker: InMemoryTracker | None = None,
    decision: object = True,
    ask: AskApproval | None = None,
) -> AgentNodes:
    deps = AgentDependencies(
        model=mock or MockModelClient([]),
        opportunities=records,
        today=today,
        prompt=load_prompt("search_agent", 1),
        max_tool_calls=max_tool_calls,
        tracker=tracker or InMemoryTracker(),
        ask_approval=ask or (lambda proposal: decision),
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


def test_router_composes_when_model_is_done(records: list[Opportunity]) -> None:
    state: AgentState = {"last_response": answer("done"), "tool_calls_made": 1}
    assert _nodes(records).route_after_model(state) == "compose"


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
# compose and finish
# ---------------------------------------------------------------------------


def _search_results(
    records: list[Opportunity], profile: StudentProfile
) -> list[ToolResult]:
    """A real 'python' search: matches opp-003/006, excludes opp-001/002."""
    call = ToolCall(id="c1", name="search_opportunities", arguments={"text": "python"})
    return [run_search_tool(call, profile=profile, opportunities=records, today=TODAY)]


def test_compose_accepts_grounded_answer(
    records: list[Opportunity], profile: StudentProfile
) -> None:
    mock = MockModelClient([json_answer("opp-006", "opp-003")])
    state: AgentState = {
        "last_response": answer("done", response_id="r2"),
        "tool_results": _search_results(records, profile),
        "model_calls_made": 2,
    }

    update = _nodes(records, mock).compose(state)

    assert update["stop_reason"] == "completed"
    assert update["final_answer"] == "Here are your matches."
    assert [m.opportunity_id for m in update["answer"].matches] == [
        "opp-006",
        "opp-003",
    ]
    assert update["model_calls_made"] == 3


def test_compose_asks_for_json_without_tools(
    records: list[Opportunity], profile: StudentProfile
) -> None:
    """The compose turn continues the conversation and requires the schema."""
    mock = MockModelClient([json_answer("opp-006")])
    state: AgentState = {
        "last_response": answer("done", response_id="r2"),
        "tool_results": _search_results(records, profile),
    }

    _nodes(records, mock).compose(state)

    turn = mock.turns[0]
    assert turn.previous_response_id == "r2"
    assert turn.response_schema is not None
    assert "summary" in turn.response_schema["properties"]
    assert turn.tools == []
    assert turn.system_instruction == load_prompt("search_agent", 1).text


def test_compose_repairs_an_excluded_recommendation(
    records: list[Opportunity], profile: StudentProfile
) -> None:
    """First reply recommends ineligible opp-001; the problem goes back once."""
    mock = MockModelClient(
        [
            json_answer("opp-001", response_id="bad"),
            json_answer("opp-006", response_id="fixed"),
        ]
    )
    state: AgentState = {
        "last_response": answer("done"),
        "tool_results": _search_results(records, profile),
        "model_calls_made": 2,
    }

    update = _nodes(records, mock).compose(state)

    repair_turn = mock.turns[1]
    assert repair_turn.previous_response_id == "bad"
    assert "opp-001 was excluded as ineligible" in repair_turn.prompt
    assert update["stop_reason"] == "completed"
    assert update["model_calls_made"] == 4


def test_compose_gives_up_after_one_failed_repair(
    records: list[Opportunity], profile: StudentProfile
) -> None:
    """Two bad replies: no bad answer is shown, and no third attempt is made."""
    mock = MockModelClient([json_answer("opp-099"), json_answer("opp-099")])
    state: AgentState = {
        "last_response": answer("done"),
        "tool_results": _search_results(records, profile),
    }

    update = _nodes(records, mock).compose(state)

    assert update["stop_reason"] == "invalid_answer"
    assert update["answer"] is None
    assert update["final_answer"] == INVALID_ANSWER_MESSAGE
    assert len(mock.turns) == 2


def test_finish_reports_tool_limit(records: list[Opportunity]) -> None:
    update = _nodes(records).finish(
        {"last_response": search_request("c6", "r6", text="python")}
    )

    assert update["stop_reason"] == "tool_limit"
    assert update["answer"] is None
    assert "limit of 5 searches" in update["final_answer"]


# ---------------------------------------------------------------------------
# Approval and saving
# ---------------------------------------------------------------------------


def _answered_state(*ids: str) -> AgentState:
    text = json_answer(*ids).text
    assert text is not None
    return {"answer": AgentAnswer.model_validate_json(text)}


def test_approval_is_requested_only_when_there_are_matches(
    records: list[Opportunity],
) -> None:
    nodes = _nodes(records)

    assert nodes.route_after_compose(_answered_state("opp-006")) == "request_approval"
    assert nodes.route_after_compose(_answered_state()) == "end"
    assert nodes.route_after_compose({"answer": None}) == "end"


def test_approval_request_shows_the_proposed_ids(records: list[Opportunity]) -> None:
    seen: list[dict[str, object]] = []

    def record(proposal: dict[str, object]) -> bool:
        seen.append(proposal)
        return True

    _nodes(records, ask=record).request_approval(_answered_state("opp-006", "opp-003"))

    assert seen == [
        {"action": "save_shortlist", "opportunity_ids": ["opp-006", "opp-003"]}
    ]


@pytest.mark.parametrize(
    ("decision", "approved"),
    [(True, True), (False, False), ("yes", False), (None, False), (1, False)],
)
def test_only_an_explicit_true_counts_as_approval(
    records: list[Opportunity], decision: object, approved: bool
) -> None:
    """Anything other than exactly True is treated as "no"."""
    update = _nodes(records, decision=decision).request_approval(
        _answered_state("opp-006")
    )

    assert update == {"approved": approved}


def test_route_after_approval(records: list[Opportunity]) -> None:
    nodes = _nodes(records)

    assert nodes.route_after_approval({"approved": True}) == "save_shortlist"
    assert nodes.route_after_approval({"approved": False}) == "end"
    assert nodes.route_after_approval({}) == "end"


def test_save_writes_approved_matches(records: list[Opportunity]) -> None:
    tracker = InMemoryTracker()
    state = _answered_state("opp-006", "opp-003")
    state["approved"] = True

    update = _nodes(records, tracker=tracker).save_shortlist(state)

    assert update == {"saved_ids": ["opp-006", "opp-003"]}
    saved = {e.opportunity_id: e for e in tracker.entries()}
    assert saved["opp-006"].rank == 1
    assert saved["opp-006"].saved_on == TODAY


def test_save_refuses_without_approval(records: list[Opportunity]) -> None:
    """Defensive check: even if called directly, nothing is saved unapproved."""
    tracker = InMemoryTracker()
    state = _answered_state("opp-006")
    state["approved"] = False

    update = _nodes(records, tracker=tracker).save_shortlist(state)

    assert update == {"saved_ids": []}
    assert tracker.entries() == []
