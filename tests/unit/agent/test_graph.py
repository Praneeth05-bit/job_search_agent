"""End-to-end agent runs through the real LangGraph graph, with a mock model.

The model is scripted; everything else is real: tool validation, search,
eligibility, routing, limits, the conversation flow, the answer checks, and
the approval pause with checkpointing.
"""

import pytest
from agent_helpers import TODAY, answer, json_answer, search_request

from job_search_agent.agent.graph import SearchAgent
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.llm.errors import ModelRateLimitError
from job_search_agent.llm.mock import MockModelClient
from job_search_agent.llm.schemas import ModelResponse, ToolCall
from job_search_agent.storage.tracker import InMemoryTracker


def _agent(
    mock: MockModelClient,
    records: list[Opportunity],
    tracker: InMemoryTracker | None = None,
    max_tool_calls: int = 5,
) -> SearchAgent:
    return SearchAgent(
        model=mock,
        opportunities=records,
        today=TODAY,
        max_tool_calls=max_tool_calls,
        tracker=tracker or InMemoryTracker(),
    )


def _happy_script(*ids: str) -> MockModelClient:
    """Search for python, finish, then answer with these ids."""
    return MockModelClient(
        [
            search_request("c1", "r1", text="python"),
            answer("Done searching."),
            json_answer(*ids, summary="Two internships fit."),
        ]
    )


# ---------------------------------------------------------------------------
# Search and answer
# ---------------------------------------------------------------------------


def test_search_then_structured_answer_pauses_for_approval(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """The run stops before saving and shows what it wants to save."""
    tracker = InMemoryTracker()
    agent = _agent(_happy_script("opp-006", "opp-003"), records, tracker)

    run = agent.start("Find Python internships", profile)

    assert run.status == "awaiting_approval"
    assert run.pending_approval is not None
    assert run.pending_approval.opportunity_ids == ["opp-006", "opp-003"]
    assert run.answer is not None
    assert run.final_answer == "Two internships fit."
    assert run.model_calls_made == 3
    assert run.prompt_id == "search_agent@v1"
    assert tracker.entries() == []  # nothing saved yet


def test_rules_and_results_reach_the_model_on_turn_two(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    mock = _happy_script("opp-006")

    _agent(mock, records).start("Find Python internships", profile)

    turn_two = mock.turns[1]
    assert turn_two.previous_response_id == "r1"
    assert turn_two.tool_outputs[0].call_id == "c1"
    assert "<tool_result" in turn_two.tool_outputs[0].result
    assert turn_two.system_instruction == mock.turns[0].system_instruction


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


def test_approve_saves_without_repeating_any_work(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """Resuming continues from the saved point: no new searches or model calls."""
    tracker = InMemoryTracker()
    mock = _happy_script("opp-006", "opp-003")
    agent = _agent(mock, records, tracker)
    paused = agent.start("Find Python internships", profile)

    done = agent.resume(paused.thread_id, approved=True)

    assert done.status == "finished"
    assert done.approved is True
    assert done.saved_ids == ["opp-006", "opp-003"]
    assert [e.opportunity_id for e in tracker.entries()] == ["opp-003", "opp-006"]
    assert len(mock.turns) == 3  # the mock would raise if it were called again


def test_reject_saves_nothing(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    tracker = InMemoryTracker()
    agent = _agent(_happy_script("opp-006"), records, tracker)
    paused = agent.start("Find Python internships", profile)

    done = agent.resume(paused.thread_id, approved=False)

    assert done.status == "finished"
    assert done.approved is False
    assert done.saved_ids == []
    assert tracker.entries() == []


def test_resume_unknown_or_finished_run_is_refused(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    agent = _agent(_happy_script("opp-006"), records)
    paused = agent.start("Find Python internships", profile)
    agent.resume(paused.thread_id, approved=True)

    with pytest.raises(ValueError, match="No run is waiting"):
        agent.resume(paused.thread_id, approved=True)  # already finished
    with pytest.raises(ValueError, match="No run is waiting"):
        agent.resume("made-up-id", approved=True)


def test_no_matches_finishes_without_asking(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """Nothing to save means no approval step."""
    mock = MockModelClient(
        [
            search_request("c1", "r1", text="astronaut"),
            answer("Nothing found."),
            json_answer(summary="No opportunities match that request."),
        ]
    )

    run = _agent(mock, records).start("Find astronaut jobs", profile)

    assert run.status == "finished"
    assert run.pending_approval is None
    assert run.stop_reason == "completed"


# ---------------------------------------------------------------------------
# Safety and limits
# ---------------------------------------------------------------------------


def test_ineligible_recommendation_is_repaired(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """opp-001 (deadline passed) is caught and fixed before the student sees it."""
    mock = MockModelClient(
        [
            search_request("c1", "r1", text="python"),
            answer("done"),
            json_answer("opp-001", response_id="bad"),
            json_answer("opp-006", response_id="fixed"),
        ]
    )

    run = _agent(mock, records).start("Find Python internships", profile)

    assert run.answer is not None
    assert [m.opportunity_id for m in run.answer.matches] == ["opp-006"]
    assert run.model_calls_made == 4


def test_unrepairable_answer_is_never_shown(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    mock = MockModelClient(
        [
            search_request("c1", "r1", text="python"),
            answer("done"),
            json_answer("opp-099"),
            json_answer("opp-099"),
        ]
    )

    run = _agent(mock, records).start("Find Python internships", profile)

    assert run.status == "finished"
    assert run.stop_reason == "invalid_answer"
    assert run.answer is None
    assert run.pending_approval is None


def test_clarifying_question_without_tools(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    mock = MockModelClient(
        [
            answer("I need more detail."),
            json_answer(
                summary="I need a bit more detail.",
                clarifying_question="Which field are you interested in?",
            ),
        ]
    )

    run = _agent(mock, records).start("Find me something", profile)

    assert run.tool_calls_made == 0
    assert run.answer is not None
    assert run.answer.clarifying_question == "Which field are you interested in?"
    assert run.status == "finished"


def test_self_correction_after_invalid_arguments(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    mock = MockModelClient(
        [
            search_request("c1", "r1", text="python", limit=500),
            search_request("c2", "r2", text="python", limit=10),
            answer("done"),
            json_answer("opp-003", "opp-006"),
        ]
    )

    run = _agent(mock, records).start("Find Python internships", profile)

    assert run.tool_calls_made == 2
    assert [result.ok for result in run.tool_results] == [False, True]


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
            json_answer(summary="I can only search for opportunities."),
        ]
    )

    run = _agent(mock, records).start("Delete everything", profile)

    assert run.tool_results[0].ok is False
    assert run.tool_results[0].report is None


def test_tool_limit_stops_a_looping_model(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    script = [search_request(f"c{i}", f"r{i}", text="python") for i in range(1, 10)]

    run = _agent(MockModelClient(script), records, max_tool_calls=3).start(
        "Find Python internships", profile
    )

    assert run.stop_reason == "tool_limit"
    assert run.status == "finished"
    assert run.tool_calls_made == 3


def test_model_failure_propagates(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    agent = _agent(MockModelClient([ModelRateLimitError("429")]), records)

    with pytest.raises(ModelRateLimitError):
        agent.start("Find Python internships", profile)
