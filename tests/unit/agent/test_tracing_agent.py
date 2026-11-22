"""The trace an agent run leaves behind: which events, in what order, and safe.

Runs go through the real graph with a scripted model. Log events are caught
with structlog's capture_logs(), which also applies merge_contextvars so the
run's thread_id and prompt_id are visible, as they are in real output.
"""

from collections.abc import Iterator

import pytest
from agent_helpers import TODAY, answer, json_answer, search_request
from structlog.contextvars import merge_contextvars
from structlog.testing import capture_logs
from structlog.typing import EventDict

from job_search_agent.agent.graph import SearchAgent
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.llm.errors import ModelUnavailableError
from job_search_agent.llm.mock import MockModelClient
from job_search_agent.llm.schemas import ModelResponse, ToolCall
from job_search_agent.storage.tracker import InMemoryTracker

REQUEST = "Find Python internships, my name is Alex Example"
SUMMARY = "Two internships fit your profile."


@pytest.fixture
def logs() -> Iterator[list[EventDict]]:
    """Every log event written during the test, as dictionaries."""
    with capture_logs(processors=[merge_contextvars]) as entries:
        yield entries


def _agent(
    mock: MockModelClient, records: list[Opportunity], max_tool_calls: int = 5
) -> SearchAgent:
    return SearchAgent(
        model=mock,
        opportunities=records,
        today=TODAY,
        max_tool_calls=max_tool_calls,
        tracker=InMemoryTracker(),
    )


def _with_usage(
    response: ModelResponse, tokens_in: int, tokens_out: int
) -> ModelResponse:
    return response.model_copy(
        update={"input_tokens": tokens_in, "output_tokens": tokens_out}
    )


def _happy_script() -> MockModelClient:
    """Search, finish searching, answer; each reply reports token usage."""
    return MockModelClient(
        [
            _with_usage(search_request("c1", "r1", text="python"), 100, 10),
            _with_usage(answer("Done searching."), 200, 5),
            _with_usage(json_answer("opp-006", "opp-003", summary=SUMMARY), 300, 40),
        ]
    )


def _events(logs: list[EventDict], name: str) -> list[EventDict]:
    return [entry for entry in logs if entry["event"] == name]


# ---------------------------------------------------------------------------
# What the trace contains
# ---------------------------------------------------------------------------


def test_a_full_run_logs_each_step_in_order(
    logs: list[EventDict], profile: StudentProfile, records: list[Opportunity]
) -> None:
    agent = _agent(_happy_script(), records)

    paused = agent.start(REQUEST, profile)
    agent.resume(paused.thread_id, approved=True)

    assert [entry["event"] for entry in logs] == [
        "run_started",
        "model_call",  # turn 1: asks for a search
        "tool_call",
        "model_call",  # turn 2: done searching
        "model_call",  # turn 3: the JSON answer
        "answer_checked",
        "run_paused",  # waiting for approval
        "run_resumed",
        "approval_decided",
        "shortlist_saved",
        "run_finished",
    ]


def test_every_event_carries_the_run_ids(
    logs: list[EventDict], profile: StudentProfile, records: list[Opportunity]
) -> None:
    agent = _agent(_happy_script(), records)

    paused = agent.start(REQUEST, profile)
    agent.resume(paused.thread_id, approved=False)

    assert logs  # something was logged
    for entry in logs:
        assert entry["thread_id"] == paused.thread_id
        assert entry["prompt_id"] == "search_agent@v1"


def test_model_calls_show_step_turn_and_tokens(
    logs: list[EventDict], profile: StudentProfile, records: list[Opportunity]
) -> None:
    _agent(_happy_script(), records).start(REQUEST, profile)

    calls = _events(logs, "model_call")
    assert [(c["step"], c["turn"]) for c in calls] == [
        ("agent", 1),
        ("agent", 2),
        ("compose", 3),
    ]
    assert [c["input_tokens"] for c in calls] == [100, 200, 300]
    assert [c["tool_calls"] for c in calls] == [1, 0, 0]
    assert all(c["latency_ms"] >= 0 for c in calls)


def test_token_totals_reach_the_run_and_its_last_event(
    logs: list[EventDict], profile: StudentProfile, records: list[Opportunity]
) -> None:
    run = _agent(_happy_script(), records).start(REQUEST, profile)

    assert (run.input_tokens, run.output_tokens) == (600, 55)
    (paused,) = _events(logs, "run_paused")
    assert paused["input_tokens"] == 600
    assert paused["output_tokens"] == 55
    assert paused["model_calls"] == 3
    assert paused["tool_calls"] == 1


def test_tool_call_logs_counts_and_argument_names(
    logs: list[EventDict], profile: StudentProfile, records: list[Opportunity]
) -> None:
    run = _agent(_happy_script(), records).start(REQUEST, profile)

    (event,) = _events(logs, "tool_call")
    report = run.tool_results[0].report
    assert report is not None
    assert event["tool"] == "search_opportunities"
    assert event["ok"] is True
    assert event["argument_names"] == ["text"]  # names only, never values
    assert event["matches"] == len(report.matches)
    assert event["excluded"] == len(report.excluded)


def test_a_repaired_answer_logs_both_checks(
    logs: list[EventDict], profile: StudentProfile, records: list[Opportunity]
) -> None:
    mock = MockModelClient(
        [
            search_request("c1", "r1", text="python"),
            answer("Done searching."),
            json_answer("opp-999"),  # not a search result: fails grounding
            json_answer("opp-006"),
        ]
    )

    _agent(mock, records).start(REQUEST, profile)

    checks = _events(logs, "answer_checked")
    assert [(c["attempt"], c["valid"]) for c in checks] == [(1, False), (2, True)]
    assert checks[0]["problems"] >= 1
    assert [c["turn"] for c in _events(logs, "model_call")] == [1, 2, 3, 4]


def test_tool_limit_is_visible_in_the_trace(
    logs: list[EventDict], profile: StudentProfile, records: list[Opportunity]
) -> None:
    two_searches = ModelResponse(
        response_id="r1",
        tool_calls=[
            ToolCall(id="c1", name="search_opportunities", arguments={"text": "a"}),
            ToolCall(id="c2", name="search_opportunities", arguments={"text": "b"}),
        ],
    )
    mock = MockModelClient([two_searches, search_request("c3", "r2", text="c")])

    run = _agent(mock, records, max_tool_calls=1).start(REQUEST, profile)

    assert run.stop_reason == "tool_limit"
    (skipped,) = _events(logs, "tool_call_skipped")
    assert skipped["call_id"] == "c2"
    assert skipped["reason"] == "tool_limit"
    assert len(_events(logs, "tool_limit_reached")) == 1
    assert _events(logs, "run_finished")[0]["stop_reason"] == "tool_limit"


def test_a_model_failure_logs_where_it_happened(
    logs: list[EventDict], profile: StudentProfile, records: list[Opportunity]
) -> None:
    mock = MockModelClient(
        [search_request("c1", "r1", text="python"), ModelUnavailableError("503")]
    )

    with pytest.raises(ModelUnavailableError):
        _agent(mock, records).start(REQUEST, profile)

    (error,) = _events(logs, "model_error")
    assert (error["step"], error["turn"]) == ("agent", 2)
    assert error["error_type"] == "ModelUnavailableError"
    assert error["retryable"] is True
    (failed,) = _events(logs, "run_failed")
    assert failed["error_type"] == "ModelUnavailableError"
    assert failed["log_level"] == "error"


# ---------------------------------------------------------------------------
# What the trace must never contain
# ---------------------------------------------------------------------------


def test_trace_never_contains_personal_data_or_model_text(
    logs: list[EventDict], profile: StudentProfile, records: list[Opportunity]
) -> None:
    """The privacy rule, enforced: no profile, request, arguments, or output."""
    agent = _agent(_happy_script(), records)
    paused = agent.start(REQUEST, profile)
    agent.resume(paused.thread_id, approved=True)

    everything = repr(logs)
    forbidden = [
        "Alex Example",  # from the request text
        "Computer Science",  # profile: major
        "2028-05-20",  # profile: graduation date
        "authorized_to_work_in_us",  # profile: work authorization
        "masters",  # profile: degree
        "python",  # the tool argument's value
        SUMMARY,  # model output
        "Fits the request.",  # model output ("why")
        "Done searching.",  # model output
    ]
    for text in forbidden:
        assert text not in everything, f"{text!r} leaked into the logs"

    risky_keys = {"profile", "request", "prompt", "text", "arguments", "gpa"}
    for entry in logs:
        assert risky_keys.isdisjoint(entry), f"risky key in {entry['event']}"
