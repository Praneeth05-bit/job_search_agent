"""Tests for the offline demo model, alone and driving the real agent."""

import json
from datetime import date
from pathlib import Path

import pytest

from job_search_agent.agent.answer import AgentAnswer
from job_search_agent.agent.graph import SearchAgent
from job_search_agent.agent.nodes import first_turn_input
from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.models import StudentProfile
from job_search_agent.llm.demo import DemoModelClient
from job_search_agent.llm.schemas import ToolOutput, ToolSpec
from job_search_agent.storage.tracker import InMemoryTracker
from job_search_agent.tools.search_tool import search_tool_spec

FIXTURES = Path(__file__).parents[2] / "fixtures"
TODAY = date(2026, 11, 4)
SEARCH: list[ToolSpec] = [search_tool_spec()]


def _tool_output(*ids: str) -> ToolOutput:
    payload = {
        "matches": [
            {
                "opportunity_id": i,
                "matched_terms": ["python"],
                "eligibility": "eligible",
            }
            for i in ids
        ],
        "excluded": [],
    }
    return ToolOutput(
        call_id="c1",
        name="search_opportunities",
        result=f"<tool_result>\n{json.dumps(payload)}\n</tool_result>",
    )


def test_first_turn_searches_with_the_request_keywords() -> None:
    model = DemoModelClient()

    response = model.generate(first_turn_input("Find me Python internships"), SEARCH)

    (call,) = response.tool_calls
    assert call.name == "search_opportunities"
    assert call.arguments == {"text": "python internships"}  # filler words removed
    assert call.id


def test_without_tools_or_keywords_it_does_not_search() -> None:
    model = DemoModelClient()

    assert not model.generate(first_turn_input("python")).wants_tools
    assert not model.generate(first_turn_input("find me some"), SEARCH).wants_tools


def test_answer_cites_only_matches_from_the_conversation() -> None:
    model = DemoModelClient()
    after_tools = model.generate(tool_outputs=[_tool_output("opp-1", "opp-2")])

    response = model.generate(
        previous_response_id=after_tools.response_id,
        response_schema=AgentAnswer.model_json_schema(),
    )

    answer = AgentAnswer.model_validate_json(response.text or "")
    assert [(m.opportunity_id, m.rank) for m in answer.matches] == [
        ("opp-1", 1),
        ("opp-2", 2),
    ]
    assert "python" in answer.matches[0].why


def test_answer_is_capped_and_deduplicated() -> None:
    model = DemoModelClient()
    after_tools = model.generate(
        tool_outputs=[_tool_output("a", "b", "a"), _tool_output("c", "d")]
    )

    response = model.generate(
        previous_response_id=after_tools.response_id, response_schema={}
    )

    answer = AgentAnswer.model_validate_json(response.text or "")
    assert [m.opportunity_id for m in answer.matches] == ["a", "b", "c"]


@pytest.mark.parametrize("previous", [None, "unknown-id"])
def test_answer_without_known_matches_is_valid_and_empty(previous: str | None) -> None:
    response = DemoModelClient().generate(
        previous_response_id=previous, response_schema={}
    )

    answer = AgentAnswer.model_validate_json(response.text or "")
    assert answer.matches == []


def test_error_tool_results_give_no_matches() -> None:
    model = DemoModelClient()
    failed = ToolOutput(call_id="c1", name="x", result="bad", is_error=True)
    after_tools = model.generate(tool_outputs=[failed])

    response = model.generate(
        previous_response_id=after_tools.response_id, response_schema={}
    )

    assert AgentAnswer.model_validate_json(response.text or "").matches == []


def test_demo_model_drives_the_real_agent_to_a_grounded_answer() -> None:
    profile = StudentProfile.model_validate_json(
        (FIXTURES / "profiles" / "demo_student.json").read_text()
    )
    agent = SearchAgent(
        model=DemoModelClient(),
        opportunities=load_opportunities(FIXTURES / "opportunities"),
        today=TODAY,
        max_tool_calls=5,
        tracker=InMemoryTracker(),
    )

    run = agent.start("Find Python internships", profile)

    assert run.stop_reason == "completed"  # passed schema and grounding checks
    assert run.status == "awaiting_approval"
    assert run.answer is not None
    assert run.answer.matches
    assert run.model_calls_made == 3
