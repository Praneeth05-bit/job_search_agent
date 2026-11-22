"""Tests for the agent endpoints: start, approve or reject, and error handling.

The agent is real (graph, tools, eligibility). Its model is the offline
DemoModelClient, or a scripted MockModelClient for failures, injected with
app.dependency_overrides, so no API key or network is needed.
"""

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from job_search_agent.agent.graph import SearchAgent
from job_search_agent.api.app import NO_WAITING_RUN, app
from job_search_agent.api.dependencies import get_agent, get_opportunities
from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.models import Opportunity
from job_search_agent.llm.client import ModelClient
from job_search_agent.llm.demo import DemoModelClient
from job_search_agent.llm.errors import (
    ModelAuthError,
    ModelError,
    ModelResponseError,
    ModelUnavailableError,
)
from job_search_agent.llm.mock import MockModelClient
from job_search_agent.storage.tracker import InMemoryTracker

FIXTURES = Path(__file__).parents[2] / "fixtures"
TODAY = date(2026, 11, 4)

ALEX: dict[str, Any] = {
    "degree_level": "masters",
    "major": "Computer Science",
    "gpa": 3.4,
    "expected_graduation": "2028-05-20",
    "work_authorization": "authorized_to_work_in_us",
}
PYTHON_RUN = {"request": "Find Python internships", "profile": ALEX}

ClientFactory = Callable[[ModelClient], TestClient]


@pytest.fixture
def records() -> list[Opportunity]:
    return load_opportunities(FIXTURES / "opportunities")


@pytest.fixture
def tracker() -> InMemoryTracker:
    return InMemoryTracker()


@pytest.fixture
def make_client(
    records: list[Opportunity], tracker: InMemoryTracker
) -> Iterator[ClientFactory]:
    """Build a test client whose shared agent uses the given model."""

    def make(model: ModelClient) -> TestClient:
        agent = SearchAgent(
            model=model,
            opportunities=records,
            today=TODAY,
            max_tool_calls=5,
            tracker=tracker,
        )
        app.dependency_overrides[get_agent] = lambda: agent  # one shared agent
        app.dependency_overrides[get_opportunities] = lambda: records
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


@pytest.fixture
def client(make_client: ClientFactory) -> TestClient:
    return make_client(DemoModelClient())


def _start(client: TestClient) -> dict[str, Any]:
    response = client.post("/agent/runs", json=PYTHON_RUN)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


# ---------------------------------------------------------------------------
# Start, approve, reject
# ---------------------------------------------------------------------------


def test_start_pauses_with_matches_to_approve(client: TestClient) -> None:
    body = _start(client)

    assert body["status"] == "awaiting_approval"
    assert body["thread_id"]
    assert body["matches"], "the demo model should recommend something"
    first = body["matches"][0]
    assert first["rank"] == 1
    assert first["title"]  # record details are joined in for the UI
    assert first["eligibility"] in {"eligible", "unknown"}
    assert first["source_url"].startswith("http")
    proposal = body["pending_approval"]
    assert proposal["opportunity_ids"] == [m["opportunity_id"] for m in body["matches"]]
    assert body["saved_ids"] == []
    assert body["usage"]["model_calls"] == 3
    assert body["usage"]["tool_calls"] == 1
    assert body["prompt_id"] == "search_agent@v1"


def test_approve_saves_the_shortlist(
    client: TestClient, tracker: InMemoryTracker
) -> None:
    started = _start(client)

    response = client.post(
        f"/agent/runs/{started['thread_id']}/approval", json={"approved": True}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "finished"
    assert body["approved"] is True
    assert body["saved_ids"] == started["pending_approval"]["opportunity_ids"]
    assert {e.opportunity_id for e in tracker.entries()} == set(body["saved_ids"])


def test_reject_saves_nothing(client: TestClient, tracker: InMemoryTracker) -> None:
    started = _start(client)

    response = client.post(
        f"/agent/runs/{started['thread_id']}/approval", json={"approved": False}
    )

    assert response.status_code == 200
    assert response.json()["saved_ids"] == []
    assert tracker.entries() == []


def test_no_matches_finishes_without_approval(client: TestClient) -> None:
    response = client.post(
        "/agent/runs", json={"request": "astronaut", "profile": ALEX}
    )

    body = response.json()
    assert body["status"] == "finished"
    assert body["matches"] == []
    assert body["pending_approval"] is None


def test_unknown_or_already_decided_run_is_404(client: TestClient) -> None:
    started = _start(client)
    url = f"/agent/runs/{started['thread_id']}/approval"
    client.post(url, json={"approved": True})

    again = client.post(url, json={"approved": True})
    unknown = client.post("/agent/runs/made-up-id/approval", json={"approved": True})

    for response in (again, unknown):
        assert response.status_code == 404
        assert response.json()["detail"] == NO_WAITING_RUN


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("approved", ["true", "yes", 1, None])
def test_approval_must_be_a_real_boolean(client: TestClient, approved: Any) -> None:
    """Strings and numbers never count as approval."""
    started = _start(client)

    response = client.post(
        f"/agent/runs/{started['thread_id']}/approval", json={"approved": approved}
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"request": "", "profile": ALEX},  # empty request
        {"request": "x" * 1001, "profile": ALEX},  # too long
        {"profile": ALEX},  # missing request
        {**PYTHON_RUN, "debug": True},  # unknown field
    ],
)
def test_bad_start_requests_are_422(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/agent/runs", json=body).status_code == 422


# ---------------------------------------------------------------------------
# Model failures become friendly errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ModelUnavailableError("HTTP 503 high demand"), 503),
        (ModelAuthError("HTTP 401"), 500),
        (ModelResponseError("no text"), 502),
    ],
)
def test_model_errors_return_the_friendly_message(
    make_client: ClientFactory, error: ModelError, code: int
) -> None:
    client = make_client(MockModelClient([error]))

    response = client.post("/agent/runs", json=PYTHON_RUN)

    assert response.status_code == code
    body = response.json()
    assert body["detail"] == error.user_message
    assert body["error_type"] == type(error).__name__
    assert str(error) not in response.text  # technical details stay in the logs


def test_overload_tells_the_client_when_to_retry(make_client: ClientFactory) -> None:
    client = make_client(MockModelClient([ModelUnavailableError("503")]))

    response = client.post("/agent/runs", json=PYTHON_RUN)

    assert response.headers["Retry-After"] == "30"


# ---------------------------------------------------------------------------
# What the response must never contain
# ---------------------------------------------------------------------------


def test_response_never_echoes_the_profile_or_internals(client: TestClient) -> None:
    response = client.post("/agent/runs", json=PYTHON_RUN)

    text = response.text
    for secret in ("2028-05-20", "authorized_to_work_in_us", "gpa", "masters"):
        assert secret not in text
    for internal in ("tool_results", "last_response", "profile"):
        assert internal not in response.json()
