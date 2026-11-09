"""Tests for the HTTP API, using FastAPI's TestClient (no real server)."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from job_search_agent.api.app import DEMO_DATA_DIR, app, get_opportunities, get_today
from job_search_agent.data.loaders import load_opportunities

FIXTURES = Path(__file__).parents[2] / "fixtures"
TODAY = date(2026, 11, 4)

ALEX: dict[str, Any] = {
    "degree_level": "masters",
    "major": "Computer Science",
    "gpa": 3.4,
    "expected_graduation": "2028-05-20",
    "work_authorization": "authorized_to_work_in_us",
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client that uses fixture data and a fixed date."""
    app.dependency_overrides[get_opportunities] = lambda: load_opportunities(
        FIXTURES / "opportunities"
    )
    app.dependency_overrides[get_today] = lambda: TODAY
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_returns_ok(client: TestClient) -> None:
    """The liveness check responds with 200 and status ok."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_matches_and_exclusions(client: TestClient) -> None:
    """End to end over HTTP: same answer as the matching service."""
    response = client.post(
        "/search", json={"query": {"text": "python"}, "profile": ALEX}
    )

    assert response.status_code == 200
    body = response.json()
    matches = [m["search"]["opportunity_id"] for m in body["matches"]]
    excluded = [m["search"]["opportunity_id"] for m in body["excluded"]]
    assert matches == ["opp-003", "opp-006"]
    assert excluded == ["opp-001", "opp-002"]


def test_search_rejects_limit_over_30(client: TestClient) -> None:
    """The SearchQuery contract is enforced at the API boundary."""
    response = client.post(
        "/search", json={"query": {"text": "python", "limit": 50}, "profile": ALEX}
    )

    assert response.status_code == 422


def test_search_rejects_unknown_fields(client: TestClient) -> None:
    """Unexpected fields are refused, not silently ignored."""
    response = client.post(
        "/search",
        json={"query": {"text": "python"}, "profile": ALEX, "salary_min": 50000},
    )

    assert response.status_code == 422


def test_demo_data_folder_loads() -> None:
    """Smoke test of the real wiring: the app's own data folder exists and loads.

    The other API tests override get_opportunities, so they cannot catch a
    missing or misplaced demo folder. This test uses no overrides.
    """
    records = load_opportunities(DEMO_DATA_DIR)

    assert len(records) > 0
