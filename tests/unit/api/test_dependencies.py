"""The real wiring, with no overrides: settings -> agent -> HTTP, in mock mode."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from job_search_agent.api import dependencies
from job_search_agent.api.app import app
from job_search_agent.api.dependencies import get_agent
from job_search_agent.config import get_settings

FIXTURES = Path(__file__).parents[2] / "fixtures"


@pytest.fixture
def mock_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Mock-mode settings with fixture data and a temporary tracker file."""
    tracker_file = tmp_path / "shortlist.json"
    monkeypatch.setenv("USE_MOCK_MODEL", "true")
    monkeypatch.setenv("DATA_DIR", str(FIXTURES / "opportunities"))
    monkeypatch.setenv("TRACKER_PATH", str(tracker_file))
    get_settings.cache_clear()
    dependencies._agent_for_day.cache_clear()
    yield tracker_file
    get_settings.cache_clear()
    dependencies._agent_for_day.cache_clear()


def test_one_agent_per_day_is_shared(mock_mode: Path) -> None:
    monday = get_agent(date(2026, 11, 2))

    assert get_agent(date(2026, 11, 2)) is monday  # same day: same agent
    assert get_agent(date(2026, 11, 3)) is not monday  # new day: new agent


def test_mock_mode_runs_end_to_end_over_http(mock_mode: Path) -> None:
    """No API key needed: the demo model answers and the tracker file is written."""
    client = TestClient(app)
    profile = {"degree_level": "masters", "major": "Computer Science"}

    started = client.post(
        "/agent/runs", json={"request": "python internships", "profile": profile}
    )
    assert started.status_code == 201, started.text
    thread_id = started.json()["thread_id"]

    finished = client.post(f"/agent/runs/{thread_id}/approval", json={"approved": True})

    assert finished.status_code == 200
    assert finished.json()["saved_ids"]
    assert mock_mode.exists()
