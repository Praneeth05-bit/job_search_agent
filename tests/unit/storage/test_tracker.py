"""Tests for the shortlist trackers."""

from datetime import date
from pathlib import Path

import pytest

from job_search_agent.storage.tracker import (
    InMemoryTracker,
    JsonFileTracker,
    Tracker,
    TrackerEntry,
)

TODAY = date(2026, 11, 4)


def _entry(opportunity_id: str, rank: int = 1, why: str = "fits") -> TrackerEntry:
    return TrackerEntry(
        opportunity_id=opportunity_id, rank=rank, why=why, saved_on=TODAY
    )


@pytest.fixture(params=["memory", "file"])
def tracker(request: pytest.FixtureRequest, tmp_path: Path) -> Tracker:
    """Run every test against both implementations."""
    if request.param == "memory":
        return InMemoryTracker()
    return JsonFileTracker(tmp_path / "tracker" / "shortlist.json")


def test_starts_empty(tracker: Tracker) -> None:
    assert tracker.entries() == []


def test_saves_and_lists_sorted_by_id(tracker: Tracker) -> None:
    tracker.save([_entry("opp-006", 1), _entry("opp-003", 2)])

    assert [e.opportunity_id for e in tracker.entries()] == ["opp-003", "opp-006"]


def test_saving_again_replaces_instead_of_duplicating(tracker: Tracker) -> None:
    tracker.save([_entry("opp-006", why="first")])
    tracker.save([_entry("opp-006", why="second")])

    entries = tracker.entries()
    assert len(entries) == 1
    assert entries[0].why == "second"


def test_file_tracker_persists_between_instances(tmp_path: Path) -> None:
    """A new tracker object reads what an earlier one wrote."""
    path = tmp_path / "shortlist.json"
    JsonFileTracker(path).save([_entry("opp-006")])

    assert [e.opportunity_id for e in JsonFileTracker(path).entries()] == ["opp-006"]
