"""Where approved opportunities are saved: the student's shortlist tracker.

Saving changes data, so the agent only calls ``save`` after the student
approves. Two implementations share one small contract:

* InMemoryTracker: for tests and quick experiments.
* JsonFileTracker: a local JSON file for the demo.

Saving the same opportunity again replaces its earlier entry, so the
shortlist never holds duplicates.
"""

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class TrackerEntry(BaseModel):
    """One opportunity the student chose to keep."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opportunity_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    why: str
    saved_on: date


_ENTRIES = TypeAdapter(list[TrackerEntry])


class Tracker(Protocol):
    """Anything that can store and list shortlist entries."""

    def save(self, entries: Sequence[TrackerEntry]) -> None:
        """Add entries, replacing any existing entry with the same id."""
        ...

    def entries(self) -> list[TrackerEntry]:
        """All saved entries, ordered by opportunity id."""
        ...


def _merge(
    existing: Sequence[TrackerEntry], new: Sequence[TrackerEntry]
) -> list[TrackerEntry]:
    by_id = {entry.opportunity_id: entry for entry in existing}
    by_id.update({entry.opportunity_id: entry for entry in new})
    return sorted(by_id.values(), key=lambda entry: entry.opportunity_id)


class InMemoryTracker:
    """A tracker that keeps entries in memory only."""

    def __init__(self) -> None:
        self._entries: list[TrackerEntry] = []

    def save(self, entries: Sequence[TrackerEntry]) -> None:
        self._entries = _merge(self._entries, entries)

    def entries(self) -> list[TrackerEntry]:
        return list(self._entries)


class JsonFileTracker:
    """A tracker stored as a small JSON file on disk."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def save(self, entries: Sequence[TrackerEntry]) -> None:
        merged = _merge(self.entries(), entries)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [entry.model_dump(mode="json") for entry in merged]
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def entries(self) -> list[TrackerEntry]:
        if not self._path.is_file():
            return []
        return _ENTRIES.validate_json(self._path.read_text(encoding="utf-8"))
