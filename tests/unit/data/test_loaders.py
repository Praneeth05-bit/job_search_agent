"""Unit tests for loading opportunity records from JSON files."""

from pathlib import Path

import pytest

from job_search_agent.data.loaders import load_opportunities, load_opportunity
from job_search_agent.domain.models import OpportunityType

FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "fixtures"
    / "opportunities"
    / "eligible_internship.json"
)
FIXTURE_DIR = FIXTURE_PATH.parent


def test_load_opportunity_returns_validated_model() -> None:
    """One JSON file becomes one validated Opportunity."""
    opportunity = load_opportunity(FIXTURE_PATH)

    assert opportunity.id == "opp-001"
    assert opportunity.title == "Software Engineering Intern"
    assert opportunity.opportunity_type is OpportunityType.INTERNSHIP
    assert opportunity.eligibility.minimum_gpa == 3.0


def test_load_opportunities_returns_all_records_sorted_by_id() -> None:
    """Every fixture file is loaded, in id order regardless of file order."""
    records = load_opportunities(FIXTURE_DIR)

    assert [record.id for record in records] == [
        "opp-001",
        "opp-002",
        "opp-003",
        "opp-004",
        "opp-005",
        "opp-006",
    ]


def test_load_opportunities_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Two files with the same id raise a clear error instead of loading."""
    # Arrange: two copies of the same record in a temporary folder.
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")
    (tmp_path / "a.json").write_text(fixture_text, encoding="utf-8")
    (tmp_path / "b.json").write_text(fixture_text, encoding="utf-8")

    # Act and assert: loading the folder must fail.
    with pytest.raises(ValueError, match="Duplicate opportunity id: opp-001"):
        load_opportunities(tmp_path)


def test_load_opportunities_rejects_missing_folder(tmp_path: Path) -> None:
    """A folder that does not exist raises instead of returning no records."""
    with pytest.raises(FileNotFoundError, match="Opportunity folder not found"):
        load_opportunities(tmp_path / "missing")
