"""Tests for the matching service: search followed by eligibility."""

from datetime import date
from pathlib import Path

import pytest

from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.eligibility import EligibilityStatus
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.retrieval.schemas import SearchQuery
from job_search_agent.services.matching import MatchReport, find_matches

TODAY = date(2026, 11, 4)
FIXTURES = Path(__file__).parents[2] / "fixtures"


@pytest.fixture
def profile() -> StudentProfile:
    """The fictional demo student, Alex."""
    path = FIXTURES / "profiles" / "demo_student.json"
    return StudentProfile.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.fixture
def records() -> list[Opportunity]:
    """All six synthetic opportunities."""
    return load_opportunities(FIXTURES / "opportunities")


def _ids(report: MatchReport) -> tuple[list[str], list[str]]:
    """Return (match ids, excluded ids) in order."""
    return (
        [match.search.opportunity_id for match in report.matches],
        [match.search.opportunity_id for match in report.excluded],
    )


def test_python_search_splits_eligible_and_excluded(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """End to end: search finds 4 jobs, eligibility removes 2."""
    report = find_matches(SearchQuery(text="python"), profile, records, TODAY)

    matches, excluded = _ids(report)
    assert matches == ["opp-003", "opp-006"]
    assert excluded == ["opp-001", "opp-002"]


def test_unknown_results_stay_in_matches(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """A job we cannot fully verify is kept, with its unknown status."""
    report = find_matches(SearchQuery(text="machine learning"), profile, records, TODAY)

    assert [m.search.opportunity_id for m in report.matches] == ["opp-004"]
    assert report.matches[0].eligibility.status is EligibilityStatus.UNKNOWN


def test_excluded_results_keep_their_reasons(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """Excluded jobs carry the failing checks so the user can see why."""
    report = find_matches(SearchQuery(text="python"), profile, records, TODAY)

    opp_001 = report.excluded[0]
    reasons = [
        check.reason
        for check in opp_001.eligibility.checks
        if check.status is EligibilityStatus.INELIGIBLE
    ]
    assert "Deadline 2026-10-15 has passed" in reasons


def test_no_search_results_gives_empty_report(
    profile: StudentProfile, records: list[Opportunity]
) -> None:
    """Nothing relevant means both lists are empty, not an error."""
    report = find_matches(SearchQuery(text="quantum"), profile, records, TODAY)

    assert report.matches == []
    assert report.excluded == []
