"""Unit tests for the deterministic eligibility rules.

Tests use a fixed date (TODAY) so results never change over time. Most tests
start from opp-006, where every requirement is present and the demo student
passes them all, then change exactly one field so the cause of each result is
obvious.
"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.eligibility import (
    EligibilityDecision,
    EligibilityStatus,
    RuleCheck,
    evaluate_eligibility,
)
from job_search_agent.domain.models import Opportunity, StudentProfile

TODAY = date(2026, 11, 4)
FIXTURES = Path(__file__).parents[2] / "fixtures"


@pytest.fixture
def profile() -> StudentProfile:
    """The fictional demo student, Alex."""
    path = FIXTURES / "profiles" / "demo_student.json"
    return StudentProfile.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.fixture
def opportunities() -> dict[str, Opportunity]:
    """All six synthetic opportunities, keyed by id."""
    records = load_opportunities(FIXTURES / "opportunities")
    return {record.id: record for record in records}


def _check_for(decision: EligibilityDecision, rule: str) -> RuleCheck:
    """Return the single check for one rule from a decision."""
    return next(check for check in decision.checks if check.rule == rule)


def _with_requirements(opportunity: Opportunity, **changes: object) -> Opportunity:
    """Copy an opportunity, changing fields inside its eligibility block."""
    requirements = opportunity.eligibility.model_copy(update=changes)
    return opportunity.model_copy(update={"eligibility": requirements})


# ---------------------------------------------------------------------------
# Whole-fixture outcomes (the prediction table)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("opportunity_id", "expected"),
    [
        pytest.param(
            "opp-001", EligibilityStatus.INELIGIBLE, id="opp-001_deadline_passed"
        ),
        pytest.param("opp-002", EligibilityStatus.INELIGIBLE, id="opp-002_wrong_major"),
        pytest.param("opp-003", EligibilityStatus.ELIGIBLE, id="opp-003_eligible"),
        pytest.param("opp-004", EligibilityStatus.UNKNOWN, id="opp-004_no_deadline"),
        pytest.param("opp-005", EligibilityStatus.ELIGIBLE, id="opp-005_eligible"),
        pytest.param("opp-006", EligibilityStatus.ELIGIBLE, id="opp-006_eligible"),
    ],
)
def test_fixture_statuses(
    opportunities: dict[str, Opportunity],
    profile: StudentProfile,
    opportunity_id: str,
    expected: EligibilityStatus,
) -> None:
    """Each fixture gets the overall status from the prediction table."""
    decision = evaluate_eligibility(opportunities[opportunity_id], profile, TODAY)

    assert decision.status is expected


# ---------------------------------------------------------------------------
# Deadline boundaries
# ---------------------------------------------------------------------------


def test_deadline_day_is_still_open(
    opportunities: dict[str, Opportunity], profile: StudentProfile
) -> None:
    """Applying on the deadline day itself is allowed."""
    job = opportunities["opp-006"].model_copy(update={"deadline": TODAY})

    decision = evaluate_eligibility(job, profile, TODAY)

    assert _check_for(decision, "deadline").status is EligibilityStatus.ELIGIBLE


def test_deadline_after_today_has_passed(
    opportunities: dict[str, Opportunity], profile: StudentProfile
) -> None:
    """One day after the deadline, the opportunity is closed."""
    yesterday = TODAY - timedelta(days=1)
    job = opportunities["opp-006"].model_copy(update={"deadline": yesterday})

    decision = evaluate_eligibility(job, profile, TODAY)

    assert _check_for(decision, "deadline").status is EligibilityStatus.INELIGIBLE


def test_missing_deadline_is_unknown(
    opportunities: dict[str, Opportunity], profile: StudentProfile
) -> None:
    """No listed deadline means we cannot say whether it is still open."""
    job = opportunities["opp-006"].model_copy(update={"deadline": None})

    decision = evaluate_eligibility(job, profile, TODAY)

    assert _check_for(decision, "deadline").status is EligibilityStatus.UNKNOWN


# ---------------------------------------------------------------------------
# GPA boundaries
# ---------------------------------------------------------------------------


def test_gpa_equal_to_minimum_passes(
    opportunities: dict[str, Opportunity], profile: StudentProfile
) -> None:
    """A GPA exactly at the minimum (3.0) meets the requirement."""
    student = profile.model_copy(update={"gpa": 3.0})

    decision = evaluate_eligibility(opportunities["opp-006"], student, TODAY)

    assert _check_for(decision, "minimum_gpa").status is EligibilityStatus.ELIGIBLE


def test_gpa_below_minimum_fails(
    opportunities: dict[str, Opportunity], profile: StudentProfile
) -> None:
    """A GPA just under the minimum fails the requirement."""
    student = profile.model_copy(update={"gpa": 2.9})

    decision = evaluate_eligibility(opportunities["opp-006"], student, TODAY)

    assert _check_for(decision, "minimum_gpa").status is EligibilityStatus.INELIGIBLE


# ---------------------------------------------------------------------------
# Allowed-value rules and missing data
# ---------------------------------------------------------------------------


def test_major_match_ignores_case(
    opportunities: dict[str, Opportunity], profile: StudentProfile
) -> None:
    """'computer science' matches 'Computer Science'."""
    student = profile.model_copy(update={"major": "computer science"})

    decision = evaluate_eligibility(opportunities["opp-006"], student, TODAY)

    assert _check_for(decision, "major").status is EligibilityStatus.ELIGIBLE


def test_missing_profile_fact_is_unknown(
    opportunities: dict[str, Opportunity],
) -> None:
    """When the job has a rule but the profile is empty, the result is unknown."""
    decision = evaluate_eligibility(opportunities["opp-006"], StudentProfile(), TODAY)

    major_check = _check_for(decision, "major")
    assert decision.status is EligibilityStatus.UNKNOWN
    assert major_check.status is EligibilityStatus.UNKNOWN
    assert "no major" in major_check.reason


def test_no_job_requirement_passes_without_profile_fact(
    opportunities: dict[str, Opportunity],
) -> None:
    """If the job has no major rule, a missing major does not matter."""
    decision = evaluate_eligibility(opportunities["opp-005"], StudentProfile(), TODAY)

    assert _check_for(decision, "major").status is EligibilityStatus.ELIGIBLE


# ---------------------------------------------------------------------------
# Graduation window boundaries
# ---------------------------------------------------------------------------


def test_graduation_on_start_date_is_inside_window(
    opportunities: dict[str, Opportunity], profile: StudentProfile
) -> None:
    """Graduating exactly on the window's start date is allowed."""
    student = profile.model_copy(update={"expected_graduation": date(2027, 5, 1)})

    decision = evaluate_eligibility(opportunities["opp-006"], student, TODAY)

    assert (
        _check_for(decision, "graduation_window").status is EligibilityStatus.ELIGIBLE
    )


def test_graduation_before_window_fails(
    opportunities: dict[str, Opportunity], profile: StudentProfile
) -> None:
    """Graduating one day before the window starts fails."""
    student = profile.model_copy(update={"expected_graduation": date(2027, 4, 30)})

    decision = evaluate_eligibility(opportunities["opp-006"], student, TODAY)

    assert (
        _check_for(decision, "graduation_window").status is EligibilityStatus.INELIGIBLE
    )


def test_graduation_only_end_bound_is_checked(
    opportunities: dict[str, Opportunity], profile: StudentProfile
) -> None:
    """With only an end date, graduating after it fails."""
    job = _with_requirements(
        opportunities["opp-006"],
        graduation_start=None,
        graduation_end=date(2028, 1, 1),
    )

    decision = evaluate_eligibility(job, profile, TODAY)

    assert (
        _check_for(decision, "graduation_window").status is EligibilityStatus.INELIGIBLE
    )


# ---------------------------------------------------------------------------
# Combining results
# ---------------------------------------------------------------------------


def test_ineligible_reports_every_failing_rule(
    opportunities: dict[str, Opportunity], profile: StudentProfile
) -> None:
    """All six rules run, so the student sees every reason at once."""
    decision = evaluate_eligibility(opportunities["opp-001"], profile, TODAY)

    failing = {
        check.rule
        for check in decision.checks
        if check.status is EligibilityStatus.INELIGIBLE
    }
    assert len(decision.checks) == 6
    assert failing == {"deadline", "degree_level"}


def test_ineligible_outranks_unknown(
    opportunities: dict[str, Opportunity],
) -> None:
    """One failed rule makes the result ineligible even if others are unknown."""
    job = opportunities["opp-006"].model_copy(
        update={"deadline": TODAY - timedelta(days=1)}
    )

    decision = evaluate_eligibility(job, StudentProfile(), TODAY)

    assert decision.status is EligibilityStatus.INELIGIBLE
