"""Boundary tests for the domain models (StudentProfile and Opportunity)."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from job_search_agent.domain.models import Opportunity, StudentProfile

OPPORTUNITY_FIXTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "opportunities"
    / "eligible_internship.json"
)


@pytest.fixture
def opportunity_data() -> dict[str, Any]:
    """A valid opportunity as a plain dict, ready to be modified by a test."""
    data: dict[str, Any] = json.loads(OPPORTUNITY_FIXTURE.read_text(encoding="utf-8"))
    return data


# ---------------------------------------------------------------------------
# StudentProfile
# ---------------------------------------------------------------------------


def test_empty_profile_is_valid() -> None:
    """Every profile field is optional; missing facts become unknown later."""
    assert StudentProfile().gpa is None


@pytest.mark.parametrize("gpa", [0.0, 4.0])
def test_gpa_inside_bounds_is_accepted(gpa: float) -> None:
    """Both ends of the 0-4 scale are valid."""
    assert StudentProfile(gpa=gpa).gpa == gpa


@pytest.mark.parametrize("gpa", [-0.1, 4.1])
def test_gpa_outside_bounds_is_rejected(gpa: float) -> None:
    """Just below 0 and just above 4 are refused."""
    with pytest.raises(ValidationError):
        StudentProfile(gpa=gpa)


def test_invalid_graduation_date_is_rejected() -> None:
    """Dates must be real dates, not free text."""
    with pytest.raises(ValidationError):
        StudentProfile(expected_graduation="next spring")  # type: ignore[arg-type]


def test_profile_extra_field_is_rejected() -> None:
    """Unknown profile fields are refused instead of silently ignored."""
    with pytest.raises(ValidationError):
        StudentProfile(age=25)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Opportunity
# ---------------------------------------------------------------------------


def test_valid_fixture_data_is_accepted(opportunity_data: dict[str, Any]) -> None:
    """Baseline: the unmodified fixture validates, so failures below are real."""
    assert Opportunity.model_validate(opportunity_data).id == "opp-001"


def test_opportunity_extra_field_is_rejected(opportunity_data: dict[str, Any]) -> None:
    """A typo or unexpected field in a listing fails loudly."""
    opportunity_data["salary"] = 5000

    with pytest.raises(ValidationError):
        Opportunity.model_validate(opportunity_data)


def test_opportunity_bad_url_is_rejected(opportunity_data: dict[str, Any]) -> None:
    """Every listing must link to a real source URL."""
    opportunity_data["source_url"] = "not-a-url"

    with pytest.raises(ValidationError):
        Opportunity.model_validate(opportunity_data)


def test_opportunity_minimum_gpa_outside_bounds_is_rejected(
    opportunity_data: dict[str, Any],
) -> None:
    """A job's GPA requirement must be on the same 0-4 scale."""
    opportunity_data["eligibility"]["minimum_gpa"] = 4.5

    with pytest.raises(ValidationError):
        Opportunity.model_validate(opportunity_data)
