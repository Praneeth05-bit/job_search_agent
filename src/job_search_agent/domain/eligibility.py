"""Deterministic eligibility rules for matching students to opportunities.

This module contains no LLM calls. Every decision is made by plain rules so
the result is repeatable, testable, and explainable. Each rule produces a
RuleCheck with a status and a human-readable reason, and the overall
EligibilityDecision keeps every check as evidence.

Every rule follows the same three questions, in this order:
    1. Does the job require anything?   No  -> ELIGIBLE
    2. Does the profile have the fact?  No  -> UNKNOWN
    3. Does the fact meet the rule?     Yes -> ELIGIBLE, No -> INELIGIBLE

The deadline is the one exception: a missing deadline is UNKNOWN, because
we cannot tell whether the opportunity is still open.

Work authorization is checked only by matching the value the student
stated. It is not immigration or legal advice.
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from job_search_agent.domain.models import Opportunity, StudentProfile


class EligibilityStatus(StrEnum):
    """The three possible outcomes of an eligibility check."""

    ELIGIBLE = "eligible"  # every known requirement is met
    INELIGIBLE = "ineligible"  # at least one requirement failed
    UNKNOWN = "unknown"  # data is missing, so we cannot decide


class RuleCheck(BaseModel):
    """The result of applying one rule to one opportunity and profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str = Field(min_length=1)
    status: EligibilityStatus
    reason: str = Field(min_length=1)


class EligibilityDecision(BaseModel):
    """The overall eligibility outcome for one opportunity, with evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opportunity_id: str = Field(min_length=1)
    status: EligibilityStatus
    checks: list[RuleCheck]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _check_allowed_values(
    rule: str, label: str, allowed: list[str] | None, value: str | None
) -> RuleCheck:
    """Check that a profile value is one of the job's accepted values.

    Used by the degree, major, and work authorization rules, which all ask
    the same question. Comparison ignores letter case.

    Args:
        rule: Rule name stored on the result, such as "major".
        label: Human-readable name used in the reason, such as "Major".
        allowed: Values the job accepts, or None if the job has no rule.
        value: The student's value, or None if not provided.

    Returns:
        A RuleCheck with ELIGIBLE, INELIGIBLE, or UNKNOWN and a reason.
    """
    if allowed is None:
        return RuleCheck(
            rule=rule,
            status=EligibilityStatus.ELIGIBLE,
            reason=f"No {label.lower()} requirement listed",
        )
    if value is None:
        return RuleCheck(
            rule=rule,
            status=EligibilityStatus.UNKNOWN,
            reason=f"Profile has no {label.lower()}",
        )

    accepted = {item.casefold() for item in allowed}
    if value.casefold() in accepted:
        return RuleCheck(
            rule=rule,
            status=EligibilityStatus.ELIGIBLE,
            reason=f"{label} '{value}' is accepted",
        )
    return RuleCheck(
        rule=rule,
        status=EligibilityStatus.INELIGIBLE,
        reason=f"{label} '{value}' is not accepted (accepted: {', '.join(allowed)})",
    )


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


def _check_deadline(opportunity: Opportunity, today: date) -> RuleCheck:
    """Check whether the application deadline has passed.

    The deadline day itself still counts as open. A missing deadline is
    UNKNOWN because the opportunity may already be closed.
    """
    deadline = opportunity.deadline
    if deadline is None:
        return RuleCheck(
            rule="deadline",
            status=EligibilityStatus.UNKNOWN,
            reason="No deadline listed; verify before applying",
        )
    if today > deadline:
        return RuleCheck(
            rule="deadline",
            status=EligibilityStatus.INELIGIBLE,
            reason=f"Deadline {deadline.isoformat()} has passed",
        )
    return RuleCheck(
        rule="deadline",
        status=EligibilityStatus.ELIGIBLE,
        reason=f"Open until {deadline.isoformat()}",
    )


def _check_degree_level(
    opportunity: Opportunity, profile: StudentProfile
) -> RuleCheck:
    """Check the student's degree level against the accepted levels."""
    return _check_allowed_values(
        rule="degree_level",
        label="Degree level",
        allowed=opportunity.eligibility.degree_levels,
        value=profile.degree_level,
    )


def _check_major(opportunity: Opportunity, profile: StudentProfile) -> RuleCheck:
    """Check the student's major against the accepted majors."""
    return _check_allowed_values(
        rule="major",
        label="Major",
        allowed=opportunity.eligibility.majors,
        value=profile.major,
    )


def _check_minimum_gpa(
    opportunity: Opportunity, profile: StudentProfile
) -> RuleCheck:
    """Check that the student's GPA meets the minimum. Equal passes."""
    minimum = opportunity.eligibility.minimum_gpa
    if minimum is None:
        return RuleCheck(
            rule="minimum_gpa",
            status=EligibilityStatus.ELIGIBLE,
            reason="No GPA requirement listed",
        )
    if profile.gpa is None:
        return RuleCheck(
            rule="minimum_gpa",
            status=EligibilityStatus.UNKNOWN,
            reason=f"Profile has no GPA (minimum is {minimum})",
        )
    if profile.gpa >= minimum:
        return RuleCheck(
            rule="minimum_gpa",
            status=EligibilityStatus.ELIGIBLE,
            reason=f"GPA {profile.gpa} meets minimum {minimum}",
        )
    return RuleCheck(
        rule="minimum_gpa",
        status=EligibilityStatus.INELIGIBLE,
        reason=f"GPA {profile.gpa} is below minimum {minimum}",
    )


def _check_graduation_window(
    opportunity: Opportunity, profile: StudentProfile
) -> RuleCheck:
    """Check that expected graduation falls inside the job's window.

    Either end of the window may be missing; only the bounds that exist are
    checked. Both boundary dates are inside the window.
    """
    start = opportunity.eligibility.graduation_start
    end = opportunity.eligibility.graduation_end
    graduation = profile.expected_graduation

    if start is None and end is None:
        return RuleCheck(
            rule="graduation_window",
            status=EligibilityStatus.ELIGIBLE,
            reason="No graduation window listed",
        )
    if graduation is None:
        return RuleCheck(
            rule="graduation_window",
            status=EligibilityStatus.UNKNOWN,
            reason="Profile has no expected graduation date",
        )
    if start is not None and graduation < start:
        return RuleCheck(
            rule="graduation_window",
            status=EligibilityStatus.INELIGIBLE,
            reason=f"Graduation {graduation.isoformat()} is before {start.isoformat()}",
        )
    if end is not None and graduation > end:
        return RuleCheck(
            rule="graduation_window",
            status=EligibilityStatus.INELIGIBLE,
            reason=f"Graduation {graduation.isoformat()} is after {end.isoformat()}",
        )
    return RuleCheck(
        rule="graduation_window",
        status=EligibilityStatus.ELIGIBLE,
        reason=f"Graduation {graduation.isoformat()} is inside the window",
    )


def _check_work_authorization(
    opportunity: Opportunity, profile: StudentProfile
) -> RuleCheck:
    """Check the student's stated work authorization against accepted values."""
    return _check_allowed_values(
        rule="work_authorization",
        label="Work authorization",
        allowed=opportunity.eligibility.work_authorizations,
        value=profile.work_authorization,
    )


# ---------------------------------------------------------------------------
# Combining results
# ---------------------------------------------------------------------------


def _combine(checks: list[RuleCheck]) -> EligibilityStatus:
    """Merge individual rule results into one overall status.

    Any INELIGIBLE wins; otherwise any UNKNOWN wins; otherwise ELIGIBLE.
    """
    statuses = {check.status for check in checks}
    if EligibilityStatus.INELIGIBLE in statuses:
        return EligibilityStatus.INELIGIBLE
    if EligibilityStatus.UNKNOWN in statuses:
        return EligibilityStatus.UNKNOWN
    return EligibilityStatus.ELIGIBLE


def evaluate_eligibility(
    opportunity: Opportunity, profile: StudentProfile, today: date
) -> EligibilityDecision:
    """Decide whether a student can apply to an opportunity.

    Args:
        opportunity: The job, internship, or research role to check.
        profile: The student's facts. Missing fields become UNKNOWN.
        today: The date to compare deadlines against. Passed in (instead of
            calling date.today()) so results are repeatable in tests.

    Returns:
        The overall status plus every individual rule check as evidence.
    """
    checks = [
        _check_deadline(opportunity, today),
        _check_degree_level(opportunity, profile),
        _check_major(opportunity, profile),
        _check_minimum_gpa(opportunity, profile),
        _check_graduation_window(opportunity, profile),
        _check_work_authorization(opportunity, profile),
    ]
    return EligibilityDecision(
        opportunity_id=opportunity.id,
        status=_combine(checks),
        checks=checks,
    )
