"""Tests for the page's text helpers."""

from datetime import date

import pytest

from job_search_agent.api.schemas import RunUsage
from job_search_agent.domain.eligibility import EligibilityStatus
from job_search_agent.ui.formatting import (
    deadline_text,
    eligibility_badge,
    option_label,
    usage_text,
)

TODAY = date(2026, 11, 22)


def test_every_status_has_a_badge() -> None:
    for status in EligibilityStatus:
        assert eligibility_badge(status).label
    assert eligibility_badge(EligibilityStatus.ELIGIBLE).color == "green"
    assert eligibility_badge(EligibilityStatus.UNKNOWN).color == "orange"


@pytest.mark.parametrize(
    ("deadline", "expected"),
    [
        (None, "No deadline listed"),
        (date(2027, 1, 10), "Apply by Jan 10, 2027 (49 days left)"),
        (date(2026, 11, 23), "Apply by Nov 23, 2026 (1 day left)"),
        (date(2026, 11, 22), "Apply by Nov 22, 2026 (today)"),
        (date(2026, 11, 1), "Deadline passed (Nov 1, 2026)"),
    ],
)
def test_deadline_text(deadline: date | None, expected: str) -> None:
    assert deadline_text(deadline, TODAY) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "Not provided"),
        ("masters", "Masters"),
        ("authorized_to_work_in_us", "Authorized to work in US"),
    ],
)
def test_option_label(value: str | None, expected: str) -> None:
    assert option_label(value) == expected


def test_usage_text() -> None:
    usage = RunUsage(model_calls=3, tool_calls=1, input_tokens=900, output_tokens=80)

    assert usage_text(usage) == (
        "3 model calls · 1 tool call · 900 input / 80 output tokens"
    )
