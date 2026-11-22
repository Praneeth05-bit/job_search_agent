"""Small, pure helpers that turn API data into text for the page.

Kept out of the Streamlit script so they can be unit tested without a page.
"""

from datetime import date
from typing import Literal, NamedTuple

from job_search_agent.api.schemas import RunUsage
from job_search_agent.domain.eligibility import EligibilityStatus

BadgeColor = Literal["green", "orange", "red"]


class Badge(NamedTuple):
    """How to show an eligibility status with ``st.badge``."""

    label: str
    icon: str
    color: BadgeColor


_BADGES: dict[EligibilityStatus, Badge] = {
    EligibilityStatus.ELIGIBLE: Badge("Eligible", "✅", "green"),
    EligibilityStatus.UNKNOWN: Badge("Check eligibility", "❓", "orange"),
    EligibilityStatus.INELIGIBLE: Badge("Not eligible", "⛔", "red"),
}


def eligibility_badge(status: EligibilityStatus) -> Badge:
    """The badge for an eligibility status."""
    return _BADGES[status]


def deadline_text(deadline: date | None, today: date) -> str:
    """A friendly deadline line, such as "Apply by Jan 10, 2027 (49 days left)"."""
    if deadline is None:
        return "No deadline listed"
    days = (deadline - today).days
    when = f"{deadline:%b} {deadline.day}, {deadline.year}"
    if days < 0:
        return f"Deadline passed ({when})"
    if days == 0:
        return f"Apply by {when} (today)"
    return f"Apply by {when} ({days} day{'s' if days != 1 else ''} left)"


def option_label(value: str | None) -> str:
    """Show a stored value as a readable label: "authorized_to_work_in_us" ->
    "Authorized to work in US"."""
    if value is None:
        return "Not provided"
    words = value.replace("_", " ").capitalize().split(" ")
    return " ".join("US" if word == "us" else word for word in words)


def usage_text(usage: RunUsage) -> str:
    """One line describing how much work the run took."""
    return (
        f"{_count(usage.model_calls, 'model call')} · "
        f"{_count(usage.tool_calls, 'tool call')} · "
        f"{usage.input_tokens} input / {usage.output_tokens} output tokens"
    )


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"
