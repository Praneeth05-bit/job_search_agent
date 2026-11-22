"""SpartanScout's Streamlit page: search, review the matches, approve or reject.

Run (with the API already running in another terminal):
    uv run streamlit run src/job_search_agent/ui/app.py

How Streamlit works, and how this page uses it:

* Streamlit runs this whole script again, top to bottom, after every click.
  Ordinary variables are lost on each rerun; ``st.session_state`` survives.
  The current run (``"run"``) and the last error (``"error"``) live there.
* Buttons use ``on_click`` callbacks. A callback runs *before* the rerun, so
  when the page is drawn again the session state already holds the result.
* The page only draws things. All work happens behind the API, through
  ``ApiClient``; tests replace the client via ``st.session_state["api_client"]``.

Privacy: the sidebar profile is pre-filled with a fictional demo student.
The profile is sent only to the SpartanScout API, never to the model.
"""

from datetime import date
from pathlib import Path

import streamlit as st

from job_search_agent.api.schemas import AgentRunResponse, MatchView
from job_search_agent.config import PROJECT_ROOT, get_settings
from job_search_agent.domain.models import StudentProfile
from job_search_agent.ui.api_client import ApiClient, ApiError
from job_search_agent.ui.formatting import (
    deadline_text,
    eligibility_badge,
    option_label,
    usage_text,
)

DEMO_PROFILE_PATH = PROJECT_ROOT / "data" / "demo" / "profiles" / "demo_student.json"
DEGREE_LEVELS: list[str | None] = [None, "bachelors", "masters", "phd"]
WORK_AUTHORIZATIONS: list[str | None] = [
    None,
    "authorized_to_work_in_us",
    "requires_sponsorship",
]
EXAMPLE_REQUEST = "Python internships for a CS master's student"


# ---------------------------------------------------------------------------
# State and actions
# ---------------------------------------------------------------------------


def get_client() -> ApiClient:
    """The API client: a test's fake if one was placed in session state."""
    injected = st.session_state.get("api_client")
    if isinstance(injected, ApiClient):
        return injected
    settings = get_settings()
    return ApiClient(
        settings.api_base_url, timeout_seconds=settings.ui_request_timeout_seconds
    )


def demo_profile(path: Path = DEMO_PROFILE_PATH) -> StudentProfile:
    """The fictional demo student, or an empty profile if the file is missing."""
    if not path.exists():
        return StudentProfile()
    return StudentProfile.model_validate_json(path.read_text(encoding="utf-8"))


def profile_from_form() -> StudentProfile:
    """Build the profile from the sidebar widgets' current values."""
    state = st.session_state
    return StudentProfile(
        degree_level=state.get("degree_level"),
        major=(state.get("major") or "").strip() or None,
        gpa=state.get("gpa"),
        expected_graduation=state.get("expected_graduation"),
        work_authorization=state.get("work_authorization"),
    )


def on_search() -> None:
    """Callback for the Search button: start a new run."""
    request = (st.session_state.get("request") or "").strip()
    st.session_state["run"] = None
    st.session_state["error"] = None
    if not request:
        st.session_state["error"] = "Please describe what you are looking for."
        return
    try:
        st.session_state["run"] = get_client().start_run(request, profile_from_form())
    except ApiError as error:
        st.session_state["error"] = error.message


def on_decide(approved: bool) -> None:
    """Callback for Approve and Reject: send the decision for the current run."""
    run: AgentRunResponse | None = st.session_state.get("run")
    if run is None:
        return
    try:
        st.session_state["run"] = get_client().decide(run.thread_id, approved=approved)
        st.session_state["error"] = None
    except ApiError as error:
        st.session_state["error"] = error.message


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_profile_form(default: StudentProfile) -> None:
    """The sidebar: the student's facts used for eligibility checks."""
    with st.sidebar:
        st.header("Your profile")
        st.caption("Used only for eligibility checks. Pre-filled with a demo student.")
        st.selectbox(
            "Degree level",
            DEGREE_LEVELS,
            index=_index(DEGREE_LEVELS, default.degree_level),
            format_func=option_label,
            key="degree_level",
        )
        st.text_input("Major", value=default.major or "", key="major")
        st.number_input(
            "GPA",
            min_value=0.0,
            max_value=4.0,
            value=default.gpa,
            step=0.1,
            format="%.2f",
            key="gpa",
            placeholder="Not provided",
        )
        st.date_input(
            "Expected graduation",
            value=default.expected_graduation,
            min_value=date(2020, 1, 1),
            max_value=date(2035, 12, 31),
            key="expected_graduation",
        )
        st.selectbox(
            "Work authorization",
            WORK_AUTHORIZATIONS,
            index=_index(WORK_AUTHORIZATIONS, default.work_authorization),
            format_func=option_label,
            key="work_authorization",
        )


def draw_search_box() -> None:
    """The request box and the Search button."""
    st.text_area(
        "What are you looking for?",
        key="request",
        max_chars=1000,
        placeholder=EXAMPLE_REQUEST,
    )
    st.button("Search", type="primary", key="search", on_click=on_search)


def draw_match(match: MatchView, today: date) -> None:
    """One recommendation as a card."""
    badge = eligibility_badge(match.eligibility)
    with st.container(border=True):
        st.subheader(f"{match.rank}. {match.title}")
        st.badge(badge.label, icon=badge.icon, color=badge.color)
        st.caption(f"{match.organization} · {deadline_text(match.deadline, today)}")
        st.write(match.why)
        for caveat in match.caveats:
            st.caption(f"⚠️ {caveat}")
        st.link_button("View posting", str(match.source_url))


def draw_run(run: AgentRunResponse, today: date) -> None:
    """The result of a run: summary, match cards, and the approval step."""
    st.info(run.summary)
    if run.clarifying_question:
        st.warning(run.clarifying_question)

    for match in run.matches:
        draw_match(match, today)

    if run.status == "awaiting_approval":
        st.write(
            "**Save these to your shortlist?** Nothing is saved until you approve."
        )
        approve, reject = st.columns(2)
        approve.button(
            "Approve and save",
            type="primary",
            key="approve",
            on_click=on_decide,
            args=(True,),
        )
        reject.button("Reject", key="reject", on_click=on_decide, args=(False,))
    elif run.approved is True:
        count = len(run.saved_ids)
        noun = "opportunity" if count == 1 else "opportunities"
        st.success(f"Saved {count} {noun} to your shortlist.")
    elif run.approved is False:
        st.warning("Nothing was saved.")

    with st.expander("Run details"):
        st.caption(usage_text(run.usage))
        st.caption(f"Prompt {run.prompt_id} · stop reason: {run.stop_reason}")
        st.caption(f"Run id: {run.thread_id}")


def _index(options: list[str | None], value: str | None) -> int:
    return options.index(value) if value in options else 0


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def main() -> None:
    """Draw the whole page. Streamlit calls this on every rerun."""
    st.set_page_config(page_title="SpartanScout", page_icon="🎓")
    st.title("🎓 SpartanScout")
    st.caption(
        "Find campus opportunities you are eligible for. Every recommendation "
        "comes from real search results, and nothing is saved without your OK."
    )

    draw_profile_form(demo_profile())
    draw_search_box()

    error: str | None = st.session_state.get("error")
    if error:
        st.error(error)
    run: AgentRunResponse | None = st.session_state.get("run")
    if run is not None:
        draw_run(run, date.today())


main()
