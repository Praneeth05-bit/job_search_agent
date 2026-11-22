"""Smoke tests of the Streamlit page, using Streamlit's AppTest (no browser).

AppTest runs the real script in-process. A fake API client is placed in
session state before the first run, so no server is needed.
"""

from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest
from ui_helpers import FakeApi

from job_search_agent.ui.api_client import ApiClient

APP = Path(__file__).parents[3] / "src" / "job_search_agent" / "ui" / "app.py"


def _page(transport: httpx.BaseTransport) -> AppTest:
    page = AppTest.from_file(str(APP), default_timeout=30)
    page.session_state["api_client"] = ApiClient(
        "http://api.test", timeout_seconds=5, transport=transport
    )
    return page.run()


def _search(page: AppTest, request: str) -> AppTest:
    page.text_area(key="request").input(request)
    return page.button(key="search").click().run()


def test_page_loads_with_the_demo_profile() -> None:
    page = _page(FakeApi().transport)

    assert not page.exception
    assert page.title[0].value == "🎓 SpartanScout"
    assert page.text_input(key="major").value == "Computer Science"


def test_search_shows_match_cards_and_asks_for_approval() -> None:
    api = FakeApi()
    page = _search(_page(api.transport), "python internships")

    assert not page.exception
    assert [h.value for h in page.subheader] == ["1. Python Backend Developer"]
    assert page.button(key="approve")  # the approval buttons are shown
    path, body = api.requests[0]
    assert path == "/agent/runs"
    assert body["request"] == "python internships"
    assert body["profile"]["major"] == "Computer Science"  # from the sidebar


def test_approve_saves_and_says_so() -> None:
    api = FakeApi()
    page = _search(_page(api.transport), "python internships")

    page = page.button(key="approve").click().run()

    assert not page.exception
    assert api.requests[-1] == ("/agent/runs/t-1/approval", {"approved": True})
    assert "Saved 1 opportunity" in page.success[0].value


def test_empty_request_shows_a_message_without_calling_the_api() -> None:
    api = FakeApi()
    page = _search(_page(api.transport), "   ")

    assert page.error[0].value == "Please describe what you are looking for."
    assert api.requests == []


def test_api_down_shows_a_friendly_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    page = _search(_page(httpx.MockTransport(refuse)), "python")

    assert not page.exception
    assert "Can't reach the SpartanScout API" in page.error[0].value
