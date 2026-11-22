"""Tests for the UI's API client, against a fake transport (no server)."""

from collections.abc import Callable

import httpx
import pytest
from ui_helpers import FakeApi, run_json

from job_search_agent.domain.models import StudentProfile
from job_search_agent.ui.api_client import ApiClient, ApiError

PROFILE = StudentProfile(major="Computer Science", gpa=3.4)
Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, base_url: str = "http://api.test") -> ApiClient:
    return ApiClient(
        base_url, timeout_seconds=5, transport=httpx.MockTransport(handler)
    )


def test_start_run_sends_request_and_profile() -> None:
    api = FakeApi()
    client = ApiClient("http://api.test/", timeout_seconds=5, transport=api.transport)

    run = client.start_run("python internships", PROFILE)

    assert run.status == "awaiting_approval"
    assert run.matches[0].title == "Python Backend Developer"
    path, body = api.requests[0]
    assert path == "/agent/runs"  # the trailing "/" in the base URL is handled
    assert body["request"] == "python internships"
    assert body["profile"]["major"] == "Computer Science"
    assert body["profile"]["gpa"] == 3.4


def test_decide_sends_the_decision_for_the_run() -> None:
    api = FakeApi()
    client = ApiClient("http://api.test", timeout_seconds=5, transport=api.transport)

    run = client.decide("t-1", approved=True)

    assert run.saved_ids == ["opp-003"]
    assert api.requests == [("/agent/runs/t-1/approval", {"approved": True})]


def test_api_error_detail_is_passed_through() -> None:
    message = "The AI model is busy right now. Please try again in a few minutes."
    client = _client(lambda _: httpx.Response(503, json={"detail": message}))

    with pytest.raises(ApiError) as caught:
        client.start_run("python", PROFILE)

    assert caught.value.message == message
    assert caught.value.status_code == 503


def test_validation_errors_name_the_field() -> None:
    detail = [
        {"loc": ["body", "request"], "msg": "String should have at least 1 character"}
    ]
    client = _client(lambda _: httpx.Response(422, json={"detail": detail}))

    with pytest.raises(ApiError, match="Please check request: String should"):
        client.start_run("x", PROFILE)


def test_unreadable_error_gets_a_generic_message() -> None:
    client = _client(lambda _: httpx.Response(500, text="<html>oops</html>"))

    with pytest.raises(ApiError, match=r"Something went wrong \(HTTP 500\)"):
        client.start_run("python", PROFILE)


def test_wrong_response_shape_is_reported() -> None:
    client = _client(lambda _: httpx.Response(201, json={"unexpected": True}))

    with pytest.raises(ApiError, match="could not read"):
        client.start_run("python", PROFILE)


def test_server_not_running() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ApiError, match=r"Can't reach .*http://api.test.*running"):
        _client(refuse).start_run("python", PROFILE)


def test_slow_server_times_out() -> None:
    def too_slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ApiError, match="took too long"):
        _client(too_slow).start_run("python", PROFILE)


def test_valid_response_round_trips() -> None:
    client = _client(lambda _: httpx.Response(201, json=run_json(summary="Hello")))

    assert client.start_run("python", PROFILE).summary == "Hello"
