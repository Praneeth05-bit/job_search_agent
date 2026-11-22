"""A small HTTP client the UI uses to talk to the SpartanScout API.

The UI never calls the agent directly. It is one more API client, like
/docs: every rule, the approval step, and error handling stay on the server.

All failures become ``ApiError`` with a message that is safe and useful to
show on the page, so the page code only needs one ``except``.
"""

from typing import Any

import httpx
from pydantic import ValidationError

from job_search_agent.api.schemas import AgentRunResponse
from job_search_agent.domain.models import StudentProfile


class ApiError(Exception):
    """A request to the API failed. ``message`` is safe to show to the student."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ApiClient:
    """Calls the agent endpoints and returns validated ``AgentRunResponse``s."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create a client.

        Args:
            base_url: Where the API runs, such as "http://127.0.0.1:8000".
            timeout_seconds: How long to wait for one request before giving up.
            transport: Tests pass a fake transport; real use leaves it None.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport

    def start_run(self, request: str, profile: StudentProfile) -> AgentRunResponse:
        """Start an agent run (POST /agent/runs)."""
        body = {"request": request, "profile": profile.model_dump(mode="json")}
        return self._post("/agent/runs", body)

    def decide(self, thread_id: str, *, approved: bool) -> AgentRunResponse:
        """Approve or reject a paused run (POST /agent/runs/{id}/approval)."""
        return self._post(f"/agent/runs/{thread_id}/approval", {"approved": approved})

    def _post(self, path: str, body: dict[str, Any]) -> AgentRunResponse:
        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = client.post(path, json=body)
        except httpx.TimeoutException:
            raise ApiError(
                "The request took too long. The AI model may be slow right now; "
                "please try again."
            ) from None
        except httpx.TransportError:
            raise ApiError(
                f"Can't reach the SpartanScout API at {self._base_url}. Is it running?"
            ) from None

        if response.is_error:
            raise ApiError(_error_message(response), status_code=response.status_code)
        try:
            return AgentRunResponse.model_validate(response.json())
        except (ValueError, ValidationError):  # bad JSON or wrong shape
            raise ApiError("The API sent a response the page could not read.") from None


def _error_message(response: httpx.Response) -> str:
    """The friendliest message available for an error response."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None

    if isinstance(detail, str):  # our own errors: already friendly
        return detail
    if isinstance(detail, list) and detail:  # a 422 validation error from FastAPI
        first = detail[0]
        field = ".".join(str(part) for part in first.get("loc", [])[1:]) or "input"
        return f"Please check {field}: {first.get('msg', 'invalid value')}."
    return f"Something went wrong (HTTP {response.status_code}). Please try again."
