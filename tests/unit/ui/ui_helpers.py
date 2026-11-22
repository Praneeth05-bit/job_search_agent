"""A fake SpartanScout API for UI tests, built on httpx.MockTransport."""

import json
from typing import Any

import httpx

from job_search_agent.api.schemas import AgentRunResponse

MATCH: dict[str, Any] = {
    "rank": 1,
    "opportunity_id": "opp-003",
    "title": "Python Backend Developer",
    "organization": "Spartan Campus IT Services",
    "deadline": "2027-01-10",
    "source_url": "https://example.com/opportunities/opp-003",
    "eligibility": "eligible",
    "why": "Matches your search terms (python).",
    "caveats": ["Confirm the start date"],
}


def run_json(**overrides: Any) -> dict[str, Any]:
    """A valid AgentRunResponse body, paused for approval unless overridden."""
    body: dict[str, Any] = {
        "thread_id": "t-1",
        "status": "awaiting_approval",
        "stop_reason": "completed",
        "summary": "Found 1 opportunity that matches your search.",
        "clarifying_question": None,
        "matches": [MATCH],
        "pending_approval": {
            "action": "save_shortlist",
            "opportunity_ids": ["opp-003"],
        },
        "approved": None,
        "saved_ids": [],
        "prompt_id": "search_agent@v1",
        "usage": {
            "model_calls": 3,
            "tool_calls": 1,
            "input_tokens": 900,
            "output_tokens": 80,
        },
    }
    body.update(overrides)
    AgentRunResponse.model_validate(body)  # keep the fake honest
    return body


FINISHED = run_json(
    status="finished", pending_approval=None, approved=True, saved_ids=["opp-003"]
)


class FakeApi:
    """Answers like the real API and records every request it receives."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path == "/agent/runs":
            return httpx.Response(201, json=run_json())
        if request.url.path == "/agent/runs/t-1/approval":
            return httpx.Response(200, json=FINISHED)
        return httpx.Response(404, json={"detail": "No run is waiting."})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
