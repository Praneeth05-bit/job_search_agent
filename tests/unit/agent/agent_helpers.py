"""Scripted model replies and constants shared by the agent tests."""

import json
from datetime import date
from pathlib import Path
from typing import Any

from job_search_agent.llm.schemas import ModelResponse, ToolCall

TODAY = date(2026, 11, 4)
FIXTURES = Path(__file__).parents[2] / "fixtures"


def search_request(call_id: str, response_id: str, **arguments: Any) -> ModelResponse:
    """A scripted model reply that asks for one search."""
    return ModelResponse(
        response_id=response_id,
        tool_calls=[
            ToolCall(id=call_id, name="search_opportunities", arguments=arguments)
        ],
    )


def answer(text: str, response_id: str = "r-done") -> ModelResponse:
    """A scripted model reply saying it is done searching (plain text)."""
    return ModelResponse(response_id=response_id, text=text)


def json_answer(
    *opportunity_ids: str,
    summary: str = "Here are your matches.",
    response_id: str = "r-json",
    **extra: Any,
) -> ModelResponse:
    """A scripted structured answer recommending these ids, in rank order."""
    matches = [
        {"opportunity_id": opp_id, "rank": rank, "why": "Fits the request."}
        for rank, opp_id in enumerate(opportunity_ids, start=1)
    ]
    payload = {"summary": summary, "matches": matches, **extra}
    return ModelResponse(response_id=response_id, text=json.dumps(payload))
