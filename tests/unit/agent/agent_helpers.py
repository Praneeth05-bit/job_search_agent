"""Scripted model replies and constants shared by the agent tests."""

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


def answer(text: str, response_id: str = "r-final") -> ModelResponse:
    """A scripted model reply with a final text answer."""
    return ModelResponse(response_id=response_id, text=text)
