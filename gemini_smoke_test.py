"""Run the live Gemini readiness checks for SpartanScout."""

from __future__ import annotations

import os
from typing import Any

from google import genai
from pydantic import BaseModel


MODEL = "gemini-3.6-flash"


class StructuredCheck(BaseModel):
    """Expected response from the structured-output check."""

    passed: bool
    message: str


SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "search_opportunities",
    "description": "Searches approved campus opportunity records.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms for matching opportunities.",
            }
        },
        "required": ["query"],
    },
}


def check_structured_output(client: genai.Client) -> None:
    """Verify that Gemini returns JSON matching the required schema."""
    interaction = client.interactions.create(
        model=MODEL,
        input=(
            "Return passed=true and a short message saying that Gemini "
            "structured output works."
        ),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": StructuredCheck.model_json_schema(),
        },
    )

    result = StructuredCheck.model_validate_json(interaction.output_text)
    if not result.passed:
        raise RuntimeError(f"Structured-output check returned passed=false: {result}")

    print(f"Structured output: PASS ({result.message})")


def check_function_call(client: genai.Client) -> None:
    """Verify that Gemini selects the expected tool with a valid query."""
    interaction = client.interactions.create(
        model=MODEL,
        input=(
            "Use the search_opportunities tool to find Python automation "
            "internships."
        ),
        tools=[SEARCH_TOOL],
    )

    calls = [
        step
        for step in (interaction.steps or [])
        if step.type == "function_call"
    ]
    if not calls:
        raise RuntimeError("Function-calling check failed: no function call returned.")

    call = calls[0]
    if call.name != "search_opportunities":
        raise RuntimeError(f"Unexpected function selected: {call.name}")

    query = call.arguments.get("query") if call.arguments else None
    if not isinstance(query, str) or not query.strip():
        raise RuntimeError(f"Function call contained an invalid query: {call.arguments}")

    print(f"Function calling: PASS ({call.name}, query={query!r})")


def main() -> None:
    """Run both Gemini smoke checks and report their results."""
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured in this terminal.")

    print(f"Testing model: {MODEL}")
    client = genai.Client()
    try:
        check_structured_output(client)
        check_function_call(client)
    finally:
        client.close()

    print("Gemini preflight: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
