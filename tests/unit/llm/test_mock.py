"""Tests for the scripted mock model and the model-layer types."""

import pytest
from pydantic import ValidationError

from job_search_agent.llm.client import ModelClient
from job_search_agent.llm.errors import (
    ModelAuthError,
    ModelError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from job_search_agent.llm.mock import MockModelClient
from job_search_agent.llm.schemas import ModelResponse, ToolCall, ToolOutput, ToolSpec

SEARCH_TOOL = ToolSpec(
    name="search_opportunities",
    description="Searches approved campus opportunity records.",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)


def test_mock_satisfies_model_client_contract() -> None:
    """mypy checks this assignment: the mock fulfills the ModelClient protocol."""
    client: ModelClient = MockModelClient([ModelResponse(text="hi")])

    assert client.generate("hello").text == "hi"


def test_returns_scripted_responses_in_order() -> None:
    """First a tool call, then a final answer: a typical agent round trip."""
    mock = MockModelClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(name="search_opportunities", arguments={"text": "python"})
                ]
            ),
            ModelResponse(text="I found 2 eligible internships."),
        ]
    )

    first = mock.generate("find python internships", tools=[SEARCH_TOOL])
    second = mock.generate("here are the tool results")

    assert first.wants_tools
    assert first.tool_calls[0].arguments == {"text": "python"}
    assert not second.wants_tools
    assert second.text == "I found 2 eligible internships."
    assert mock.remaining == 0


def test_records_prompts_and_tools() -> None:
    """Tests can inspect exactly what the agent sent to the model."""
    mock = MockModelClient([ModelResponse(text="ok")])

    mock.generate("find python internships", tools=[SEARCH_TOOL])

    assert mock.prompts == ["find python internships"]
    assert mock.tools_seen == [[SEARCH_TOOL]]


def test_raises_scripted_error() -> None:
    """A mock can simulate provider failures, such as a rate limit."""
    mock = MockModelClient([ModelRateLimitError("429 from provider")])

    with pytest.raises(ModelRateLimitError, match="429"):
        mock.generate("hello")


def test_exhausted_script_raises_clear_error() -> None:
    """Calling more times than scripted is a test bug, reported clearly."""
    mock = MockModelClient([ModelResponse(text="only one")])
    mock.generate("first")

    with pytest.raises(ModelResponseError, match="exhausted after 1 calls"):
        mock.generate("second")


@pytest.mark.parametrize(
    ("error_class", "retryable"),
    [
        (ModelTimeoutError, True),
        (ModelRateLimitError, True),
        (ModelUnavailableError, True),
        (ModelAuthError, False),
        (ModelResponseError, False),
    ],
)
def test_errors_declare_whether_retry_can_help(
    error_class: type[ModelError], retryable: bool
) -> None:
    """Temporary failures are retryable; permanent ones are not."""
    assert error_class.retryable is retryable
    assert issubclass(error_class, ModelError)


def test_model_response_rejects_negative_usage() -> None:
    """Latency and token counts can never be negative."""
    with pytest.raises(ValidationError):
        ModelResponse(text="x", input_tokens=-1)


def test_records_multi_turn_details() -> None:
    """Tests can check that the agent re-sends rules and returns tool results."""
    mock = MockModelClient([ModelResponse(text="done")])
    output = ToolOutput(call_id="c1", name="search_opportunities", result="{}")

    mock.generate(
        tools=[SEARCH_TOOL],
        system_instruction="rules",
        previous_response_id="r1",
        tool_outputs=[output],
    )

    turn = mock.turns[0]
    assert turn.system_instruction == "rules"
    assert turn.previous_response_id == "r1"
    assert turn.tool_outputs == [output]
    assert turn.tools == [SEARCH_TOOL]


@pytest.mark.parametrize(
    "error_class",
    [
        ModelError,
        ModelTimeoutError,
        ModelRateLimitError,
        ModelUnavailableError,
        ModelAuthError,
        ModelResponseError,
    ],
)
def test_every_error_has_a_friendly_user_message(
    error_class: type[ModelError],
) -> None:
    """Users get plain language, never codes or configuration hints."""
    message = error_class.user_message

    assert message.endswith(".")
    assert "HTTP" not in message
    assert "GEMINI_API_KEY" not in message


def test_records_response_schema() -> None:
    """Tests can check that the final turn asked for structured output."""
    mock = MockModelClient([ModelResponse(text="{}")])
    schema = {"type": "object"}

    mock.generate("final answer", response_schema=schema)

    assert mock.turns[0].response_schema == schema
