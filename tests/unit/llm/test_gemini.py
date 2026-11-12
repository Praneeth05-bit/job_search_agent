"""Tests for the Gemini adapter, using a fake SDK. No network, no API key."""

from types import SimpleNamespace
from typing import Any

import pytest

from job_search_agent.config import Settings
from job_search_agent.llm.client import ModelClient
from job_search_agent.llm.errors import (
    ModelAuthError,
    ModelError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from job_search_agent.llm.gemini import (
    GeminiClient,
    create_gemini_client,
    translate_error,
)
from job_search_agent.llm.schemas import ToolOutput, ToolSpec

SEARCH_TOOL = ToolSpec(
    name="search_opportunities",
    description="Searches approved campus opportunity records.",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)


class FakeInteractions:
    """Stands in for sdk.interactions: records calls, returns or raises."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result
        self._error = error

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._result


class FakeHTTPError(Exception):
    """Mimics an SDK error that carries an HTTP status in `.code`."""

    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class ReadTimeout(Exception):
    """Named like the HTTP library's timeout error."""


class ConnectError(Exception):
    """Named like the HTTP library's connection error."""


def _interaction(**overrides: Any) -> SimpleNamespace:
    """A fake Interaction with the fields the adapter reads."""
    fields: dict[str, Any] = {
        "id": "int-123",
        "status": "completed",
        "output_text": None,
        "steps": [],
        "usage": SimpleNamespace(total_input_tokens=40, total_output_tokens=12),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _client(interactions: FakeInteractions) -> GeminiClient:
    sdk = SimpleNamespace(interactions=interactions)
    return GeminiClient(sdk, model="gemini-3.6-flash", timeout_seconds=30.0)


def test_adapter_satisfies_model_client_contract() -> None:
    """mypy checks this assignment: GeminiClient is a ModelClient."""
    client: ModelClient = _client(FakeInteractions(_interaction(output_text="hi")))

    assert client.generate("hello").text == "hi"


def test_sends_model_input_timeout_and_tools() -> None:
    """The request uses the configured model and timeout, and our tool format."""
    fake = FakeInteractions(_interaction(output_text="ok"))

    _client(fake).generate("find python internships", tools=[SEARCH_TOOL])

    call = fake.calls[0]
    assert call["model"] == "gemini-3.6-flash"
    assert call["input"] == "find python internships"
    assert call["timeout"] == 30.0
    assert call["tools"] == [
        {
            "type": "function",
            "name": "search_opportunities",
            "description": "Searches approved campus opportunity records.",
            "parameters": SEARCH_TOOL.parameters,
        }
    ]


def test_no_tools_key_when_no_tools() -> None:
    """Plain prompts do not send an empty tools list."""
    fake = FakeInteractions(_interaction(output_text="ok"))

    _client(fake).generate("hello")

    assert "tools" not in fake.calls[0]


def test_text_reply_and_usage_are_mapped() -> None:
    """Text, ids, and token counts come through; latency is measured."""
    response = _client(FakeInteractions(_interaction(output_text="Hello!"))).generate(
        "hi"
    )

    assert response.text == "Hello!"
    assert response.response_id == "int-123"
    assert response.input_tokens == 40
    assert response.output_tokens == 12
    assert response.latency_ms >= 0
    assert not response.wants_tools


def test_function_call_steps_become_tool_calls() -> None:
    """Only function_call steps are turned into ToolCalls, with their ids."""
    steps = [
        SimpleNamespace(type="thought"),
        SimpleNamespace(
            type="function_call",
            id="call-1",
            name="search_opportunities",
            arguments={"text": "python"},
        ),
    ]

    response = _client(FakeInteractions(_interaction(steps=steps))).generate("hi")

    assert response.wants_tools
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert (call.id, call.name, call.arguments) == (
        "call-1",
        "search_opportunities",
        {"text": "python"},
    )


def test_empty_reply_is_rejected() -> None:
    """A reply with no text and no tool call is unusable."""
    with pytest.raises(ModelResponseError, match="neither text nor a tool call"):
        _client(FakeInteractions(_interaction())).generate("hi")


def test_failed_status_is_rejected() -> None:
    """A failed interaction is reported, not treated as an answer."""
    fake = FakeInteractions(_interaction(status="failed", output_text="partial"))

    with pytest.raises(ModelResponseError, match="failed"):
        _client(fake).generate("hi")


def test_function_call_without_name_is_rejected() -> None:
    """Malformed tool calls are caught at the boundary."""
    steps = [SimpleNamespace(type="function_call", name="", arguments={})]

    with pytest.raises(ModelResponseError, match="no name"):
        _client(FakeInteractions(_interaction(steps=steps))).generate("hi")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FakeHTTPError(401), ModelAuthError),
        (FakeHTTPError(403), ModelAuthError),
        (FakeHTTPError(429), ModelRateLimitError),
        (FakeHTTPError(408), ModelTimeoutError),
        (FakeHTTPError(500), ModelUnavailableError),
        (FakeHTTPError(503), ModelUnavailableError),
        (FakeHTTPError(400), ModelResponseError),
        (ReadTimeout("slow"), ModelTimeoutError),
        (TimeoutError("slow"), ModelTimeoutError),
        (ConnectError("down"), ModelUnavailableError),
    ],
)
def test_provider_errors_are_translated(
    error: Exception, expected: type[ModelError]
) -> None:
    """Every provider failure becomes one of our errors, keeping the cause."""
    with pytest.raises(expected) as caught:
        _client(FakeInteractions(error=error)).generate("hi")

    assert caught.value.__cause__ is error


def test_unrelated_exception_is_not_translated() -> None:
    """A bug (not a provider problem) surfaces unchanged."""
    assert translate_error(ValueError("bug")) is None

    with pytest.raises(ValueError, match="bug"):
        _client(FakeInteractions(error=ValueError("bug"))).generate("hi")


def test_missing_api_key_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the real client without a key gives a clear, early error."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(gemini_api_key=None)

    with pytest.raises(ModelAuthError, match="GEMINI_API_KEY"):
        create_gemini_client(settings)


# ---------------------------------------------------------------------------
# Multi-turn
# ---------------------------------------------------------------------------


def test_follow_up_turn_sends_function_results_and_previous_id() -> None:
    """A tool result goes back with its call_id, continuing the earlier reply."""
    fake = FakeInteractions(_interaction(output_text="You qualify for opp-006."))

    _client(fake).generate(
        tools=[SEARCH_TOOL],
        system_instruction="Only recommend tool results.",
        previous_response_id="int-123",
        tool_outputs=[
            ToolOutput(
                call_id="call-1",
                name="search_opportunities",
                result="<tool_result>...</tool_result>",
            )
        ],
    )

    call = fake.calls[0]
    assert call["previous_interaction_id"] == "int-123"
    assert call["system_instruction"] == "Only recommend tool results."
    assert call["input"] == [
        {
            "type": "function_result",
            "call_id": "call-1",
            "name": "search_opportunities",
            "result": "<tool_result>...</tool_result>",
            "is_error": False,
        }
    ]
    assert "tools" in call  # tools are re-sent on every turn


def test_first_turn_omits_optional_fields() -> None:
    """No previous id or system instruction keys when they are not given."""
    fake = FakeInteractions(_interaction(output_text="ok"))

    _client(fake).generate("hello")

    assert "previous_interaction_id" not in fake.calls[0]
    assert "system_instruction" not in fake.calls[0]


def test_turn_without_prompt_or_tool_outputs_is_a_bug() -> None:
    """Calling with nothing to send is a programming error, raised before any call."""
    fake = FakeInteractions(_interaction(output_text="ok"))

    with pytest.raises(ValueError, match="prompt or tool outputs"):
        _client(fake).generate()

    assert fake.calls == []


# ---------------------------------------------------------------------------
# Provider messages
# ---------------------------------------------------------------------------


class FakeInteractionsServerError(Exception):
    """Mimics the Interactions client's error: status code plus a JSON body."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code
        self.body = {"error": {"message": message, "code": "service_unavailable"}}


def test_provider_message_from_body_is_kept_in_details() -> None:
    """The 'high demand' explanation from Google reaches our logs."""
    error = FakeInteractionsServerError(
        503, "gemini-3.6-flash is currently experiencing high demand."
    )

    translated = translate_error(error)

    assert isinstance(translated, ModelUnavailableError)
    assert str(translated) == (
        "Gemini server error (HTTP 503): "
        "gemini-3.6-flash is currently experiencing high demand."
    )
    assert translated.user_message.startswith("The AI model is busy right now")


def test_provider_message_attribute_is_used_and_flattened() -> None:
    """Errors with a .message attribute work too; long text is one line, capped."""
    error = FakeHTTPError(429)
    error.message = "Too many\nrequests " + "x" * 500  # type: ignore[attr-defined]

    translated = translate_error(error)

    assert translated is not None
    detail = str(translated).split(": ", 1)[1]
    assert "\n" not in detail
    assert len(detail) <= 200


def test_auth_error_never_echoes_provider_text() -> None:
    """Auth failures keep a fixed message, so nothing sensitive is repeated."""
    error = FakeHTTPError(401)
    error.message = "API key AIza-secret is invalid"  # type: ignore[attr-defined]

    translated = translate_error(error)

    assert "AIza-secret" not in str(translated)


def test_response_schema_requests_json_output() -> None:
    """Structured output uses the response_format shape verified in preflight."""
    fake = FakeInteractions(_interaction(output_text='{"summary": "ok"}'))
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}

    _client(fake).generate("answer as JSON", response_schema=schema)

    assert fake.calls[0]["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema,
    }


def test_no_response_format_without_schema() -> None:
    fake = FakeInteractions(_interaction(output_text="ok"))

    _client(fake).generate("hello")

    assert "response_format" not in fake.calls[0]
