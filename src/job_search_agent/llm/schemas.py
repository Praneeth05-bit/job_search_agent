"""Provider-neutral request and response types for the model layer.

The agent only ever sees these types, never a provider's SDK objects. That
keeps Google's library inside one adapter file, so switching providers or
models later does not ripple through the rest of the code.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """A tool the model may ask to call: its name, purpose, and argument schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Any]  # JSON schema for the arguments


class ToolCall(BaseModel):
    """A request from the model to run one tool with specific arguments.

    The model never runs anything itself. Our code validates the arguments
    and decides whether to execute the call.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str | None = None  # provider's call id, needed to send the result back
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolOutput(BaseModel):
    """The result of one tool call, sent back to the model on the next turn.

    ``call_id`` must match the ``ToolCall.id`` the model sent, so the model
    knows which request this result answers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    result: str
    is_error: bool = False


class ModelResponse(BaseModel):
    """One model reply: either final text, tool calls, or both, plus usage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_id: str | None = None  # provider's id for this reply, for follow-ups
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @property
    def wants_tools(self) -> bool:
        """True when the model is asking our code to run at least one tool."""
        return bool(self.tool_calls)
