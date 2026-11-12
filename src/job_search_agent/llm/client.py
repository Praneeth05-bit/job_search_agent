"""The contract every model client must fulfill.

The agent depends on this interface, not on a specific provider. Both the
real Gemini client and the mock satisfy it, so the agent cannot tell which
one it is using.
"""

from collections.abc import Sequence
from typing import Any, Protocol

from job_search_agent.llm.schemas import ModelResponse, ToolOutput, ToolSpec


class ModelClient(Protocol):
    """Anything that can send a prompt (with optional tools) to a model."""

    def generate(
        self,
        prompt: str = "",
        tools: Sequence[ToolSpec] = (),
        *,
        system_instruction: str | None = None,
        previous_response_id: str | None = None,
        tool_outputs: Sequence[ToolOutput] = (),
        response_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Send one turn to the model and return its reply.

        A turn is either a new ``prompt`` or, when continuing a conversation,
        the ``tool_outputs`` for the tool calls the model asked for.

        Args:
            prompt: The user's request for a new turn.
            tools: Tools the model may call. Must be re-sent every turn.
            system_instruction: Rules for the model. Must be re-sent every turn.
            previous_response_id: Continue from this earlier reply.
            tool_outputs: Results of the model's previous tool calls.
            response_schema: If given, the reply text must be JSON matching
                this schema (structured output).

        Raises:
            ModelError: Any subclass, when the call fails.
        """
        ...
