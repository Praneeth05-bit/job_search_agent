"""The contract every model client must fulfill.

The agent depends on this interface, not on a specific provider. Both the
real Gemini client and the mock satisfy it, so the agent cannot tell which
one it is using.
"""

from collections.abc import Sequence
from typing import Protocol

from job_search_agent.llm.schemas import ModelResponse, ToolSpec


class ModelClient(Protocol):
    """Anything that can send a prompt (with optional tools) to a model."""

    def generate(self, prompt: str, tools: Sequence[ToolSpec] = ()) -> ModelResponse:
        """Send one prompt and return the model's reply.

        Raises:
            ModelError: Any subclass, when the call fails.
        """
        ...
