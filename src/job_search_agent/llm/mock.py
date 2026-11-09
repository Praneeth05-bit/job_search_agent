"""A scripted, offline stand-in for a real model.

Used by tests, CI, and local development (USE_MOCK_MODEL=true). It returns
responses in the order they were scripted, can be scripted to raise errors,
and records everything it was sent so tests can inspect it.
"""

from collections import deque
from collections.abc import Sequence

from job_search_agent.llm.errors import ModelError, ModelResponseError
from job_search_agent.llm.schemas import ModelResponse, ToolSpec


class MockModelClient:
    """A ModelClient that plays back a fixed script instead of calling an API."""

    def __init__(self, script: Sequence[ModelResponse | ModelError]) -> None:
        """Create a mock that will return or raise each script item in order.

        Args:
            script: Responses to return, or errors to raise, one per call.
        """
        self._script: deque[ModelResponse | ModelError] = deque(script)
        self.prompts: list[str] = []
        self.tools_seen: list[list[ToolSpec]] = []

    @property
    def remaining(self) -> int:
        """How many scripted items have not been used yet."""
        return len(self._script)

    def generate(self, prompt: str, tools: Sequence[ToolSpec] = ()) -> ModelResponse:
        """Record the call, then return or raise the next scripted item."""
        self.prompts.append(prompt)
        self.tools_seen.append(list(tools))

        if not self._script:
            raise ModelResponseError(
                f"Mock script exhausted after {len(self.prompts) - 1} calls"
            )

        item = self._script.popleft()
        if isinstance(item, ModelError):
            raise item
        return item
