"""A scripted, offline stand-in for a real model.

Used by tests, CI, and local development (USE_MOCK_MODEL=true). It returns
responses in the order they were scripted, can be scripted to raise errors,
and records every turn it was sent so tests can inspect it.
"""

from collections import deque
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from job_search_agent.llm.errors import ModelError, ModelResponseError
from job_search_agent.llm.schemas import ModelResponse, ToolOutput, ToolSpec


class RecordedTurn(BaseModel):
    """Everything the mock received on one generate() call."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    tools: list[ToolSpec]
    system_instruction: str | None
    previous_response_id: str | None
    tool_outputs: list[ToolOutput]


class MockModelClient:
    """A ModelClient that plays back a fixed script instead of calling an API."""

    def __init__(self, script: Sequence[ModelResponse | ModelError]) -> None:
        """Create a mock that will return or raise each script item in order.

        Args:
            script: Responses to return, or errors to raise, one per call.
        """
        self._script: deque[ModelResponse | ModelError] = deque(script)
        self.turns: list[RecordedTurn] = []

    @property
    def remaining(self) -> int:
        """How many scripted items have not been used yet."""
        return len(self._script)

    @property
    def prompts(self) -> list[str]:
        """The prompt sent on each call, in order."""
        return [turn.prompt for turn in self.turns]

    @property
    def tools_seen(self) -> list[list[ToolSpec]]:
        """The tools offered on each call, in order."""
        return [turn.tools for turn in self.turns]

    def generate(
        self,
        prompt: str = "",
        tools: Sequence[ToolSpec] = (),
        *,
        system_instruction: str | None = None,
        previous_response_id: str | None = None,
        tool_outputs: Sequence[ToolOutput] = (),
    ) -> ModelResponse:
        """Record the turn, then return or raise the next scripted item."""
        self.turns.append(
            RecordedTurn(
                prompt=prompt,
                tools=list(tools),
                system_instruction=system_instruction,
                previous_response_id=previous_response_id,
                tool_outputs=list(tool_outputs),
            )
        )

        if not self._script:
            raise ModelResponseError(
                f"Mock script exhausted after {len(self.turns) - 1} calls"
            )

        item = self._script.popleft()
        if isinstance(item, ModelError):
            raise item
        return item
