"""The agent's nodes and router as plain, testable functions.

This module does not import LangGraph. Each node takes the state and returns
only the fields it changes, so nodes can be unit-tested directly. graph.py
wires them together.

Flow:
    call_model -> (router) -> run_tools -> call_model -> ... -> finish

Guards enforced in code, not in the prompt:
    * At most ``max_tool_calls`` tool calls per run. Calls beyond the limit
      are not run; the model is told the limit was reached.
    * The profile, data, and date come from our dependencies, never from the
      model's tool arguments.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from job_search_agent.agent.state import AgentState
from job_search_agent.domain.models import Opportunity
from job_search_agent.llm.client import ModelClient
from job_search_agent.llm.schemas import ModelResponse, ToolCall, ToolOutput
from job_search_agent.prompts.loader import Prompt
from job_search_agent.tools.search_tool import (
    ToolResult,
    run_search_tool,
    search_tool_spec,
)

Route = Literal["run_tools", "finish"]

LIMIT_MESSAGE = (
    "Tool call limit reached. Do not request more tools. "
    "Answer using the results you already have."
)


@dataclass(frozen=True)
class AgentDependencies:
    """Everything the nodes need that is not part of the state."""

    model: ModelClient
    opportunities: Sequence[Opportunity]
    today: date
    prompt: Prompt
    max_tool_calls: int


def first_turn_input(request: str) -> str:
    """The text sent to the model on the first turn of a run.

    The profile itself is not sent: eligibility is computed by the tool from
    the profile in the state. This keeps personal details out of the prompt
    and saves tokens.
    """
    return (
        f"Student request:\n{request}\n\n"
        "The student's profile is on file. The search tool checks eligibility "
        "against it, so rely on the tool's eligibility results."
    )


class AgentNodes:
    """The agent's steps, bound to one set of dependencies."""

    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps
        self._tools = [search_tool_spec()]

    def call_model(self, state: AgentState) -> dict[str, Any]:
        """Ask the model what to do next: call a tool, or answer.

        The first turn sends the request. Later turns continue the previous
        reply and send the pending tool outputs. The system instruction and
        tools are re-sent on every turn.
        """
        previous = state.get("last_response")
        if previous is None:
            response = self._deps.model.generate(
                first_turn_input(state["request"]),
                self._tools,
                system_instruction=self._deps.prompt.text,
            )
        else:
            response = self._deps.model.generate(
                tools=self._tools,
                system_instruction=self._deps.prompt.text,
                previous_response_id=previous.response_id,
                tool_outputs=state.get("pending_outputs", []),
            )
        return {
            "last_response": response,
            "pending_outputs": [],
            "model_calls_made": state.get("model_calls_made", 0) + 1,
        }

    def run_tools(self, state: AgentState) -> dict[str, Any]:
        """Run the tool calls the model asked for, within the call limit.

        Every requested call gets an output, so the model can match each
        result to its request. Calls beyond the limit are not run.
        """
        response = state["last_response"]
        assert response is not None, "run_tools requires a model response"
        made = state.get("tool_calls_made", 0)

        results: list[ToolResult] = []
        outputs: list[ToolOutput] = []
        for call in response.tool_calls:
            call_id = call.id or f"call-{made + len(outputs) + 1}"
            if made >= self._deps.max_tool_calls:
                outputs.append(
                    ToolOutput(
                        call_id=call_id,
                        name=call.name,
                        result=LIMIT_MESSAGE,
                        is_error=True,
                    )
                )
                continue

            result = run_search_tool(
                ToolCall(id=call_id, name=call.name, arguments=call.arguments),
                profile=state["profile"],
                opportunities=self._deps.opportunities,
                today=self._deps.today,
            )
            made += 1  # failed calls count too, so mistakes cannot loop forever
            results.append(result)
            outputs.append(
                ToolOutput(
                    call_id=call_id,
                    name=call.name,
                    result=result.content,
                    is_error=not result.ok,
                )
            )

        return {
            "tool_results": results,
            "pending_outputs": outputs,
            "tool_calls_made": made,
        }

    def route_after_model(self, state: AgentState) -> Route:
        """Decide the next step after the model replies.

        Tools run only if the model asked for them and calls remain.
        Otherwise the run finishes.
        """
        response = state.get("last_response")
        made = state.get("tool_calls_made", 0)
        if (
            response is not None
            and response.wants_tools
            and made < self._deps.max_tool_calls
        ):
            return "run_tools"
        return "finish"

    def finish(self, state: AgentState) -> dict[str, Any]:
        """Record the final answer and why the run stopped."""
        response = state.get("last_response")
        if response is not None and response.wants_tools:
            return {
                "stop_reason": "tool_limit",
                "final_answer": (
                    f"I reached the limit of {self._deps.max_tool_calls} searches "
                    "before finishing. Please try a more specific request."
                ),
            }
        return {"stop_reason": "completed", "final_answer": _text_of(response)}


def _text_of(response: ModelResponse | None) -> str:
    """The model's final text, or an empty string if there is none."""
    if response is None or response.text is None:
        return ""
    return response.text
