"""The agent's nodes and router as plain, testable functions.

This module does not import LangGraph. Each node takes the state and returns
only the fields it changes, so nodes can be unit-tested directly. graph.py
wires them together.

Flow:
    call_model -> (router) -> run_tools -> call_model -> ...
                           -> compose  (model is done searching)
                           -> finish   (tool limit reached)

Guards enforced in code, not in the prompt:
    * At most ``max_tool_calls`` tool calls per run. Calls beyond the limit
      are not run; the model is told the limit was reached.
    * The profile, data, and date come from our dependencies, never from the
      model's tool arguments.
    * The final answer must be valid JSON in the AgentAnswer shape and cite
      only opportunities the search tool matched in this run. The model gets
      one repair attempt; after that the run stops with "invalid_answer".
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from job_search_agent.agent.answer import AgentAnswer, parse_answer
from job_search_agent.agent.state import AgentState
from job_search_agent.domain.models import Opportunity
from job_search_agent.llm.client import ModelClient
from job_search_agent.llm.schemas import ModelResponse, ToolCall, ToolOutput
from job_search_agent.prompts.loader import Prompt
from job_search_agent.tools.search_tool import (
    ToolResult,
    model_facing_schema,
    run_search_tool,
    search_tool_spec,
)

Route = Literal["run_tools", "compose", "finish"]

LIMIT_MESSAGE = (
    "Tool call limit reached. Do not request more tools. "
    "Answer using the results you already have."
)

COMPOSE_INSTRUCTION = (
    "Now write your final answer for the student as JSON matching the provided "
    "schema. Rank only opportunities that appear under 'matches' in the search "
    "results, best first, and never include excluded ones. In 'why', explain "
    "each match using the search results. Put eligibility notes the student "
    "should verify in 'caveats'. If nothing fits, return an empty 'matches' "
    "list and say so in 'summary'. If you need more information from the "
    "student, set 'clarifying_question'."
)

INVALID_ANSWER_MESSAGE = (
    "Sorry, I could not put together a reliable answer this time. "
    "Please try again or rephrase your request."
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


def repair_instruction(problems: Sequence[str]) -> str:
    """The text sent to the model when its JSON answer failed a check."""
    listed = "\n".join(f"- {problem}" for problem in problems)
    return (
        "Your answer could not be used because of these problems:\n"
        f"{listed}\n"
        "Return corrected JSON matching the schema."
    )


class AgentNodes:
    """The agent's steps, bound to one set of dependencies."""

    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps
        self._tools = [search_tool_spec()]
        self._answer_schema = model_facing_schema(AgentAnswer)

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

        * The model wants tools and calls remain: run them.
        * The model wants tools but the limit is reached: finish.
        * The model is done searching: compose the structured answer.
        """
        response = state.get("last_response")
        made = state.get("tool_calls_made", 0)
        if response is not None and response.wants_tools:
            if made < self._deps.max_tool_calls:
                return "run_tools"
            return "finish"
        return "compose"

    def compose(self, state: AgentState) -> dict[str, Any]:
        """Ask for the final answer as JSON, check it, and allow one repair.

        The answer must pass the schema check and the grounding check (only
        opportunities the search tool matched in this run). If the first
        reply fails, the problems are sent back once. A second failure ends
        the run with ``invalid_answer`` instead of showing a bad answer.
        """
        previous = state.get("last_response")
        tool_results = state.get("tool_results", [])
        calls = 0

        response = self._ask_for_json(
            COMPOSE_INSTRUCTION, previous.response_id if previous else None
        )
        calls += 1
        answer, problems = parse_answer(response.text, tool_results)

        if answer is None:
            response = self._ask_for_json(
                repair_instruction(problems), response.response_id
            )
            calls += 1
            answer, problems = parse_answer(response.text, tool_results)

        update: dict[str, Any] = {
            "last_response": response,
            "model_calls_made": state.get("model_calls_made", 0) + calls,
        }
        if answer is None:
            update.update(
                answer=None,
                final_answer=INVALID_ANSWER_MESSAGE,
                stop_reason="invalid_answer",
            )
        else:
            update.update(
                answer=answer,
                final_answer=answer.summary,
                stop_reason="completed",
            )
        return update

    def _ask_for_json(
        self, instruction: str, previous_response_id: str | None
    ) -> ModelResponse:
        """One structured-output turn: no tools, the answer schema required."""
        return self._deps.model.generate(
            instruction,
            system_instruction=self._deps.prompt.text,
            previous_response_id=previous_response_id,
            response_schema=self._answer_schema,
        )

    def finish(self, state: AgentState) -> dict[str, Any]:
        """End the run when the tool-call limit stopped the model."""
        return {
            "answer": None,
            "stop_reason": "tool_limit",
            "final_answer": (
                f"I reached the limit of {self._deps.max_tool_calls} searches "
                "before finishing. Please try a more specific request."
            ),
        }
