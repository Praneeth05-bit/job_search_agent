"""The agent's nodes and router as plain, testable functions.

This module does not import LangGraph. Each node takes the state and returns
only the fields it changes, so nodes can be unit-tested directly. graph.py
wires them together.

Flow:
    call_model -> (router) -> run_tools -> call_model -> ...
                           -> compose  (model is done searching)
                           -> finish   (tool limit reached)
    compose -> request_approval (pause) -> save_shortlist (only if approved)

Guards enforced in code, not in the prompt:
    * At most ``max_tool_calls`` tool calls per run. Calls beyond the limit
      are not run; the model is told the limit was reached.
    * The profile, data, and date come from our dependencies, never from the
      model's tool arguments.
    * The final answer must be valid JSON in the AgentAnswer shape and cite
      only opportunities the search tool matched in this run. The model gets
      one repair attempt; after that the run stops with "invalid_answer".
    * Nothing is saved without the student's explicit approval. The approval
      node pauses the run; only an approval of exactly ``True`` saves.

Tracing: every model call, tool call, answer check, approval, and save writes
one structured log event (see observability/tracing.py). Events hold only
counts, ids, timings, and statuses, never the profile, the request text,
tool arguments, or model output.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from job_search_agent.agent.answer import AgentAnswer, parse_answer
from job_search_agent.agent.state import AgentState
from job_search_agent.domain.models import Opportunity
from job_search_agent.llm.client import ModelClient
from job_search_agent.llm.errors import ModelError
from job_search_agent.llm.schemas import ModelResponse, ToolCall, ToolOutput, ToolSpec
from job_search_agent.observability.tracing import elapsed_ms, get_logger
from job_search_agent.prompts.loader import Prompt
from job_search_agent.storage.tracker import Tracker, TrackerEntry
from job_search_agent.tools.search_tool import (
    ToolResult,
    model_facing_schema,
    run_search_tool,
    search_tool_spec,
)

log = get_logger(__name__)

Route = Literal["run_tools", "compose", "finish"]
ApprovalRoute = Literal["request_approval", "end"]
SaveRoute = Literal["save_shortlist", "end"]

# Pauses the run and returns the decision given when it resumes. In the
# graph this is LangGraph's interrupt(); tests pass a plain function.
AskApproval = Callable[[dict[str, Any]], Any]

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
    tracker: Tracker
    ask_approval: AskApproval


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


def usage_update(state: AgentState, *responses: ModelResponse) -> dict[str, int]:
    """Add the token counts of these responses to the run's running totals.

    A provider may not report usage (the mock does not); that counts as 0.
    """
    return {
        "input_tokens": state.get("input_tokens", 0)
        + sum(r.input_tokens or 0 for r in responses),
        "output_tokens": state.get("output_tokens", 0)
        + sum(r.output_tokens or 0 for r in responses),
    }


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
        turn = state.get("model_calls_made", 0) + 1
        if previous is None:
            response = self._generate(
                "agent", turn, first_turn_input(state["request"]), self._tools
            )
        else:
            response = self._generate(
                "agent",
                turn,
                tools=self._tools,
                previous_response_id=previous.response_id,
                tool_outputs=state.get("pending_outputs", []),
            )
        return {
            "last_response": response,
            "pending_outputs": [],
            "model_calls_made": turn,
            **usage_update(state, response),
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
                log.warning(
                    "tool_call_skipped",
                    tool=call.name,
                    call_id=call_id,
                    reason="tool_limit",
                    max_tool_calls=self._deps.max_tool_calls,
                )
                outputs.append(
                    ToolOutput(
                        call_id=call_id,
                        name=call.name,
                        result=LIMIT_MESSAGE,
                        is_error=True,
                    )
                )
                continue

            started = time.perf_counter()
            result = run_search_tool(
                ToolCall(id=call_id, name=call.name, arguments=call.arguments),
                profile=state["profile"],
                opportunities=self._deps.opportunities,
                today=self._deps.today,
            )
            made += 1  # failed calls count too, so mistakes cannot loop forever
            _trace_tool_call(result, call, elapsed_ms(started))
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

        turn = state.get("model_calls_made", 0)
        responses: list[ModelResponse] = []

        response = self._ask_for_json(
            COMPOSE_INSTRUCTION,
            previous.response_id if previous else None,
            turn=turn + 1,
        )
        responses.append(response)
        answer, problems = parse_answer(response.text, tool_results)
        _trace_answer_check(1, answer, problems)

        if answer is None:
            response = self._ask_for_json(
                repair_instruction(problems), response.response_id, turn=turn + 2
            )
            responses.append(response)
            answer, problems = parse_answer(response.text, tool_results)
            _trace_answer_check(2, answer, problems)

        update: dict[str, Any] = {
            "last_response": response,
            "model_calls_made": turn + len(responses),
            **usage_update(state, *responses),
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
        self, instruction: str, previous_response_id: str | None, *, turn: int
    ) -> ModelResponse:
        """One structured-output turn: no tools, the answer schema required."""
        return self._generate(
            "compose",
            turn,
            instruction,
            previous_response_id=previous_response_id,
            response_schema=self._answer_schema,
        )

    def _generate(
        self,
        step: str,
        turn: int,
        prompt: str = "",
        tools: Sequence[ToolSpec] = (),
        *,
        previous_response_id: str | None = None,
        tool_outputs: Sequence[ToolOutput] = (),
        response_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Make one model call with the system instruction, and trace it.

        Args:
            step: Which part of the agent is calling: "agent" or "compose".
            turn: The number of this model call in the run, starting at 1.

        Raises:
            ModelError: After logging which step and turn failed.
        """
        started = time.perf_counter()
        try:
            response = self._deps.model.generate(
                prompt,
                tools,
                system_instruction=self._deps.prompt.text,
                previous_response_id=previous_response_id,
                tool_outputs=tool_outputs,
                response_schema=response_schema,
            )
        except ModelError as error:
            log.warning(
                "model_error",
                step=step,
                turn=turn,
                error_type=type(error).__name__,
                retryable=error.retryable,
                latency_ms=elapsed_ms(started),
            )
            raise
        log.info(
            "model_call",
            step=step,
            turn=turn,
            latency_ms=elapsed_ms(started),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            tool_calls=len(response.tool_calls),
        )
        return response

    def finish(self, state: AgentState) -> dict[str, Any]:
        """End the run when the tool-call limit stopped the model."""
        log.warning("tool_limit_reached", max_tool_calls=self._deps.max_tool_calls)
        return {
            "answer": None,
            "stop_reason": "tool_limit",
            "final_answer": (
                f"I reached the limit of {self._deps.max_tool_calls} searches "
                "before finishing. Please try a more specific request."
            ),
        }

    def route_after_compose(self, state: AgentState) -> ApprovalRoute:
        """Ask for approval only when there is something to save."""
        answer = state.get("answer")
        if answer is not None and answer.matches:
            return "request_approval"
        return "end"

    def request_approval(self, state: AgentState) -> dict[str, Any]:
        """Pause and ask the student whether to save the recommended matches.

        This node has no side effects before the pause, because LangGraph
        runs it again from the start when the run resumes. Only a decision
        of exactly ``True`` counts as approval; anything else is a "no".
        """
        answer = state.get("answer")
        assert answer is not None, "request_approval requires an answer"
        proposal = {
            "action": "save_shortlist",
            "opportunity_ids": [m.opportunity_id for m in answer.matches],
        }
        decision = self._deps.ask_approval(proposal)
        # Logged after the pause, so it appears once, when the run resumes.
        log.info(
            "approval_decided",
            approved=decision is True,
            proposed=len(proposal["opportunity_ids"]),
        )
        return {"approved": decision is True}

    def route_after_approval(self, state: AgentState) -> SaveRoute:
        """Save only after an explicit approval."""
        return "save_shortlist" if state.get("approved") is True else "end"

    def save_shortlist(self, state: AgentState) -> dict[str, Any]:
        """Write the approved matches to the tracker."""
        answer = state.get("answer")
        if state.get("approved") is not True or answer is None:
            return {"saved_ids": []}  # defensive: never save without approval
        entries = [
            TrackerEntry(
                opportunity_id=match.opportunity_id,
                rank=match.rank,
                why=match.why,
                saved_on=self._deps.today,
            )
            for match in answer.matches
        ]
        self._deps.tracker.save(entries)
        log.info("shortlist_saved", saved=len(entries))
        return {"saved_ids": [entry.opportunity_id for entry in entries]}


def _trace_tool_call(result: ToolResult, call: ToolCall, latency_ms: float) -> None:
    """Log one tool call. Argument names only: their values may be personal."""
    report = result.report
    fields: dict[str, Any] = {
        "tool": result.name,
        "call_id": result.call_id,
        "ok": result.ok,
        "argument_names": sorted(call.arguments),
        "matches": len(report.matches) if report else 0,
        "excluded": len(report.excluded) if report else 0,
        "latency_ms": latency_ms,
    }
    if result.ok:
        log.info("tool_call", **fields)
    else:
        log.warning("tool_call", **fields)


def _trace_answer_check(
    attempt: int, answer: AgentAnswer | None, problems: Sequence[str]
) -> None:
    """Log whether an answer passed its checks.

    Only the number of problems is logged: the messages can quote model output.
    """
    log.info(
        "answer_checked",
        attempt=attempt,
        valid=answer is not None,
        problems=len(problems),
        matches=len(answer.matches) if answer else 0,
    )
