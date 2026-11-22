"""Wire the agent's nodes into a LangGraph state graph and run it.

Written against LangGraph 1.2.x (``langgraph==1.2.11``,
``langgraph-checkpoint==4.2.0`` are pinned). APIs checked against the
LangGraph docs on 2026-11-10 (StateGraph) and 2026-11-16 (interrupts):
``StateGraph``, ``add_conditional_edges``, ``compile(checkpointer=...)``,
``interrupt()`` and ``Command(resume=...)`` from ``langgraph.types``,
``InMemorySaver`` from ``langgraph.checkpoint.memory``, and a
``thread_id`` in ``config["configurable"]``.

Graph:
    START -> call_model -(router)-> run_tools -> call_model ...
                        |-> finish  -> END   (tool limit reached)
                        \\-> compose -(has matches?)-> request_approval (pause)
                               -(approved?)-> save_shortlist -> END

Pausing for approval needs a checkpointer: the run's state is saved under a
thread id, and ``resume`` continues that thread without repeating the
searches or model calls. InMemorySaver keeps paused runs in this process
only; a restart forgets them (a durable checkpointer would be needed for
that).

Tracing: ``start`` and ``resume`` bind the thread id and prompt version to
every log event of the run (structlog context variables), and log when a run
starts, pauses, finishes, or fails, with its totals.
"""

import time
from collections.abc import Sequence
from datetime import date
from typing import Any, Literal
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict
from structlog.contextvars import bound_contextvars

from job_search_agent.agent.answer import AgentAnswer
from job_search_agent.agent.nodes import AgentDependencies, AgentNodes
from job_search_agent.agent.state import AgentState, StopReason
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.llm.client import ModelClient
from job_search_agent.observability.tracing import elapsed_ms, get_logger
from job_search_agent.prompts.loader import Prompt, load_prompt
from job_search_agent.storage.tracker import Tracker
from job_search_agent.tools.search_tool import ToolResult

log = get_logger(__name__)

PROMPT_NAME = "search_agent"
PROMPT_VERSION = 1


class ApprovalRequest(BaseModel):
    """What the agent wants to do and is waiting for the student to approve."""

    model_config = ConfigDict(frozen=True)

    action: str
    opportunity_ids: list[str]


class AgentRun(BaseModel):
    """The outcome of one agent run (or one resume), for the API, UI, and evals."""

    model_config = ConfigDict(frozen=True)

    thread_id: str
    status: Literal["awaiting_approval", "finished"]
    pending_approval: ApprovalRequest | None
    answer: AgentAnswer | None
    final_answer: str
    stop_reason: StopReason
    approved: bool | None
    saved_ids: list[str]
    tool_results: list[ToolResult]
    tool_calls_made: int
    model_calls_made: int
    input_tokens: int
    output_tokens: int
    prompt_id: str


def build_agent(deps: AgentDependencies, checkpointer: Any = None) -> Any:
    """Build and compile the agent graph for one set of dependencies."""
    nodes = AgentNodes(deps)

    graph = StateGraph(AgentState)
    graph.add_node("call_model", nodes.call_model)
    graph.add_node("run_tools", nodes.run_tools)
    graph.add_node("compose", nodes.compose)
    graph.add_node("finish", nodes.finish)
    graph.add_node("request_approval", nodes.request_approval)
    graph.add_node("save_shortlist", nodes.save_shortlist)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model",
        nodes.route_after_model,
        {"run_tools": "run_tools", "compose": "compose", "finish": "finish"},
    )
    graph.add_edge("run_tools", "call_model")
    graph.add_edge("finish", END)
    graph.add_conditional_edges(
        "compose",
        nodes.route_after_compose,
        {"request_approval": "request_approval", "end": END},
    )
    graph.add_conditional_edges(
        "request_approval",
        nodes.route_after_approval,
        {"save_shortlist": "save_shortlist", "end": END},
    )
    graph.add_edge("save_shortlist", END)

    return graph.compile(checkpointer=checkpointer)


class SearchAgent:
    """Runs the agent, pauses for approval, and resumes paused runs."""

    def __init__(
        self,
        *,
        model: ModelClient,
        opportunities: Sequence[Opportunity],
        today: date,
        max_tool_calls: int,
        tracker: Tracker,
        prompt: Prompt | None = None,
    ) -> None:
        self._prompt = prompt or load_prompt(PROMPT_NAME, PROMPT_VERSION)
        self._max_tool_calls = max_tool_calls
        deps = AgentDependencies(
            model=model,
            opportunities=opportunities,
            today=today,
            prompt=self._prompt,
            max_tool_calls=max_tool_calls,
            tracker=tracker,
            ask_approval=interrupt,
        )
        self._graph = build_agent(deps, checkpointer=InMemorySaver())
        self._waiting: set[str] = set()

    def start(
        self, request: str, profile: StudentProfile, *, thread_id: str | None = None
    ) -> AgentRun:
        """Run from a student's request until it finishes or pauses for approval.

        Raises:
            ModelError: If the model fails after its retries.
        """
        thread = thread_id or uuid4().hex
        initial: AgentState = {
            "request": request,
            "profile": profile,
            "last_response": None,
            "pending_outputs": [],
            "tool_results": [],
            "tool_calls_made": 0,
            "model_calls_made": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "approved": None,
            "saved_ids": [],
        }
        with bound_contextvars(thread_id=thread, prompt_id=self._prompt.id):
            # The request's length only: its text may hold personal details.
            log.info(
                "run_started",
                request_chars=len(request),
                max_tool_calls=self._max_tool_calls,
            )
            return self._invoke(thread, initial)

    def resume(self, thread_id: str, *, approved: bool) -> AgentRun:
        """Continue a paused run with the student's decision.

        Raises:
            ValueError: If no run with this id is waiting for approval.
        """
        if thread_id not in self._waiting:
            raise ValueError(f"No run is waiting for approval with id {thread_id!r}")
        with bound_contextvars(thread_id=thread_id, prompt_id=self._prompt.id):
            log.info("run_resumed", approved=approved)
            return self._invoke(thread_id, Command(resume=approved))

    def _invoke(self, thread_id: str, graph_input: Any) -> AgentRun:
        """Run the graph once, then log how the run paused, finished, or failed."""
        started = time.perf_counter()
        try:
            result = self._graph.invoke(graph_input, config=self._config(thread_id))
        except Exception as error:
            # The node that failed has already logged its step; this closes the run.
            log.error(
                "run_failed",
                error_type=type(error).__name__,
                elapsed_ms=elapsed_ms(started),
            )
            raise
        run = self._to_run(thread_id, result)
        log.info(
            "run_paused" if run.status == "awaiting_approval" else "run_finished",
            stop_reason=run.stop_reason,
            model_calls=run.model_calls_made,
            tool_calls=run.tool_calls_made,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            saved=len(run.saved_ids),
            elapsed_ms=elapsed_ms(started),
        )
        return run

    def _config(self, thread_id: str) -> Any:
        # Worst case: one model call per tool call, plus compose and approval nodes.
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 2 * self._max_tool_calls + 8,
        }

    def _to_run(self, thread_id: str, result: dict[str, Any]) -> AgentRun:
        interrupts = result.get("__interrupt__") or []
        if interrupts:
            self._waiting.add(thread_id)
            pending = ApprovalRequest.model_validate(interrupts[0].value)
            status: Literal["awaiting_approval", "finished"] = "awaiting_approval"
        else:
            self._waiting.discard(thread_id)
            pending = None
            status = "finished"

        return AgentRun(
            thread_id=thread_id,
            status=status,
            pending_approval=pending,
            answer=result.get("answer"),
            final_answer=result["final_answer"],
            stop_reason=result["stop_reason"],
            approved=result.get("approved"),
            saved_ids=result.get("saved_ids", []),
            tool_results=result["tool_results"],
            tool_calls_made=result["tool_calls_made"],
            model_calls_made=result["model_calls_made"],
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            prompt_id=self._prompt.id,
        )
