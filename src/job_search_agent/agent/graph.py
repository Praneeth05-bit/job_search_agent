"""Wire the agent's nodes into a LangGraph state graph and run it.

Written against LangGraph 1.2.x (``langgraph==1.2.11`` is pinned). API checked
against the LangGraph StateGraph reference on 2026-11-10:
``StateGraph(state_schema)``, ``add_node``, ``add_edge``,
``add_conditional_edges(source, path, path_map)``, ``compile()``, and
``START`` / ``END`` from ``langgraph.graph``.

Graph:
    START -> call_model -(router)-> run_tools -> call_model ...
                        \\-> finish -> END
"""

from collections.abc import Sequence
from datetime import date
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from job_search_agent.agent.nodes import AgentDependencies, AgentNodes
from job_search_agent.agent.state import AgentState, StopReason
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.llm.client import ModelClient
from job_search_agent.prompts.loader import Prompt, load_prompt
from job_search_agent.tools.search_tool import ToolResult

PROMPT_NAME = "search_agent"
PROMPT_VERSION = 1


class AgentRun(BaseModel):
    """The outcome of one agent run, for the API, UI, traces, and evals."""

    model_config = ConfigDict(frozen=True)

    final_answer: str
    stop_reason: StopReason
    tool_results: list[ToolResult]
    tool_calls_made: int
    model_calls_made: int
    prompt_id: str


def build_agent(deps: AgentDependencies) -> Any:
    """Build and compile the agent graph for one set of dependencies."""
    nodes = AgentNodes(deps)

    graph = StateGraph(AgentState)
    graph.add_node("call_model", nodes.call_model)
    graph.add_node("run_tools", nodes.run_tools)
    graph.add_node("finish", nodes.finish)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model",
        nodes.route_after_model,
        {"run_tools": "run_tools", "finish": "finish"},
    )
    graph.add_edge("run_tools", "call_model")
    graph.add_edge("finish", END)

    return graph.compile()


def run_agent(
    request: str,
    profile: StudentProfile,
    *,
    model: ModelClient,
    opportunities: Sequence[Opportunity],
    today: date,
    max_tool_calls: int,
    prompt: Prompt | None = None,
) -> AgentRun:
    """Run the agent once from a student's request to a final answer.

    Raises:
        ModelError: If the model fails after its retries; callers decide how
            to present that to the user.
    """
    active_prompt = prompt or load_prompt(PROMPT_NAME, PROMPT_VERSION)
    deps = AgentDependencies(
        model=model,
        opportunities=opportunities,
        today=today,
        prompt=active_prompt,
        max_tool_calls=max_tool_calls,
    )
    agent = build_agent(deps)

    initial: AgentState = {
        "request": request,
        "profile": profile,
        "last_response": None,
        "pending_outputs": [],
        "tool_results": [],
        "tool_calls_made": 0,
        "model_calls_made": 0,
    }
    # Worst case: one model call per tool call, plus the first call and finish.
    recursion_limit = 2 * max_tool_calls + 5
    final = agent.invoke(initial, config={"recursion_limit": recursion_limit})

    return AgentRun(
        final_answer=final["final_answer"],
        stop_reason=final["stop_reason"],
        tool_results=final["tool_results"],
        tool_calls_made=final["tool_calls_made"],
        model_calls_made=final["model_calls_made"],
        prompt_id=active_prompt.id,
    )
