"""The shared state that flows through the agent graph.

Every node receives the current state and returns only the fields it wants to
change. LangGraph merges those changes in. Most fields are replaced; fields
annotated with a reducer (``operator.add``) are appended to instead, so the
full history of tool results is kept for the UI, traces, and evaluations.

Trusted facts (the request and the student profile) are set by our code when
a run starts. The model never writes to them.
"""

import operator
from typing import Annotated, Literal, TypedDict

from job_search_agent.agent.answer import AgentAnswer
from job_search_agent.domain.models import StudentProfile
from job_search_agent.llm.schemas import ModelResponse, ToolOutput
from job_search_agent.tools.search_tool import ToolResult

StopReason = Literal["completed", "tool_limit", "invalid_answer"]


class AgentState(TypedDict, total=False):
    """Everything the agent knows during one run."""

    # Set once by our code when the run starts.
    request: str
    profile: StudentProfile

    # Updated as the conversation with the model progresses.
    last_response: ModelResponse | None
    pending_outputs: list[ToolOutput]  # results to send on the next model turn
    tool_results: Annotated[list[ToolResult], operator.add]  # appended, never lost
    tool_calls_made: int
    model_calls_made: int

    # Set by the compose or finish node.
    answer: AgentAnswer | None  # the validated, grounded structured answer
    final_answer: str  # short text for display: the answer's summary or a fallback
    stop_reason: StopReason

    # Set by the approval and save nodes.
    approved: bool | None  # the student's decision; None until asked
    saved_ids: list[str]  # opportunities written to the tracker
