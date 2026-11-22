"""Request and response models for the HTTP API (the API's public contract).

These are separate from the agent's internal types on purpose. The agent's
``AgentRun`` holds raw tool results and model details that a client should
not depend on or see. The API returns ``AgentRunResponse``: only what a
client needs to show the result and ask for approval.
"""

from collections.abc import Mapping
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StrictBool

from job_search_agent.agent.graph import AgentRun, ApprovalRequest
from job_search_agent.agent.state import StopReason
from job_search_agent.domain.eligibility import EligibilityStatus
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.retrieval.schemas import SearchQuery

MAX_REQUEST_CHARS = 1000


class SearchRequest(BaseModel):
    """Body of a search request: what to look for and who is looking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: SearchQuery
    profile: StudentProfile


class AgentRunRequest(BaseModel):
    """Body of POST /agent/runs: the student's request and profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: str = Field(
        min_length=1,
        max_length=MAX_REQUEST_CHARS,
        description="What the student is looking for, in their own words.",
    )
    profile: StudentProfile


class ApprovalDecision(BaseModel):
    """Body of POST /agent/runs/{thread_id}/approval.

    ``StrictBool`` accepts only JSON true or false. Without it, Pydantic
    would also turn strings like "yes" or "1" into True, and approval must
    never happen by accident.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: StrictBool


class MatchView(BaseModel):
    """One recommendation, with the record details a UI needs to show it."""

    model_config = ConfigDict(frozen=True)

    rank: int
    opportunity_id: str
    title: str
    organization: str
    deadline: date | None
    source_url: HttpUrl
    eligibility: EligibilityStatus
    why: str
    caveats: list[str]


class RunUsage(BaseModel):
    """How much work the run took: for cost and latency reporting."""

    model_config = ConfigDict(frozen=True)

    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int


class AgentRunResponse(BaseModel):
    """The state of one agent run, as returned to clients."""

    model_config = ConfigDict(frozen=True)

    thread_id: str
    status: Literal["awaiting_approval", "finished"]
    stop_reason: StopReason
    summary: str
    clarifying_question: str | None
    matches: list[MatchView]
    pending_approval: ApprovalRequest | None
    approved: bool | None
    saved_ids: list[str]
    prompt_id: str
    usage: RunUsage

    @classmethod
    def from_run(
        cls, run: AgentRun, records: Mapping[str, Opportunity]
    ) -> "AgentRunResponse":
        """Build the public view of a run, adding each match's record details."""
        return cls(
            thread_id=run.thread_id,
            status=run.status,
            stop_reason=run.stop_reason,
            summary=run.final_answer,
            clarifying_question=run.answer.clarifying_question if run.answer else None,
            matches=_match_views(run, records),
            pending_approval=run.pending_approval,
            approved=run.approved,
            saved_ids=run.saved_ids,
            prompt_id=run.prompt_id,
            usage=RunUsage(
                model_calls=run.model_calls_made,
                tool_calls=run.tool_calls_made,
                input_tokens=run.input_tokens,
                output_tokens=run.output_tokens,
            ),
        )


class ErrorResponse(BaseModel):
    """The body of every error the API returns on purpose."""

    detail: str  # a friendly message, safe to show to the student
    error_type: str | None = None


def _match_views(run: AgentRun, records: Mapping[str, Opportunity]) -> list[MatchView]:
    """Join the answer's ranked ids with their records and eligibility results.

    The answer was already checked to cite only search results, so every id
    is known; an id is skipped only if the data changed since the search.
    """
    if run.answer is None:
        return []
    status_by_id = {
        match.search.opportunity_id: match.eligibility.status
        for result in run.tool_results
        if result.report is not None
        for match in result.report.matches
    }
    views: list[MatchView] = []
    for ranked in sorted(run.answer.matches, key=lambda m: m.rank):
        record = records.get(ranked.opportunity_id)
        if record is None:
            continue
        views.append(
            MatchView(
                rank=ranked.rank,
                opportunity_id=record.id,
                title=record.title,
                organization=record.organization,
                deadline=record.deadline,
                source_url=record.source_url,
                eligibility=status_by_id.get(record.id, EligibilityStatus.UNKNOWN),
                why=ranked.why,
                caveats=ranked.caveats,
            )
        )
    return views
