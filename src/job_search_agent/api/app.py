"""HTTP API for SpartanScout.

This layer is deliberately thin: it receives a request, calls a service or
the agent, and returns the result. It contains no business rules. Request
and response models live in ``schemas.py``; shared objects (data, date,
agent) come from ``dependencies.py`` and can be replaced in tests.

Endpoints:
    GET  /health                          liveness check
    POST /search                          deterministic search + eligibility
    POST /agent/runs                      start an agent run
    POST /agent/runs/{thread_id}/approval approve or reject the shortlist

Endpoints are plain ``def`` (not ``async def``) on purpose: a model call
blocks for seconds, and FastAPI runs ``def`` endpoints in a thread pool, so
one slow run never freezes the server for everyone else.

Run locally:
    uv run uvicorn job_search_agent.api.app:app --reload
Then open http://localhost:8000/docs for interactive documentation.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Path, Request, status
from fastapi.responses import JSONResponse

from job_search_agent.agent.graph import SearchAgent
from job_search_agent.api.dependencies import (
    get_agent,
    get_opportunities,
    get_records,
    get_today,
)
from job_search_agent.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    ApprovalDecision,
    ErrorResponse,
    SearchRequest,
)
from job_search_agent.config import get_settings
from job_search_agent.domain.models import Opportunity
from job_search_agent.llm.errors import ModelAuthError, ModelError
from job_search_agent.observability.tracing import configure_logging, get_logger
from job_search_agent.services.matching import MatchReport, find_matches

log = get_logger(__name__)

NO_WAITING_RUN = "No run is waiting for approval with this id."
RETRY_AFTER_SECONDS = 30


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Set up logging when the server starts (not when tests import the app)."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    yield


app = FastAPI(
    title="SpartanScout",
    description="Evidence-grounded campus opportunity search.",
    version="0.2.0",
    lifespan=lifespan,
)

# Documented in /docs so clients know these error shapes exist.
MODEL_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
}


# ---------------------------------------------------------------------------
# Errors: model failures become friendly HTTP responses
# ---------------------------------------------------------------------------


def status_for(error: ModelError) -> int:
    """Pick the HTTP status code for a model failure.

    500: our configuration is wrong (bad API key). Not the client's fault.
    503: temporary (timeout, rate limit, overload). Retrying later can work.
    502: the model answered with something unusable.
    """
    if isinstance(error, ModelAuthError):
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    if error.retryable:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_502_BAD_GATEWAY


@app.exception_handler(ModelError)
async def handle_model_error(request: Request, error: ModelError) -> JSONResponse:
    """Return the error's friendly message; technical details stay in the logs."""
    code = status_for(error)
    log.warning(
        "api_model_error",
        path=request.url.path,
        status_code=code,
        error_type=type(error).__name__,
    )
    headers = {"Retry-After": str(RETRY_AFTER_SECONDS)} if code == 503 else None
    body = ErrorResponse(detail=error.user_message, error_type=type(error).__name__)
    return JSONResponse(status_code=code, content=body.model_dump(), headers=headers)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is running. Used by Docker and monitoring."""
    return {"status": "ok"}


@app.post("/search")
def search(
    request: SearchRequest,
    opportunities: Annotated[list[Opportunity], Depends(get_opportunities)],
    today: Annotated[date, Depends(get_today)],
) -> MatchReport:
    """Find relevant opportunities and check the student's eligibility.

    Returns eligible or unknown results in ``matches`` and ineligible
    results, with their reasons, in ``excluded``.
    """
    return find_matches(request.query, request.profile, opportunities, today)


@app.post(
    "/agent/runs",
    status_code=status.HTTP_201_CREATED,
    responses=MODEL_ERRORS,
)
def start_agent_run(
    body: AgentRunRequest,
    agent: Annotated[SearchAgent, Depends(get_agent)],
    records: Annotated[dict[str, Opportunity], Depends(get_records)],
) -> AgentRunResponse:
    """Start an agent run for the student's request.

    The run either finishes, or pauses with ``status="awaiting_approval"``
    and a ``pending_approval`` proposal. Send the decision to
    ``/agent/runs/{thread_id}/approval``. Nothing is saved before that.
    """
    run = agent.start(body.request, body.profile)
    return AgentRunResponse.from_run(run, records)


@app.post(
    "/agent/runs/{thread_id}/approval",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}, **MODEL_ERRORS},
)
def decide_approval(
    thread_id: Annotated[str, Path(min_length=1, max_length=64)],
    body: ApprovalDecision,
    agent: Annotated[SearchAgent, Depends(get_agent)],
    records: Annotated[dict[str, Opportunity], Depends(get_records)],
) -> AgentRunResponse:
    """Approve or reject the shortlist of a paused run, then finish it.

    Returns 404 if no run with this id is waiting (unknown id, already
    decided, or lost because the server restarted).
    """
    try:
        run = agent.resume(thread_id, approved=body.approved)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=NO_WAITING_RUN
        ) from None
    return AgentRunResponse.from_run(run, records)
