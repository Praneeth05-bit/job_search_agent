"""HTTP API for SpartanScout.

This layer is deliberately thin: it receives a request, calls the matching
service, and returns the result. It contains no business rules. FastAPI
validates every request with the same Pydantic models the rest of the
project uses, so bad input is rejected with a 422 before any logic runs.

Run locally:
    uv run uvicorn job_search_agent.api.app:app --reload
Then open http://localhost:8000/docs for interactive documentation.
"""

from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, ConfigDict

from job_search_agent.config import Settings, get_settings
from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.retrieval.schemas import SearchQuery
from job_search_agent.services.matching import MatchReport, find_matches

app = FastAPI(
    title="SpartanScout",
    description="Evidence-grounded campus opportunity search.",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    """Body of a search request: what to look for and who is looking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: SearchQuery
    profile: StudentProfile


# ---------------------------------------------------------------------------
# Dependencies (replaceable in tests via app.dependency_overrides)
# ---------------------------------------------------------------------------


def get_opportunities(
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[Opportunity]:
    """Load opportunity records from the configured data folder."""
    return load_opportunities(settings.data_dir)


def get_today() -> date:
    """Return today's date. This is the only place the real clock is read."""
    return date.today()


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
