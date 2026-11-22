"""Shared objects the API endpoints receive through ``Depends``.

Tests replace any of these with ``app.dependency_overrides``, so endpoints
never need to know whether they talk to real data, a real model, or fakes.
"""

from datetime import date
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from job_search_agent.agent.graph import SearchAgent
from job_search_agent.config import Settings, get_settings
from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.models import Opportunity
from job_search_agent.llm.client import ModelClient
from job_search_agent.llm.demo import DemoModelClient
from job_search_agent.llm.gemini import create_gemini_client
from job_search_agent.storage.tracker import JsonFileTracker


def get_opportunities(
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[Opportunity]:
    """Load opportunity records from the configured data folder."""
    return load_opportunities(settings.data_dir)


def get_records(
    opportunities: Annotated[list[Opportunity], Depends(get_opportunities)],
) -> dict[str, Opportunity]:
    """The same records, looked up by id."""
    return {opportunity.id: opportunity for opportunity in opportunities}


def get_today() -> date:
    """Return today's date. This is the only place the real clock is read."""
    return date.today()


def get_agent(today: Annotated[date, Depends(get_today)]) -> SearchAgent:
    """The one agent shared by every request on this date.

    It must be shared: a paused run lives in the agent's in-memory
    checkpointer, so the approval request has to reach the same agent that
    started the run. Limits (documented in the README): one server process
    only, paused runs are lost on restart, and a run paused before midnight
    must be started again the next day (deadlines are checked against today).
    """
    return _agent_for_day(today)


@lru_cache(maxsize=1)
def _agent_for_day(today: date) -> SearchAgent:
    """Build the agent once per day (a new date builds a new one)."""
    settings = get_settings()
    return SearchAgent(
        model=_model(settings),
        opportunities=load_opportunities(settings.data_dir),
        today=today,
        max_tool_calls=settings.max_tool_calls,
        tracker=JsonFileTracker(settings.tracker_path),
    )


def _model(settings: Settings) -> ModelClient:
    """The offline demo model in mock mode, otherwise Gemini with retries."""
    if settings.use_mock_model:
        return DemoModelClient()
    return create_gemini_client(settings)
