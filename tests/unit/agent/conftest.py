"""Shared fixtures for agent tests."""

import pytest
from agent_helpers import FIXTURES

from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.models import Opportunity, StudentProfile


@pytest.fixture
def profile() -> StudentProfile:
    """The fictional demo student, Alex."""
    path = FIXTURES / "profiles" / "demo_student.json"
    return StudentProfile.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.fixture
def records() -> list[Opportunity]:
    """All six synthetic opportunities."""
    return load_opportunities(FIXTURES / "opportunities")
