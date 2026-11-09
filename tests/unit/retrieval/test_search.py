"""Unit tests for the deterministic keyword retriever.

All tests run against the six synthetic fixtures in tests/fixtures/opportunities.
Each fixture exists to prove one behavior (title weighting, type filtering,
ties, no match), so a failing test points to the behavior that broke.
"""

from pathlib import Path

import pytest

from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.models import Opportunity, OpportunityType
from job_search_agent.retrieval.schemas import SearchQuery
from job_search_agent.retrieval.search import search_opportunities

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "opportunities"


@pytest.fixture
def records() -> list[Opportunity]:
    """Load the six synthetic opportunities once per test."""
    return load_opportunities(FIXTURE_DIR)


def _ids(query: SearchQuery, records: list[Opportunity]) -> list[str]:
    """Run a search and return only the result ids, in ranked order."""
    return [result.opportunity_id for result in search_opportunities(query, records)]


@pytest.mark.parametrize(
    ("text", "opportunity_type", "expected_ids"),
    [
        # Title + description match (opp-003) beats description-only matches;
        # equal scores are ordered by id.
        ("python", None, ["opp-003", "opp-001", "opp-002", "opp-006"]),
        # Type filter removes opp-003 because it is a job, not an internship.
        ("python", OpportunityType.INTERNSHIP, ["opp-001", "opp-002", "opp-006"]),
        # Exact tie between opp-001 and opp-006; id decides the order.
        ("software engineering", None, ["opp-001", "opp-006"]),
        # Only the research role mentions machine learning.
        ("machine learning", None, ["opp-004"]),
    ],
)
def test_search_returns_expected_ids(
    records: list[Opportunity],
    text: str,
    opportunity_type: OpportunityType | None,
    expected_ids: list[str],
) -> None:
    """Known queries return the expected ids in the expected order."""
    query = SearchQuery(text=text, opportunity_type=opportunity_type)

    assert _ids(query, records) == expected_ids


def test_title_match_scores_higher_than_description_match(
    records: list[Opportunity],
) -> None:
    """A word in the title is worth more than the same word in the description."""
    results = search_opportunities(SearchQuery(text="python"), records)
    scores = {result.opportunity_id: result.score for result in results}

    assert scores["opp-003"] == 3.0  # title (2) + description (1)
    assert scores["opp-001"] == 1.0  # description only
    assert scores["opp-003"] > scores["opp-001"]


def test_matched_terms_explain_the_match(records: list[Opportunity]) -> None:
    """Each result lists the query words that caused it to match."""
    results = search_opportunities(SearchQuery(text="Machine Learning"), records)

    assert results[0].opportunity_id == "opp-004"
    assert results[0].matched_terms == ["learning", "machine"]


def test_limit_caps_number_of_results(records: list[Opportunity]) -> None:
    """Only the top `limit` results are returned."""
    query = SearchQuery(text="python", limit=2)

    assert _ids(query, records) == ["opp-003", "opp-001"]


def test_no_match_returns_empty_list(records: list[Opportunity]) -> None:
    """A query that matches nothing returns an empty list, not an error."""
    assert search_opportunities(SearchQuery(text="quantum"), records) == []


def test_search_does_not_modify_input(records: list[Opportunity]) -> None:
    """Searching never reorders or changes the records passed in."""
    before = list(records)

    search_opportunities(SearchQuery(text="python"), records)

    assert records == before
