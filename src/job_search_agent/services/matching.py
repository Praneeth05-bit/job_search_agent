"""Matching service: connects search and eligibility into one answer.

This is the single entry point that the API, the UI, and the agent's search
tool all call, so they share exactly the same behavior. It contains no
business rules of its own; it only runs the retriever, applies the
eligibility rules to each result, and sorts the results into two groups:

    matches:  eligible or unknown, passed on for ranking and explanation
    excluded: ineligible, shown to the user with reasons but never ranked

Known limitation: the search limit is applied before eligibility filtering,
so asking for 10 results can return fewer than 10 matches.
"""

from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel, ConfigDict

from job_search_agent.domain.eligibility import (
    EligibilityDecision,
    EligibilityStatus,
    evaluate_eligibility,
)
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.retrieval.schemas import SearchQuery, SearchResult
from job_search_agent.retrieval.search import search_opportunities


class Match(BaseModel):
    """One opportunity with both pieces of evidence: relevance and eligibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    search: SearchResult
    eligibility: EligibilityDecision


class MatchReport(BaseModel):
    """The full answer to one search, split into usable and excluded results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matches: list[Match]
    excluded: list[Match]


def find_matches(
    query: SearchQuery,
    profile: StudentProfile,
    opportunities: Sequence[Opportunity],
    today: date,
) -> MatchReport:
    """Search for relevant opportunities and check eligibility for each one.

    Steps:
        1. Retrieve and rank relevant opportunities.
        2. Look up each result's full record by id.
        3. Evaluate eligibility for that record.
        4. Put ineligible results in ``excluded``; everything else in
           ``matches``. Both lists keep the search ranking order.

    Args:
        query: The validated search request.
        profile: The student's facts. Missing facts become UNKNOWN.
        opportunities: The records to search.
        today: The date used for deadline checks.

    Returns:
        A MatchReport with matches (eligible or unknown) and excluded
        (ineligible) results, each carrying its full evidence.
    """
    results = search_opportunities(query, opportunities)
    records_by_id = {opportunity.id: opportunity for opportunity in opportunities}

    matches: list[Match] = []
    excluded: list[Match] = []

    for result in results:
        record = records_by_id[result.opportunity_id]
        decision = evaluate_eligibility(record, profile, today)
        match = Match(search=result, eligibility=decision)

        if decision.status is EligibilityStatus.INELIGIBLE:
            excluded.append(match)
        else:
            matches.append(match)

    return MatchReport(matches=matches, excluded=excluded)
