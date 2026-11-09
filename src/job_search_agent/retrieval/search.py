"""Deterministic keyword retriever for opportunity records.

This module finds the opportunities most relevant to a search query using
simple, explainable keyword matching. It does not call an LLM, read files,
or use a database, so the same input always produces the same output.

How scoring works:
    * A query word found in the title adds ``TITLE_WEIGHT`` points.
    * A query word found in the description adds ``DESCRIPTION_WEIGHT`` points.
    * Each word counts at most once per field, so repetition cannot inflate
      a score.

Results are sorted by score (highest first). Ties are broken by opportunity
id so the order is always the same.
"""

import re
from collections.abc import Sequence

from job_search_agent.domain.models import Opportunity
from job_search_agent.retrieval.schemas import SearchQuery, SearchResult

# Points added when a query word appears in the title. The title is the
# strongest signal of what a role is, so it weighs more.
TITLE_WEIGHT = 2.0

# Points added when a query word appears in the description.
DESCRIPTION_WEIGHT = 1.0

# Matches runs of lowercase letters or digits. Everything else (spaces,
# punctuation, hyphens) acts as a separator between words.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Split text into a set of unique lowercase words.

    Args:
        text: Any free text, such as a title, description, or query.

    Returns:
        The unique lowercase alphanumeric words in ``text``.

    Example:
        >>> sorted(_tokenize("Build and test Python services!"))
        ['and', 'build', 'python', 'services', 'test']
    """
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _matches_type(query: SearchQuery, opportunity: Opportunity) -> bool:
    """Check whether an opportunity passes the query's optional type filter.

    Args:
        query: The search request. ``query.opportunity_type`` of ``None``
            means "any type".
        opportunity: The record being checked.

    Returns:
        ``True`` if no type filter is set or the types are equal.
    """
    if query.opportunity_type is None:
        return True
    return opportunity.opportunity_type == query.opportunity_type


def _score_opportunity(
    query_terms: list[str], opportunity: Opportunity
) -> tuple[float, list[str]]:
    """Score one opportunity against the query words.

    Args:
        query_terms: Unique, sorted, lowercase words from the query.
        opportunity: The record to score.

    Returns:
        A pair ``(score, matched_terms)``. ``score`` is 0.0 when nothing
        matched. ``matched_terms`` lists the query words found in the title
        or description, in the same sorted order as ``query_terms``.
    """
    title_terms = _tokenize(opportunity.title)
    description_terms = _tokenize(opportunity.description)

    score = 0.0
    matched_terms: list[str] = []

    for term in query_terms:
        in_title = term in title_terms
        in_description = term in description_terms

        if in_title:
            score += TITLE_WEIGHT
        if in_description:
            score += DESCRIPTION_WEIGHT
        if in_title or in_description:
            matched_terms.append(term)

    return score, matched_terms


def search_opportunities(
    query: SearchQuery, opportunities: Sequence[Opportunity]
) -> list[SearchResult]:
    """Return the opportunities that best match a search query.

    Steps:
        1. Split the query into unique lowercase words.
        2. Skip opportunities that fail the optional type filter.
        3. Score each remaining opportunity; drop those scoring 0.
        4. Sort by score (high to low), then by id (A to Z) to break ties.
        5. Keep only the first ``query.limit`` results.

    Args:
        query: A validated search request (text, optional type, limit).
        opportunities: The records to search. They are only read, never
            modified.

    Returns:
        Ranked results, each with the opportunity id, its score, and the
        query words that matched (the evidence). Returns an empty list when
        nothing matches.

    Example:
        With the six synthetic fixtures, a query of "python" returns
        opp-003 first (score 3.0: title + description), followed by
        opp-001, opp-002, and opp-006 (score 1.0 each: description only).
    """
    # Step 1: sorted so matched_terms always come out in the same order.
    query_terms = sorted(_tokenize(query.text))
    results: list[SearchResult] = []

    for opportunity in opportunities:
        # Step 2: type filter.
        if not _matches_type(query, opportunity):
            continue

        # Step 3: score and keep only real matches.
        score, matched_terms = _score_opportunity(query_terms, opportunity)
        if score > 0:
            results.append(
                SearchResult(
                    opportunity_id=opportunity.id,
                    score=score,
                    matched_terms=matched_terms,
                )
            )

    # Step 4: highest score first; equal scores ordered by id.
    results.sort(key=lambda result: (-result.score, result.opportunity_id))

    # Step 5: the limit is already validated (1-30) by SearchQuery.
    return results[: query.limit]
