"""Input and output contracts for opportunity retrieval."""

from pydantic import BaseModel, ConfigDict, Field

from job_search_agent.domain.models import OpportunityType


class SearchQuery(BaseModel):
    """A request to find opportunities relevant to free-text search terms."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    opportunity_type: OpportunityType | None = None
    limit: int = Field(default=10, ge=1, le=30)


class SearchResult(BaseModel):
    """One ranked match, identified by ID, with the terms that explain it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opportunity_id: str = Field(min_length=1)
    score: float = Field(ge=0)
    matched_terms: list[str]
