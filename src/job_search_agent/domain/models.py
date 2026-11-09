from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class OpportunityType(StrEnum):
    INTERNSHIP = "internship"
    JOB = "job"
    RESEARCH = "research"


class EligibilityRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    degree_levels: list[str] | None = None
    majors: list[str] | None = None
    minimum_gpa: float | None = Field(default=None, ge=0, le=4)
    graduation_start: date | None = None
    graduation_end: date | None = None
    work_authorizations: list[str] | None = None


class Opportunity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    opportunity_type: OpportunityType
    description: str = Field(min_length=1)
    location: str
    source_name: str = Field(min_length=1)
    source_url: HttpUrl
    posted_at: datetime
    deadline: date | None = None
    eligibility: EligibilityRequirements


class StudentProfile(BaseModel):
    """Facts about a student used to check eligibility.

    Every field is optional. A missing value (None) means "not provided",
    which eligibility rules treat as unknown rather than guessing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    degree_level: str | None = None
    major: str | None = None
    gpa: float | None = Field(default=None, ge=0, le=4)
    expected_graduation: date | None = None
    work_authorization: str | None = None
