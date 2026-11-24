"""The evaluation dataset: labeled cases that say what a good agent should do.

Tests check that the code works with a scripted model. Evals check that the
real model makes good decisions. Each ``EvalCase`` is one student request
with its correct outcome written down *before* the agent is run.

A dataset version is a frozen folder::

    evals/v1/
        manifest.json         version, the fixed "today" date, description
        cases.jsonl           one EvalCase per line
        opportunities/*.json  the job records the cases refer to

Once a version has been used for a reported score it never changes; new or
corrected cases go into a new version, so scores stay comparable.
"""

import json
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from job_search_agent.config import PROJECT_ROOT
from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.models import Opportunity, StudentProfile

EVALS_DIR = PROJECT_ROOT / "evals"

Category = Literal["normal", "edge", "no_match", "ambiguous", "unsafe", "out_of_scope"]


class Expectations(BaseModel):
    """What a correct run looks like. Each field becomes one metric in step 21."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    should_search: bool | None = Field(
        description="True: must call the search tool. False: must not. "
        "None: either is acceptable (not scored)."
    )
    must_include: list[str] = Field(
        default_factory=list,
        description="Opportunity ids a good answer recommends.",
    )
    must_not_include: list[str] = Field(
        default_factory=list,
        description="Opportunity ids that must never be recommended.",
    )
    expect_no_matches: bool = Field(
        default=False, description="A good answer recommends nothing."
    )
    expect_clarifying_question: bool = Field(
        default=False, description="A good answer asks a clarifying question."
    )
    summary_must_not_contain: list[str] = Field(
        default_factory=list,
        description="Phrases that must not appear anywhere in the answer text "
        "(checked case-insensitively), such as text planted by a prompt injection.",
    )

    @model_validator(mode="after")
    def labels_agree(self) -> Self:
        """Reject contradictory labels, such as an id both required and forbidden."""
        overlap = set(self.must_include) & set(self.must_not_include)
        if overlap:
            raise ValueError(f"ids both required and forbidden: {sorted(overlap)}")
        if self.must_include:
            if self.expect_no_matches:
                raise ValueError("expect_no_matches conflicts with must_include")
            if self.should_search is False:
                raise ValueError("should_search=False conflicts with must_include")
            if self.expect_clarifying_question:
                raise ValueError("a clarifying question conflicts with must_include")
        return self


class EvalCase(BaseModel):
    """One labeled request: the input, the expected outcome, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z_]+-\d{3}$")
    category: Category
    request: str = Field(min_length=1, max_length=1000)
    profile: StudentProfile
    expect: Expectations
    reference_query: str | None = Field(
        default=None,
        description="A search a sensible agent could run. Used only to check "
        "that the labels are reachable, never sent to the agent.",
    )
    why: str = Field(min_length=10, description="The reasoning behind the labels.")

    @model_validator(mode="after")
    def id_and_reference_fit(self) -> Self:
        """The id starts with the category; required ids need a reference query."""
        if not self.id.startswith(f"{self.category}-"):
            raise ValueError(f"id {self.id!r} must start with '{self.category}-'")
        if self.expect.must_include and self.reference_query is None:
            raise ValueError("cases with must_include need a reference_query")
        return self


class Manifest(BaseModel):
    """Facts about one dataset version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(pattern=r"^v\d+$")
    today: date = Field(description="The fixed date every case is evaluated on.")
    description: str


class EvalDataset(BaseModel):
    """A loaded dataset version: manifest, opportunities, and cases."""

    model_config = ConfigDict(frozen=True)

    manifest: Manifest
    opportunities: list[Opportunity]
    cases: list[EvalCase]

    @property
    def opportunity_ids(self) -> set[str]:
        return {opportunity.id for opportunity in self.opportunities}


def load_cases(path: Path) -> list[EvalCase]:
    """Read one EvalCase per non-empty line of a JSONL file.

    Raises:
        ValueError: A line is not a valid case (the message names the line),
            or two cases share an id.
    """
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate(json.loads(line)))
        except ValueError as error:  # JSON errors and validation errors
            raise ValueError(f"{path.name} line {number}: {error}") from error
    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate case ids: {duplicates}")
    return cases


def load_dataset(version: str = "v1", root: Path = EVALS_DIR) -> EvalDataset:
    """Load a frozen dataset version from ``root/<version>/``."""
    folder = root / version
    manifest = Manifest.model_validate_json(
        (folder / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.version != version:
        raise ValueError(f"{folder} holds version {manifest.version}, not {version}")
    return EvalDataset(
        manifest=manifest,
        opportunities=load_opportunities(folder / "opportunities"),
        cases=load_cases(folder / "cases.jsonl"),
    )
