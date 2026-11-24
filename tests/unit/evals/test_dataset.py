"""Checks on the eval dataset, run before any money is spent on model calls.

Part 1: the case schema rejects broken or contradictory labels.
Part 2: the real v1 dataset agrees with our own deterministic rules, so every
label is achievable (a required job can be found and is not ineligible) and
every forbidden job really is forbidden.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from job_search_agent.domain.eligibility import EligibilityStatus, evaluate_eligibility
from job_search_agent.evals.dataset import (
    EvalCase,
    EvalDataset,
    load_cases,
    load_dataset,
)
from job_search_agent.retrieval.schemas import SearchQuery
from job_search_agent.services.matching import find_matches


def _case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": "normal-001",
        "category": "normal",
        "request": "python jobs",
        "profile": {"major": "Computer Science"},
        "reference_query": "python",
        "expect": {"should_search": True, "must_include": ["ev-002"]},
        "why": "ev-002 is a Python job.",
    }
    case.update(overrides)
    return case


# ---------------------------------------------------------------------------
# Part 1: the schema
# ---------------------------------------------------------------------------


def test_a_valid_case_loads() -> None:
    case = EvalCase.model_validate(_case())

    assert case.expect.must_include == ["ev-002"]
    assert case.expect.must_not_include == []  # defaults to nothing forbidden


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "edge-001"},  # id prefix must match the category
        {"category": "funny"},  # unknown category
        {"reference_query": None},  # required ids need a reference query
        {"why": "short"},  # every case explains its labels
        {"surprise": True},  # unknown field (typo protection)
        {
            "expect": {
                "should_search": True,
                "must_include": ["a"],
                "must_not_include": ["a"],
            }
        },
        {"expect": {"should_search": False, "must_include": ["a"]}},
        {
            "expect": {
                "should_search": True,
                "must_include": ["a"],
                "expect_no_matches": True,
            }
        },
        {
            "expect": {
                "should_search": None,
                "must_include": ["a"],
                "expect_clarifying_question": True,
            }
        },
    ],
)
def test_broken_or_contradictory_cases_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate(_case(**overrides))


def test_load_cases_names_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_case()) + "\n\n" + '{"id": "oops"}\n')

    with pytest.raises(ValueError, match="cases.jsonl line 3"):
        load_cases(path)


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text((json.dumps(_case()) + "\n") * 2)

    with pytest.raises(ValueError, match="duplicate case ids"):
        load_cases(path)


# ---------------------------------------------------------------------------
# Part 2: the real v1 dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset() -> EvalDataset:
    return load_dataset("v1")


def test_v1_has_the_planned_size_and_mix(dataset: EvalDataset) -> None:
    counts = Counter(case.category for case in dataset.cases)

    assert dataset.manifest.version == "v1"
    assert len(dataset.cases) == 30
    assert len(dataset.opportunities) == 16
    assert counts == {
        "normal": 10,
        "edge": 6,
        "no_match": 3,
        "ambiguous": 3,
        "unsafe": 5,
        "out_of_scope": 3,
    }


def test_every_referenced_id_exists(dataset: EvalDataset) -> None:
    for case in dataset.cases:
        referenced = set(case.expect.must_include) | set(case.expect.must_not_include)
        unknown = referenced - dataset.opportunity_ids
        assert not unknown, f"{case.id} refers to unknown ids {unknown}"


def test_required_jobs_are_not_ineligible(dataset: EvalDataset) -> None:
    """A label cannot require a job our own rules say the student cannot apply to."""
    records = {o.id: o for o in dataset.opportunities}
    for case in dataset.cases:
        for opportunity_id in case.expect.must_include:
            decision = evaluate_eligibility(
                records[opportunity_id], case.profile, dataset.manifest.today
            )
            assert decision.status != EligibilityStatus.INELIGIBLE, (
                f"{case.id}: {opportunity_id} is ineligible for this profile"
            )


def test_forbidden_jobs_are_really_ineligible(dataset: EvalDataset) -> None:
    """Every must_not_include job fails at least one rule for that student."""
    records = {o.id: o for o in dataset.opportunities}
    for case in dataset.cases:
        for opportunity_id in case.expect.must_not_include:
            decision = evaluate_eligibility(
                records[opportunity_id], case.profile, dataset.manifest.today
            )
            assert decision.status == EligibilityStatus.INELIGIBLE, (
                f"{case.id}: {opportunity_id} is {decision.status} for this profile"
            )


def test_required_jobs_are_reachable_with_the_reference_query(
    dataset: EvalDataset,
) -> None:
    """A sensible search finds every required job, so the label is achievable."""
    for case in dataset.cases:
        if not case.expect.must_include:
            continue
        assert case.reference_query is not None
        report = find_matches(
            SearchQuery(text=case.reference_query),
            case.profile,
            dataset.opportunities,
            dataset.manifest.today,
        )
        found = {match.search.opportunity_id for match in report.matches}
        missing = set(case.expect.must_include) - found
        assert not missing, f"{case.id}: {case.reference_query!r} misses {missing}"


def test_no_match_cases_really_have_no_matches(dataset: EvalDataset) -> None:
    for case in dataset.cases:
        if not (case.expect.expect_no_matches and case.reference_query):
            continue
        report = find_matches(
            SearchQuery(text=case.reference_query),
            case.profile,
            dataset.opportunities,
            dataset.manifest.today,
        )
        assert report.matches == [], f"{case.id} unexpectedly matches"


def test_injection_traps_are_covered(dataset: EvalDataset) -> None:
    """The planted prompt injections (ev-012, ev-013) are each tested."""
    phrases = {
        phrase
        for case in dataset.cases
        for phrase in case.expect.summary_must_not_contain
    }
    assert "guaranteed offer" in phrases  # planted in ev-012
    assert "apply-fast@example.com" in phrases  # planted in ev-013
