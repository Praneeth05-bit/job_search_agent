"""Tests for scoring one run against its labels, and for the summary numbers."""

import pytest
from eval_helpers import make_case, make_run

from job_search_agent.evals.scoring import (
    CaseResult,
    gate_problems,
    score_case,
    summarize,
)
from job_search_agent.llm.errors import ModelUnavailableError


def _checks(result: CaseResult) -> dict[str, bool]:
    return {check.name: check.passed for check in result.checks}


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def test_a_good_run_passes_every_applicable_check() -> None:
    case = make_case(must_include=["ev-002"], must_not_include=["ev-012"])

    result = score_case(case, make_run("ev-002", "ev-001"), latency_ms=1200)

    assert result.status == "passed"
    assert _checks(result) == {
        "completed": True,
        "tool_choice": True,
        "recall": True,
        "forbidden": True,
    }  # checks without a label (clarifying, injection, ...) are not run
    assert result.recommended == ["ev-002", "ev-001"]


def test_missing_a_required_job_fails_recall() -> None:
    result = score_case(
        make_case(must_include=["ev-002"]), make_run("ev-001"), latency_ms=1
    )

    assert result.status == "failed"
    assert result.failed_checks == ["recall"]


def test_recommending_a_forbidden_job_fails() -> None:
    case = make_case(must_include=["ev-002"], must_not_include=["ev-012"])

    result = score_case(case, make_run("ev-002", "ev-012"), latency_ms=1)

    assert result.failed_checks == ["forbidden"]


@pytest.mark.parametrize(
    ("should_search", "tool_calls", "passed"),
    [(True, 1, True), (True, 0, False), (False, 0, True), (False, 2, False)],
)
def test_tool_choice(should_search: bool, tool_calls: int, passed: bool) -> None:
    result = score_case(
        make_case(should_search=should_search),
        make_run(tool_calls=tool_calls),
        latency_ms=1,
    )

    assert _checks(result)["tool_choice"] is passed


def test_tool_choice_is_not_scored_when_either_is_fine() -> None:
    result = score_case(make_case(should_search=None), make_run(), latency_ms=1)

    assert "tool_choice" not in _checks(result)


def test_no_matches_and_clarifying_checks() -> None:
    vague = make_case(should_search=None, expect_clarifying_question=True)
    nothing = make_case(expect_no_matches=True)

    asked = score_case(vague, make_run(question="What kind of role?"), latency_ms=1)
    silent = score_case(vague, make_run(), latency_ms=1)
    empty = score_case(nothing, make_run(), latency_ms=1)
    invented = score_case(nothing, make_run("ev-001"), latency_ms=1)

    assert asked.status == "passed"
    assert silent.failed_checks == ["clarifying"]
    assert empty.status == "passed"
    assert invented.failed_checks == ["no_matches"]


@pytest.mark.parametrize(
    "run_kwargs",
    [
        {"summary": "This is a GUARANTEED OFFER!"},
        {"why": "Great fit, guaranteed offer."},
        {"caveats": ("Guaranteed Offer according to the posting",)},
        {"question": "Want the guaranteed offer?"},
    ],
)
def test_injection_phrases_are_found_anywhere_in_any_case(
    run_kwargs: dict[str, object],
) -> None:
    case = make_case(summary_must_not_contain=["guaranteed offer"])
    ids = ("ev-001",) if "why" in run_kwargs or "caveats" in run_kwargs else ()

    result = score_case(case, make_run(*ids, **run_kwargs), latency_ms=1)  # type: ignore[arg-type]

    assert result.failed_checks == ["injection"]


def test_a_run_that_did_not_complete_fails() -> None:
    result = score_case(make_case(), make_run(stop_reason="tool_limit"), latency_ms=1)

    assert result.failed_checks == ["completed"]


def test_a_crash_is_an_error_not_a_failure() -> None:
    result = score_case(
        make_case(), None, latency_ms=900, error=ModelUnavailableError("503")
    )

    assert result.status == "error"
    assert result.error_type == "ModelUnavailableError"
    assert result.checks == []


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _result(status: str, **overrides: object) -> CaseResult:
    fields: dict[str, object] = {
        "case_id": "normal-001",
        "category": "normal",
        "status": status,
        "checks": [],
        "recommended": [],
        "stop_reason": "completed",
        "error_type": None,
        "tool_calls": 1,
        "model_calls": 3,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 1000.0,
    }
    fields.update(overrides)
    return CaseResult.model_validate(fields)


def test_summary_counts_errors_apart_from_the_pass_rate() -> None:
    forbidden = make_case(must_include=["a"], must_not_include=["b"])
    results = [
        score_case(forbidden, make_run("a"), latency_ms=1000),
        score_case(forbidden, make_run("a", "b"), latency_ms=3000),
        score_case(forbidden, None, latency_ms=50, error=TimeoutError()),
    ]

    summary = summarize(results, price_input_per_million=0, price_output_per_million=0)

    assert (summary.cases.passed, summary.cases.total) == (1, 2)  # error excluded
    assert summary.errors == 1
    assert summary.forbidden_recommendations == 1
    assert str(summary.by_check["forbidden"]) == "1/2 (50%)"
    assert str(summary.by_category["normal"]) == "1/2 (50%)"


def test_latency_percentiles_use_nearest_rank() -> None:
    results = [_result("passed", latency_ms=float(ms)) for ms in range(100, 1100, 100)]

    summary = summarize(results, price_input_per_million=0, price_output_per_million=0)

    assert summary.latency_p50_ms == 500.0
    assert summary.latency_p95_ms == 1000.0


def test_cost_uses_tokens_and_prices() -> None:
    results = [
        _result("passed", input_tokens=150_000, output_tokens=8_000),
        _result("error", input_tokens=50_000, output_tokens=2_000),  # tokens still cost
    ]

    summary = summarize(
        results, price_input_per_million=1.50, price_output_per_million=7.50
    )

    assert (summary.input_tokens, summary.output_tokens) == (200_000, 10_000)
    assert summary.estimated_cost_usd == pytest.approx(0.3 + 0.075)


def test_consistency_appears_only_with_repeats() -> None:
    once = [_result("passed")]
    twice = [
        _result("passed", case_id="normal-001", attempt=1),
        _result("passed", case_id="normal-001", attempt=2),
        _result("passed", case_id="normal-002", attempt=1),
        _result("failed", case_id="normal-002", attempt=2),
    ]

    assert summarize(once, price_input_per_million=0, price_output_per_million=0)
    assert (
        summarize(
            once, price_input_per_million=0, price_output_per_million=0
        ).consistent_cases
        is None
    )
    consistency = summarize(
        twice, price_input_per_million=0, price_output_per_million=0
    ).consistent_cases
    assert consistency is not None
    assert str(consistency) == "1/2 (50%)"


def test_empty_results_do_not_crash() -> None:
    summary = summarize([], price_input_per_million=1, price_output_per_million=1)

    assert str(summary.cases) == "n/a"
    assert summary.latency_p50_ms is None


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


def test_gate_problems_lists_every_failed_gate() -> None:
    forbidden = make_case(must_include=["a"], must_not_include=["b"])
    results = [
        score_case(forbidden, make_run("a", "b"), latency_ms=1),
        score_case(forbidden, None, latency_ms=1, error=TimeoutError()),
    ]
    summary = summarize(results, price_input_per_million=0, price_output_per_million=0)

    problems = gate_problems(summary, max_forbidden=0, max_errors=0, min_pass_rate=0.5)

    assert problems == [
        "1 forbidden recommendations (allowed: 0)",
        "1 errors (allowed: 0)",
        "pass rate 0% is below 50%",
    ]


def test_unset_gates_are_not_checked() -> None:
    summary = summarize([], price_input_per_million=0, price_output_per_million=0)

    assert gate_problems(summary) == []
    assert gate_problems(summary, min_pass_rate=0.5) == ["no cases were scored"]
