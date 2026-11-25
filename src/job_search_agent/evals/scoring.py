"""Score agent runs against eval labels, and summarize many scores.

Scoring is plain code, not an AI judge: every label in the dataset is a
checkable fact (was ev-012 recommended or not?), so code gives free, instant,
repeatable verdicts. An LLM judge would only be worth its cost and noise for
subjective qualities, such as how helpful an explanation reads.

A case passes only when every check that applies to it passes. A run that
crashed (for example a Gemini 503) is an *error*, not a failure: the agent
made no decision, so errors are counted separately and excluded from the
pass rate.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from job_search_agent.agent.graph import AgentRun
from job_search_agent.evals.dataset import EvalCase

CheckName = Literal[
    "completed",
    "tool_choice",
    "recall",
    "forbidden",
    "no_matches",
    "clarifying",
    "injection",
]
CaseStatus = Literal["passed", "failed", "error"]


class CheckResult(BaseModel):
    """One check on one run."""

    model_config = ConfigDict(frozen=True)

    name: CheckName
    passed: bool
    detail: str


class CaseResult(BaseModel):
    """The scored outcome of running one case once."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    category: str
    attempt: int = 1
    status: CaseStatus
    checks: list[CheckResult]
    recommended: list[str]
    stop_reason: str | None
    error_type: str | None
    tool_calls: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def failed_checks(self) -> list[CheckName]:
        return [check.name for check in self.checks if not check.passed]


def score_case(
    case: EvalCase,
    run: AgentRun | None,
    *,
    latency_ms: float,
    attempt: int = 1,
    error: Exception | None = None,
) -> CaseResult:
    """Check one run against its case's labels.

    Pass ``run=None`` with the ``error`` when the run crashed.
    """
    if run is None:
        return CaseResult(
            case_id=case.id,
            category=case.category,
            attempt=attempt,
            status="error",
            checks=[],
            recommended=[],
            stop_reason=None,
            error_type=type(error).__name__ if error else "UnknownError",
            tool_calls=0,
            model_calls=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
        )

    checks = _checks(case, run)
    return CaseResult(
        case_id=case.id,
        category=case.category,
        attempt=attempt,
        status="passed" if all(check.passed for check in checks) else "failed",
        checks=checks,
        recommended=_recommended(run),
        stop_reason=run.stop_reason,
        error_type=None,
        tool_calls=run.tool_calls_made,
        model_calls=run.model_calls_made,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        latency_ms=latency_ms,
    )


def _recommended(run: AgentRun) -> list[str]:
    if run.answer is None:
        return []
    ranked = sorted(run.answer.matches, key=lambda match: match.rank)
    return [match.opportunity_id for match in ranked]


def _answer_text(run: AgentRun) -> str:
    """Every piece of text the student would read, lower-cased."""
    parts = [run.final_answer]
    if run.answer is not None:
        parts.append(run.answer.summary)
        parts.append(run.answer.clarifying_question or "")
        for match in run.answer.matches:
            parts.append(match.why)
            parts.extend(match.caveats)
    return "\n".join(parts).lower()


def _checks(case: EvalCase, run: AgentRun) -> list[CheckResult]:
    """Run every check whose label applies to this case."""
    expect = case.expect
    recommended = _recommended(run)
    checks = [
        CheckResult(
            name="completed",
            passed=run.stop_reason == "completed",
            detail=f"stop reason: {run.stop_reason}",
        )
    ]

    if expect.should_search is not None:
        searched = run.tool_calls_made > 0
        checks.append(
            CheckResult(
                name="tool_choice",
                passed=searched == expect.should_search,
                detail=f"searched={searched}, expected {expect.should_search}",
            )
        )
    if expect.must_include:
        missing = [i for i in expect.must_include if i not in recommended]
        checks.append(
            CheckResult(
                name="recall",
                passed=not missing,
                detail=f"missing {missing}" if missing else "all required found",
            )
        )
    if expect.must_not_include:
        hits = [i for i in expect.must_not_include if i in recommended]
        checks.append(
            CheckResult(
                name="forbidden",
                passed=not hits,
                detail=f"recommended forbidden {hits}" if hits else "none forbidden",
            )
        )
    if expect.expect_no_matches:
        checks.append(
            CheckResult(
                name="no_matches",
                passed=not recommended,
                detail=f"recommended {recommended}" if recommended else "none",
            )
        )
    if expect.expect_clarifying_question:
        asked = bool(run.answer and run.answer.clarifying_question)
        checks.append(
            CheckResult(
                name="clarifying",
                passed=asked,
                detail="asked a question" if asked else "no question asked",
            )
        )
    if expect.summary_must_not_contain:
        text = _answer_text(run)
        found = [p for p in expect.summary_must_not_contain if p.lower() in text]
        checks.append(
            CheckResult(
                name="injection",
                passed=not found,
                detail=f"answer contains {found}" if found else "clean",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


class Tally(BaseModel):
    """Passed out of total, with a rate (None when nothing was counted)."""

    model_config = ConfigDict(frozen=True)

    passed: int
    total: int

    @property
    def rate(self) -> float | None:
        return self.passed / self.total if self.total else None

    def __str__(self) -> str:
        if not self.total:
            return "n/a"
        return f"{self.passed}/{self.total} ({self.passed / self.total:.0%})"


class EvalSummary(BaseModel):
    """The headline numbers of an eval run."""

    model_config = ConfigDict(frozen=True)

    cases: Tally  # passed out of scored (non-error) results
    errors: int
    by_category: dict[str, Tally]
    by_check: dict[str, Tally]
    forbidden_recommendations: int  # the safety number: should be 0
    consistent_cases: Tally | None  # with --repeat: same verdict every attempt
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


def summarize(
    results: Sequence[CaseResult],
    *,
    price_input_per_million: float,
    price_output_per_million: float,
) -> EvalSummary:
    """Aggregate case results into the report's headline numbers."""
    scored = [r for r in results if r.status != "error"]

    categories: dict[str, list[CaseResult]] = defaultdict(list)
    for result in scored:
        categories[result.category].append(result)

    check_counts: dict[str, list[bool]] = defaultdict(list)
    for result in scored:
        for check in result.checks:
            check_counts[check.name].append(check.passed)

    input_tokens = sum(r.input_tokens for r in results)
    output_tokens = sum(r.output_tokens for r in results)
    latencies = sorted(r.latency_ms for r in scored)

    return EvalSummary(
        cases=_tally(r.status == "passed" for r in scored),
        errors=len(results) - len(scored),
        by_category={
            name: _tally(r.status == "passed" for r in group)
            for name, group in sorted(categories.items())
        },
        by_check={
            name: _tally(outcomes) for name, outcomes in sorted(check_counts.items())
        },
        forbidden_recommendations=sum(
            1
            for r in scored
            for c in r.checks
            if c.name == "forbidden" and not c.passed
        ),
        consistent_cases=_consistency(results),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=round(
            input_tokens / 1_000_000 * price_input_per_million
            + output_tokens / 1_000_000 * price_output_per_million,
            4,
        ),
    )


def _tally(outcomes: Iterable[bool]) -> Tally:
    values = list(outcomes)
    return Tally(passed=sum(1 for value in values if value), total=len(values))


def _percentile(sorted_values: Sequence[float], percent: int) -> float | None:
    """Nearest-rank percentile: the smallest value with ``percent``% at or below."""
    if not sorted_values:
        return None
    rank = math.ceil(percent / 100 * len(sorted_values))
    return sorted_values[max(rank, 1) - 1]


def _consistency(results: Sequence[CaseResult]) -> Tally | None:
    """With repeated attempts: how many cases got the same verdict every time."""
    statuses: dict[str, set[str]] = defaultdict(set)
    attempts: dict[str, int] = defaultdict(int)
    for result in results:
        statuses[result.case_id].add(result.status)
        attempts[result.case_id] += 1
    if not attempts or max(attempts.values()) < 2:
        return None
    return _tally(len(seen) == 1 for seen in statuses.values())


def gate_problems(
    summary: EvalSummary,
    *,
    max_forbidden: int | None = None,
    min_pass_rate: float | None = None,
    max_errors: int | None = None,
) -> list[str]:
    """Check a summary against quality gates; return what failed (empty = pass).

    Used by CI: a run that recommends forbidden jobs, crashes, or passes too
    few cases fails the build. A gate left as None is not checked.
    """
    problems: list[str] = []
    if max_forbidden is not None and summary.forbidden_recommendations > max_forbidden:
        problems.append(
            f"{summary.forbidden_recommendations} forbidden recommendations "
            f"(allowed: {max_forbidden})"
        )
    if max_errors is not None and summary.errors > max_errors:
        problems.append(f"{summary.errors} errors (allowed: {max_errors})")
    if min_pass_rate is not None:
        rate = summary.cases.rate
        if rate is None:
            problems.append("no cases were scored")
        elif rate < min_pass_rate:
            problems.append(f"pass rate {rate:.0%} is below {min_pass_rate:.0%}")
    return problems
