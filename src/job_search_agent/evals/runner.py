"""Run eval cases through the real agent and collect scored results.

Every case gets a fresh agent and a fresh model client, so no case can
affect another. The agent runs on the dataset's own jobs and fixed date. A
run that pauses for approval is simply left paused: evals score the
recommendations, and nothing is ever saved (the tracker is in memory).
"""

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from job_search_agent.agent.graph import SearchAgent
from job_search_agent.evals.dataset import EvalCase, EvalDataset
from job_search_agent.evals.scoring import CaseResult, EvalSummary, score_case
from job_search_agent.llm.client import ModelClient
from job_search_agent.observability.tracing import elapsed_ms, get_logger
from job_search_agent.storage.tracker import InMemoryTracker

log = get_logger(__name__)

ModelFactory = Callable[[], ModelClient]
ProgressCallback = Callable[[int, int, CaseResult], None]


class EvalReport(BaseModel):
    """Everything about one eval run: what was run, and how it scored."""

    model_config = ConfigDict(frozen=True)

    dataset_version: str
    dataset_today: str
    model: str
    prompt_id: str
    started_at: datetime
    repeat: int
    price_input_per_million: float
    price_output_per_million: float
    summary: EvalSummary
    results: list[CaseResult]


def select_cases(
    cases: Sequence[EvalCase],
    *,
    ids: Sequence[str] = (),
    categories: Sequence[str] = (),
    limit: int | None = None,
) -> list[EvalCase]:
    """Pick the cases to run, keeping dataset order.

    Raises:
        ValueError: An id or category matches nothing (a likely typo).
    """
    known_ids = {case.id for case in cases}
    known_categories = {case.category for case in cases}
    unknown = [i for i in ids if i not in known_ids]
    unknown += [c for c in categories if c not in known_categories]
    if unknown:
        raise ValueError(f"unknown case ids or categories: {unknown}")

    chosen = [
        case
        for case in cases
        if (not ids or case.id in ids)
        and (not categories or case.category in categories)
    ]
    return chosen[:limit] if limit is not None else chosen


def run_case(
    case: EvalCase,
    dataset: EvalDataset,
    model: ModelClient,
    *,
    max_tool_calls: int,
    attempt: int = 1,
) -> CaseResult:
    """Run one case once and score it. Any exception becomes an error result."""
    agent = SearchAgent(
        model=model,
        opportunities=dataset.opportunities,
        today=dataset.manifest.today,
        max_tool_calls=max_tool_calls,
        tracker=InMemoryTracker(),
    )
    started = time.perf_counter()
    try:
        run = agent.start(case.request, case.profile)
    except Exception as error:  # recorded, not raised: one outage must not stop the run
        log.warning("eval_case_error", case_id=case.id, error_type=type(error).__name__)
        return score_case(
            case, None, latency_ms=elapsed_ms(started), attempt=attempt, error=error
        )
    return score_case(case, run, latency_ms=elapsed_ms(started), attempt=attempt)


def run_evals(
    dataset: EvalDataset,
    cases: Sequence[EvalCase],
    model_factory: ModelFactory,
    *,
    max_tool_calls: int,
    repeat: int = 1,
    pause_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    on_result: ProgressCallback | None = None,
) -> list[CaseResult]:
    """Run each case ``repeat`` times, pausing between runs to respect rate limits.

    Args:
        model_factory: Makes a fresh model client for every run.
        pause_seconds: Wait between runs (free-tier requests-per-minute limits).
        sleep: Injected so tests do not really wait.
        on_result: Called after each run with (number, total, result).
    """
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    plan = [(case, attempt) for case in cases for attempt in range(1, repeat + 1)]
    results: list[CaseResult] = []
    for number, (case, attempt) in enumerate(plan, 1):
        if number > 1 and pause_seconds > 0:
            sleep(pause_seconds)
        result = run_case(
            case,
            dataset,
            model_factory(),
            max_tool_calls=max_tool_calls,
            attempt=attempt,
        )
        results.append(result)
        if on_result is not None:
            on_result(number, len(plan), result)
    return results


def now_utc() -> datetime:
    """The current time in UTC (one place, so tests can replace it)."""
    return datetime.now(UTC)
