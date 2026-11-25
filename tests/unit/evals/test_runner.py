"""Tests for running cases, choosing cases, reports, and the command line."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import structlog

from job_search_agent.evals.cli import main
from job_search_agent.evals.dataset import EvalDataset, load_dataset
from job_search_agent.evals.report import render_markdown, report_stem, write_report
from job_search_agent.evals.runner import (
    EvalReport,
    run_case,
    run_evals,
    select_cases,
)
from job_search_agent.evals.scoring import summarize
from job_search_agent.llm.demo import DemoModelClient
from job_search_agent.llm.errors import ModelUnavailableError
from job_search_agent.llm.mock import MockModelClient


@pytest.fixture(scope="module")
def dataset() -> EvalDataset:
    return load_dataset("v1")


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """main() configures logging globally; undo it so other tests are unaffected."""
    yield
    structlog.reset_defaults()


# ---------------------------------------------------------------------------
# Choosing cases
# ---------------------------------------------------------------------------


def test_select_by_id_category_and_limit(dataset: EvalDataset) -> None:
    by_id = select_cases(dataset.cases, ids=["unsafe-002", "normal-001"])
    by_category = select_cases(dataset.cases, categories=["unsafe"])
    limited = select_cases(dataset.cases, limit=3)

    assert [c.id for c in by_id] == ["normal-001", "unsafe-002"]  # dataset order
    assert len(by_category) == 5
    assert [c.id for c in limited] == ["normal-001", "normal-002", "normal-003"]


def test_unknown_ids_are_reported(dataset: EvalDataset) -> None:
    with pytest.raises(ValueError, match="normal-999"):
        select_cases(dataset.cases, ids=["normal-999"])


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_run_case_scores_a_real_agent_run(dataset: EvalDataset) -> None:
    (case,) = select_cases(dataset.cases, ids=["normal-008"])

    result = run_case(case, dataset, DemoModelClient(), max_tool_calls=5)

    assert result.status == "passed"
    assert "ev-011" in result.recommended
    assert result.tool_calls == 1


def test_a_model_outage_becomes_an_error_result(dataset: EvalDataset) -> None:
    (case,) = select_cases(dataset.cases, ids=["normal-001"])
    model = MockModelClient([ModelUnavailableError("HTTP 503")])

    result = run_case(case, dataset, model, max_tool_calls=5)

    assert result.status == "error"
    assert result.error_type == "ModelUnavailableError"


def test_run_evals_repeats_pauses_and_reports_progress(dataset: EvalDataset) -> None:
    cases = select_cases(dataset.cases, ids=["normal-001", "edge-002"])
    waits: list[float] = []
    progress: list[tuple[int, int]] = []

    results = run_evals(
        dataset,
        cases,
        DemoModelClient,
        max_tool_calls=5,
        repeat=2,
        pause_seconds=1.5,
        sleep=waits.append,
        on_result=lambda n, total, _: progress.append((n, total)),
    )

    assert [(r.case_id, r.attempt) for r in results] == [
        ("normal-001", 1),
        ("normal-001", 2),
        ("edge-002", 1),
        ("edge-002", 2),
    ]
    assert waits == [1.5, 1.5, 1.5]  # between runs, not before the first
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_repeat_must_be_positive(dataset: EvalDataset) -> None:
    with pytest.raises(ValueError):
        run_evals(
            dataset, dataset.cases[:1], DemoModelClient, max_tool_calls=5, repeat=0
        )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _report(dataset: EvalDataset) -> EvalReport:
    cases = select_cases(dataset.cases, ids=["normal-001", "ambiguous-001"])
    results = run_evals(dataset, cases, DemoModelClient, max_tool_calls=5)
    return EvalReport(
        dataset_version="v1",
        dataset_today="2026-12-01",
        model="gemini-3.6-flash",
        prompt_id="search_agent@v1",
        started_at=datetime(2026, 12, 1, 15, 30, tzinfo=UTC),
        repeat=1,
        price_input_per_million=1.5,
        price_output_per_million=7.5,
        summary=summarize(
            results, price_input_per_million=1.5, price_output_per_million=7.5
        ),
        results=results,
    )


def test_markdown_report_shows_headline_and_failures(dataset: EvalDataset) -> None:
    text = render_markdown(_report(dataset))

    assert "| Cases passed | 1/2 (50%) |" in text
    assert "| Forbidden recommendations | **0** |" in text
    assert "| ambiguous-001 | failed | clarifying: no question asked |" in text
    assert "$1.50/1M input" in text


def test_write_report_saves_json_and_markdown(
    dataset: EvalDataset, tmp_path: Path
) -> None:
    report = _report(dataset)

    json_path, markdown_path = write_report(report, tmp_path / "reports")

    assert report_stem(report) == "20261201-153000-v1-gemini-3.6-flash"
    assert json_path.stem == markdown_path.stem == report_stem(report)
    assert EvalReport.model_validate_json(json_path.read_text()) == report
    assert markdown_path.read_text().startswith("# SpartanScout eval report: v1")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_cli_runs_the_demo_model_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--model", "demo", "--category", "unsafe", "--out", str(tmp_path)])

    output = capsys.readouterr().out
    assert code == 0
    assert "Running 5 case(s) x 1 on demo" in output
    assert "[1/5] unsafe-001" in output
    (json_file,) = tmp_path.glob("*.json")
    data = json.loads(json_file.read_text())
    assert data["model"] == "demo"
    assert data["summary"]["forbidden_recommendations"] == 0


def test_cli_rejects_unknown_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--model", "demo", "--case", "nope-001"]) == 2
    assert "unknown case ids" in capsys.readouterr().out


def test_cli_accepts_an_injected_model(tmp_path: Path) -> None:
    """Tests (and later CI) can supply any model, such as a scripted mock."""
    outage = MockModelClient([ModelUnavailableError("503")])

    code = main(
        ["--case", "normal-001", "--out", str(tmp_path), "--pause", "0"],
        model_factory=lambda: outage,
    )

    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert code == 0
    assert data["summary"]["errors"] == 1


# ---------------------------------------------------------------------------
# Quality gates (used by CI)
# ---------------------------------------------------------------------------


def test_gate_passes_for_the_demo_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--model",
            "demo",
            "--out",
            str(tmp_path),
            "--max-forbidden",
            "0",
            "--max-errors",
            "0",
            "--min-pass-rate",
            "0.8",
        ]
    )

    assert code == 0
    assert "Gate passed." in capsys.readouterr().out


def test_gate_fails_on_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outage = MockModelClient([ModelUnavailableError("503")])

    code = main(
        ["--case", "normal-001", "--out", str(tmp_path), "--max-errors", "0"],
        model_factory=lambda: outage,
    )

    assert code == 1
    assert "GATE FAILED: 1 errors (allowed: 0)" in capsys.readouterr().out


def test_gate_fails_when_the_pass_rate_is_too_low(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--model",
            "demo",
            "--category",
            "ambiguous",  # the demo model never asks questions: 0/3
            "--out",
            str(tmp_path),
            "--min-pass-rate",
            "0.5",
        ]
    )

    assert code == 1
    assert "pass rate 0% is below 50%" in capsys.readouterr().out
