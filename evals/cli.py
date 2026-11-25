"""Command line for running evals: ``uv run python scripts/run_evals.py``.

Examples, from the project root:
    uv run python scripts/run_evals.py --model demo          # free, offline
    uv run python scripts/run_evals.py                       # real Gemini
    uv run python scripts/run_evals.py --case unsafe-001     # one case
    uv run python scripts/run_evals.py --category unsafe --repeat 3

Quality gates (used by CI) make the command exit with code 1 when broken:
    uv run python scripts/run_evals.py --model demo --max-forbidden 0 \
        --max-errors 0 --min-pass-rate 0.8
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from job_search_agent.agent.graph import PROMPT_NAME, PROMPT_VERSION
from job_search_agent.config import get_settings
from job_search_agent.evals.dataset import EVALS_DIR, load_dataset
from job_search_agent.evals.report import headline, write_report
from job_search_agent.evals.runner import (
    EvalReport,
    ModelFactory,
    now_utc,
    run_evals,
    select_cases,
)
from job_search_agent.evals.scoring import CaseResult, gate_problems, summarize
from job_search_agent.llm.demo import DemoModelClient
from job_search_agent.llm.gemini import create_gemini_client
from job_search_agent.observability.tracing import configure_logging
from job_search_agent.prompts.loader import load_prompt

DEFAULT_GEMINI_PAUSE_SECONDS = 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SpartanScout evals.")
    parser.add_argument(
        "--model",
        choices=["gemini", "demo"],
        default="gemini",
        help="gemini = the configured Gemini model; demo = free offline model",
    )
    parser.add_argument("--dataset", default="v1", help="dataset version folder")
    parser.add_argument(
        "--case", action="append", default=[], help="case id (repeatable)"
    )
    parser.add_argument(
        "--category", action="append", default=[], help="category (repeatable)"
    )
    parser.add_argument("--limit", type=int, default=None, help="run only the first N")
    parser.add_argument("--repeat", type=int, default=1, help="attempts per case")
    parser.add_argument(
        "--pause",
        type=float,
        default=None,
        help=f"seconds between runs (default {DEFAULT_GEMINI_PAUSE_SECONDS} for "
        "gemini, 0 for demo)",
    )
    parser.add_argument(
        "--out", type=Path, default=EVALS_DIR / "reports", help="report folder"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="show the agent's trace lines"
    )
    gates = parser.add_argument_group("quality gates (exit code 1 when a gate fails)")
    gates.add_argument(
        "--max-forbidden", type=int, default=None, help="most forbidden recommendations"
    )
    gates.add_argument("--max-errors", type=int, default=None, help="most errors")
    gates.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help="least share of scored cases that must pass, from 0 to 1",
    )
    return parser


def main(
    argv: Sequence[str] | None = None, *, model_factory: ModelFactory | None = None
) -> int:
    """Run the evals described by ``argv``. Returns the process exit code.

    ``model_factory`` lets tests supply a model; normally it comes from --model.
    """
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(level="INFO" if args.verbose else "WARNING")

    dataset = load_dataset(args.dataset)
    try:
        cases = select_cases(
            dataset.cases, ids=args.case, categories=args.category, limit=args.limit
        )
    except ValueError as error:
        print(f"Error: {error}")
        return 2

    if model_factory is None:
        if args.model == "demo":
            model_factory = DemoModelClient
        else:
            client = create_gemini_client(settings)
            model_factory = lambda: client  # noqa: E731 - one client, reused
    model_name = "demo" if args.model == "demo" else settings.gemini_model
    pause = args.pause
    if pause is None:
        pause = 0.0 if args.model == "demo" else DEFAULT_GEMINI_PAUSE_SECONDS

    started_at = now_utc()
    print(f"Running {len(cases)} case(s) x {args.repeat} on {model_name} ...")
    results = run_evals(
        dataset,
        cases,
        model_factory,
        max_tool_calls=settings.max_tool_calls,
        repeat=args.repeat,
        pause_seconds=pause,
        on_result=print_progress,
    )

    report = EvalReport(
        dataset_version=dataset.manifest.version,
        dataset_today=dataset.manifest.today.isoformat(),
        model=model_name,
        prompt_id=load_prompt(PROMPT_NAME, PROMPT_VERSION).id,
        started_at=started_at,
        repeat=args.repeat,
        price_input_per_million=settings.price_input_per_million,
        price_output_per_million=settings.price_output_per_million,
        summary=summarize(
            results,
            price_input_per_million=settings.price_input_per_million,
            price_output_per_million=settings.price_output_per_million,
        ),
        results=results,
    )
    json_path, markdown_path = write_report(report, args.out)
    print()
    print(headline(report))
    print(f"Report: {markdown_path}")
    print(f"Data:   {json_path}")

    gates = (args.max_forbidden, args.max_errors, args.min_pass_rate)
    if all(gate is None for gate in gates):
        return 0
    problems = gate_problems(
        report.summary,
        max_forbidden=args.max_forbidden,
        min_pass_rate=args.min_pass_rate,
        max_errors=args.max_errors,
    )
    if problems:
        print("GATE FAILED: " + "; ".join(problems))
        return 1
    print("Gate passed.")
    return 0


def print_progress(number: int, total: int, result: CaseResult) -> None:
    """One line per finished run, such as "[ 3/30] normal-003  PASSED  6.1 s"."""
    width = len(str(total))
    status = result.status.upper()
    extra = ""
    if result.status == "failed":
        extra = "  (" + ", ".join(result.failed_checks) + ")"
    elif result.status == "error":
        extra = f"  ({result.error_type})"
    attempt = f" #{result.attempt}" if result.attempt > 1 else ""
    print(
        f"[{number:>{width}}/{total}] {result.case_id:<18}{attempt} {status:<7}"
        f"{result.latency_ms / 1000:6.1f} s{extra}"
    )
