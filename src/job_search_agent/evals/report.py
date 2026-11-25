"""Save eval reports: JSON for machines, Markdown for people and the README."""

from pathlib import Path

from job_search_agent.evals.runner import EvalReport

CHECK_LABELS = {
    "completed": "Run completed normally",
    "tool_choice": "Tool choice (searched when it should)",
    "recall": "Recall (required jobs recommended)",
    "forbidden": "No forbidden recommendations",
    "no_matches": "Recommends nothing when nothing fits",
    "clarifying": "Asks a clarifying question when needed",
    "injection": "Injection resistance (no planted text)",
}


def report_stem(report: EvalReport) -> str:
    """A sortable file name, such as "20261201-153000-v1-gemini-3.6-flash"."""
    model = "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in report.model)
    return f"{report.started_at:%Y%m%d-%H%M%S}-{report.dataset_version}-{model}"


def write_report(report: EvalReport, directory: Path) -> tuple[Path, Path]:
    """Write the report as JSON and Markdown. Returns both paths."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = report_stem(report)
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(report: EvalReport) -> str:
    """A readable summary: headline table, categories, checks, failures."""
    s = report.summary
    lines = [
        f"# SpartanScout eval report: {report.dataset_version}",
        "",
        f"- Model: `{report.model}` · Prompt: `{report.prompt_id}`",
        f"- Run at: {report.started_at:%Y-%m-%d %H:%M} UTC · "
        f"Dataset date: {report.dataset_today} · Attempts per case: {report.repeat}",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Cases passed | {s.cases} |",
        f"| Errors (did not run, e.g. HTTP 503) | {s.errors} |",
        f"| Forbidden recommendations | **{s.forbidden_recommendations}** |",
        f"| Median latency | {_seconds(s.latency_p50_ms)} |",
        f"| 95th percentile latency | {_seconds(s.latency_p95_ms)} |",
        f"| Tokens (input / output) | {s.input_tokens:,} / {s.output_tokens:,} |",
        f"| Estimated cost at paid rates | ${s.estimated_cost_usd:.4f} |",
    ]
    if s.consistent_cases is not None:
        lines.append(f"| Same verdict on every attempt | {s.consistent_cases} |")

    lines += ["", "## By category", "", "| Category | Passed |", "|---|---|"]
    lines += [f"| {name} | {tally} |" for name, tally in s.by_category.items()]

    lines += ["", "## By check", "", "| Check | Passed |", "|---|---|"]
    lines += [
        f"| {CHECK_LABELS.get(name, name)} | {tally} |"
        for name, tally in s.by_check.items()
    ]

    problems = [r for r in report.results if r.status != "passed"]
    lines += ["", "## Failures and errors", ""]
    if not problems:
        lines.append("None.")
    else:
        lines += ["| Case | Status | Details |", "|---|---|---|"]
        for result in problems:
            if result.status == "error":
                details = f"error: {result.error_type}"
            else:
                details = "; ".join(
                    f"{c.name}: {c.detail}" for c in result.checks if not c.passed
                )
            label = result.case_id
            if report.repeat > 1:
                label += f" (attempt {result.attempt})"
            lines.append(f"| {label} | {result.status} | {details} |")

    lines += [
        "",
        f"Cost assumes ${report.price_input_per_million:.2f}/1M input and "
        f"${report.price_output_per_million:.2f}/1M output tokens; "
        "on the free tier the actual cost is $0.",
        "",
    ]
    return "\n".join(lines)


def headline(report: EvalReport) -> str:
    """One line for the terminal."""
    s = report.summary
    return (
        f"Passed {s.cases} · errors {s.errors} · forbidden "
        f"{s.forbidden_recommendations} · median {_seconds(s.latency_p50_ms)} · "
        f"est. ${s.estimated_cost_usd:.4f}"
    )


def _seconds(milliseconds: float | None) -> str:
    return "n/a" if milliseconds is None else f"{milliseconds / 1000:.1f} s"
