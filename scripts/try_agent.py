"""Run the agent once against real Gemini and print what happened.

A live smoke test, not an automated test: it checks that Gemini accepts the
tool schema, calls the tool, continues the conversation, follows the prompt,
and returns a grounded structured answer. If the agent proposes saving
matches, you are asked to approve before anything is written.

It always uses the real model (USE_MOCK_MODEL is ignored) and needs
GEMINI_API_KEY in .env. Each run uses a small amount of free-tier quota.

The run's trace (one log line per step) is written to stderr, so it appears
in the terminal above the printed summary. Set LOG_JSON=true in .env for JSON
lines, or LOG_LEVEL=WARNING to hide the trace.

Usage, from the project root:
    uv run python scripts/try_agent.py "Find Python internships I can apply to"
"""

import sys
from datetime import date

from job_search_agent.agent.graph import AgentRun, SearchAgent
from job_search_agent.config import PROJECT_ROOT, get_settings
from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.models import StudentProfile
from job_search_agent.llm.errors import ModelError
from job_search_agent.llm.gemini import create_gemini_client
from job_search_agent.observability.tracing import configure_logging
from job_search_agent.storage.tracker import JsonFileTracker

PROFILE_PATH = PROJECT_ROOT / "data" / "demo" / "profiles" / "demo_student.json"
DEFAULT_REQUEST = "Find Python internships I can apply to"


def print_run(run: AgentRun) -> None:
    """Print a readable summary of one agent run."""
    print(f"Status:       {run.status}")
    print(f"Stop reason:  {run.stop_reason}")
    print(f"Model calls:  {run.model_calls_made}")
    print(f"Tool calls:   {run.tool_calls_made}")
    print(f"Tokens:       {run.input_tokens} in / {run.output_tokens} out")

    for number, result in enumerate(run.tool_results, start=1):
        print(f"\nTool result {number}: ok={result.ok}")
        if result.report is None:
            print(f"  {result.content}")
            continue
        matches = [m.search.opportunity_id for m in result.report.matches]
        excluded = [m.search.opportunity_id for m in result.report.excluded]
        print(f"  matches:  {matches}")
        print(f"  excluded: {excluded}")

    print("\nFinal answer:")
    print(run.final_answer)
    if run.answer is None:
        return
    for match in run.answer.matches:
        print(f"\n  #{match.rank} {match.opportunity_id}: {match.why}")
        for caveat in match.caveats:
            print(f"     caveat: {caveat}")
    if run.answer.clarifying_question:
        print(f"\n  Question: {run.answer.clarifying_question}")


def main() -> int:
    """Run one live request, ask for approval if needed, and return an exit code."""
    request = " ".join(sys.argv[1:]) or DEFAULT_REQUEST
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    profile = StudentProfile.model_validate_json(
        PROFILE_PATH.read_text(encoding="utf-8")
    )
    agent = SearchAgent(
        model=create_gemini_client(settings),
        opportunities=load_opportunities(settings.data_dir),
        today=date.today(),
        max_tool_calls=settings.max_tool_calls,
        tracker=JsonFileTracker(settings.tracker_path),
    )

    print(f"Request: {request}\n")
    try:
        run = agent.start(request, profile)
    except ModelError as exc:
        print(exc.user_message)  # what a student would see
        print(f"  details: {type(exc).__name__}: {exc}")  # for you, the developer
        return 1
    print_run(run)

    if run.pending_approval is not None:
        ids = ", ".join(run.pending_approval.opportunity_ids)
        reply = input(f"\nSave {ids} to your tracker? [y/N] ").strip().lower()
        done = agent.resume(run.thread_id, approved=reply == "y")
        if done.saved_ids:
            print(f"Saved: {', '.join(done.saved_ids)} -> {settings.tracker_path}")
        else:
            print("Nothing saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
