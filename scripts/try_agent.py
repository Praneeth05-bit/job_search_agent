"""Run the agent once against real Gemini and print what happened.

A live smoke test, not an automated test: it checks that Gemini accepts the
tool schema, calls the tool, continues the conversation, and follows the
prompt. It always uses the real model (USE_MOCK_MODEL is ignored) and needs
GEMINI_API_KEY in .env. Each run uses a small amount of free-tier quota.

Usage, from the project root:
    uv run python scripts/try_agent.py "Find Python internships I can apply to"
"""

import sys
from datetime import date

from job_search_agent.agent.graph import AgentRun, run_agent
from job_search_agent.config import PROJECT_ROOT, get_settings
from job_search_agent.data.loaders import load_opportunities
from job_search_agent.domain.models import StudentProfile
from job_search_agent.llm.errors import ModelError
from job_search_agent.llm.gemini import create_gemini_client

PROFILE_PATH = PROJECT_ROOT / "data" / "demo" / "profiles" / "demo_student.json"
DEFAULT_REQUEST = "Find Python internships I can apply to"


def print_run(request: str, run: AgentRun) -> None:
    """Print a readable summary of one agent run."""
    print(f"Request:      {request}")
    print(f"Prompt:       {run.prompt_id}")
    print(f"Stop reason:  {run.stop_reason}")
    print(f"Model calls:  {run.model_calls_made}")
    print(f"Tool calls:   {run.tool_calls_made}")

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


def main() -> int:
    """Run one live request and return a process exit code."""
    request = " ".join(sys.argv[1:]) or DEFAULT_REQUEST
    settings = get_settings()
    profile = StudentProfile.model_validate_json(
        PROFILE_PATH.read_text(encoding="utf-8")
    )

    try:
        run = run_agent(
            request,
            profile,
            model=create_gemini_client(settings),
            opportunities=load_opportunities(settings.data_dir),
            today=date.today(),
            max_tool_calls=settings.max_tool_calls,
        )
    except ModelError as exc:
        print(f"Model error ({type(exc).__name__}): {exc}")
        return 1

    print_run(request, run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
