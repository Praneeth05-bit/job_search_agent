"""Run the SpartanScout evals and save a report.

Usage, from the project root:
    uv run python scripts/run_evals.py --model demo   # free, offline check
    uv run python scripts/run_evals.py                # real Gemini score
    uv run python scripts/run_evals.py --help         # all options

Reports are written to evals/reports/ as JSON and Markdown.
"""

import sys

from job_search_agent.evals.cli import main

if __name__ == "__main__":
    sys.exit(main())
