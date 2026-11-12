"""The agent's structured final answer and the checks that guard it.

The model is asked to return JSON in the AgentAnswer shape. Two checks run in
our code before any answer reaches a student:

1. Schema check: Pydantic validates the JSON (fields, types, ranks).
2. Grounding check: every recommended opportunity must have been returned as
   a match by the search tool in this run. Opportunities that were never
   retrieved (invented) or that were excluded (ineligible) are rejected.

Pydantic alone cannot catch the second kind of problem: an answer can match
the schema perfectly and still recommend an ineligible job.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from job_search_agent.tools.search_tool import ToolResult


class RankedMatch(BaseModel):
    """One recommended opportunity, with the reason and any caveats."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opportunity_id: str = Field(
        min_length=1,
        description="The id from the search results, such as 'opp-003'.",
    )
    rank: int = Field(ge=1, description="1 is the best match.")
    why: str = Field(
        min_length=1,
        description="One or two sentences explaining the match, using the tool data.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Eligibility notes the student should verify, if any.",
    )


class AgentAnswer(BaseModel):
    """The complete answer shown to the student."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(
        min_length=1,
        description="A short, friendly summary for the student.",
    )
    matches: list[RankedMatch] = Field(
        default_factory=list,
        description="Recommended opportunities, best first. Empty if none fit.",
    )
    clarifying_question: str | None = Field(
        default=None,
        description="One question to ask the student if the request is unclear.",
    )


def retrieved_ids(tool_results: Sequence[ToolResult]) -> tuple[set[str], set[str]]:
    """Collect the opportunity ids the search tool returned in this run.

    Returns:
        ``(matched_ids, excluded_ids)`` across every successful tool result.
    """
    matched: set[str] = set()
    excluded: set[str] = set()
    for result in tool_results:
        if result.report is None:
            continue
        matched.update(m.search.opportunity_id for m in result.report.matches)
        excluded.update(m.search.opportunity_id for m in result.report.excluded)
    return matched, excluded - matched


def grounding_problems(
    answer: AgentAnswer, tool_results: Sequence[ToolResult]
) -> list[str]:
    """List every way the answer is not supported by this run's tool results.

    An empty list means the answer is grounded.
    """
    matched, excluded = retrieved_ids(tool_results)
    problems: list[str] = []

    ids = [match.opportunity_id for match in answer.matches]
    for opportunity_id in ids:
        if opportunity_id in excluded:
            problems.append(
                f"{opportunity_id} was excluded as ineligible "
                "and must not be recommended"
            )
        elif opportunity_id not in matched:
            problems.append(f"{opportunity_id} was not returned by the search tool")

    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"Duplicate opportunities: {', '.join(duplicates)}")

    ranks = sorted(match.rank for match in answer.matches)
    if ranks != list(range(1, len(ranks) + 1)):
        problems.append("Ranks must be 1, 2, 3, ... with no gaps or repeats")

    return problems


def parse_answer(
    text: str | None, tool_results: Sequence[ToolResult]
) -> tuple[AgentAnswer | None, list[str]]:
    """Validate the model's JSON reply against the schema and the tool results.

    Returns:
        ``(answer, [])`` when the reply is valid and grounded, otherwise
        ``(None, problems)`` with readable problems the model can fix.
    """
    if not text:
        return None, ["The reply was empty; return JSON matching the schema"]
    try:
        answer = AgentAnswer.model_validate_json(text)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(p) for p in err['loc']) or 'answer'}: {err['msg']}"
            for err in exc.errors()[:5]
        ]
        return None, problems

    problems = grounding_problems(answer, tool_results)
    return (None, problems) if problems else (answer, [])
