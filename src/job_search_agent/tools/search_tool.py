"""The search_opportunities tool: the bridge between the model and our code.

The model can only ask for a tool by name with JSON arguments. This module
guards that boundary:

1. Only the registered tool name is accepted.
2. Arguments are validated by SearchQuery. Invalid arguments are returned to
   the model as an error message so it can correct itself, instead of
   crashing the run.
3. The student profile, data, and date come from our code, never from the
   model's arguments.
4. Results are sent back as compact JSON inside a <tool_result> envelope,
   with "<" escaped so third-party text cannot fake the end of the envelope.
"""

import json
from collections.abc import Sequence
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from job_search_agent.domain.eligibility import EligibilityStatus
from job_search_agent.domain.models import Opportunity, StudentProfile
from job_search_agent.llm.schemas import ToolCall, ToolSpec
from job_search_agent.retrieval.schemas import SearchQuery
from job_search_agent.services.matching import Match, MatchReport, find_matches

SEARCH_TOOL_NAME = "search_opportunities"
_DESCRIPTION_PREVIEW_CHARS = 300

# JSON Schema keywords we send to the model. Model APIs accept a subset of
# JSON Schema, so we keep only simple, widely supported keywords. Full
# validation still happens on our side with SearchQuery.
_ALLOWED_SCHEMA_KEYS = {
    "type",
    "description",
    "properties",
    "required",
    "enum",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "items",
}


class ToolResult(BaseModel):
    """What happened when a tool call ran: text for the model, data for us."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    call_id: str | None = None
    ok: bool
    content: str  # exactly what the model will read
    report: MatchReport | None = None  # full result for the UI and traces


# ---------------------------------------------------------------------------
# Tool specification (the menu entry the model sees)
# ---------------------------------------------------------------------------


def model_facing_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build a flat, simple JSON schema for a model from a Pydantic class.

    Inlines ``$ref`` definitions, turns "this type or null" into just the
    type (optional fields are simply omitted), and keeps only the keywords in
    ``_ALLOWED_SCHEMA_KEYS``.
    """
    raw = model.model_json_schema()
    definitions: dict[str, Any] = raw.get("$defs", {})
    cleaned = _clean_schema(raw, definitions)
    assert isinstance(cleaned, dict)
    return cleaned


def _clean_schema(node: Any, definitions: dict[str, Any]) -> Any:
    """Recursively simplify one schema node."""
    if isinstance(node, list):
        return [_clean_schema(item, definitions) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        name = str(node["$ref"]).rsplit("/", 1)[-1]
        return _clean_schema(definitions[name], definitions)

    if "anyOf" in node:
        options = [option for option in node["anyOf"] if option != {"type": "null"}]
        if len(options) == 1:
            merged = _clean_schema(options[0], definitions)
            if "description" in node:
                merged["description"] = node["description"]
            return merged

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key == "properties":
            cleaned[key] = {
                field: _clean_schema(schema, definitions)
                for field, schema in value.items()
            }
        elif key in _ALLOWED_SCHEMA_KEYS:
            cleaned[key] = _clean_schema(value, definitions)
    return cleaned


def search_tool_spec() -> ToolSpec:
    """The search tool's menu entry, generated from the SearchQuery contract."""
    return ToolSpec(
        name=SEARCH_TOOL_NAME,
        description=(
            "Search approved campus opportunity records and check the student's "
            "eligibility. Returns eligible or unknown matches, plus excluded "
            "opportunities with the reasons they were excluded."
        ),
        parameters=model_facing_schema(SearchQuery),
    )


# ---------------------------------------------------------------------------
# Running the tool
# ---------------------------------------------------------------------------


def run_search_tool(
    call: ToolCall,
    *,
    profile: StudentProfile,
    opportunities: Sequence[Opportunity],
    today: date,
) -> ToolResult:
    """Validate a model's tool call, run the search, and format the result.

    Never raises for model mistakes: unknown tools and invalid arguments come
    back as ``ok=False`` results the model can read and correct.

    Args:
        call: The tool call requested by the model.
        profile: The student's facts, from the agent state (not the model).
        opportunities: The records to search.
        today: The date used for deadline checks.
    """
    if call.name != SEARCH_TOOL_NAME:
        return _error(
            call, f"Unknown tool '{call.name}'. Available: {SEARCH_TOOL_NAME}"
        )

    try:
        query = SearchQuery.model_validate(call.arguments)
    except ValidationError as exc:
        return _error(call, f"Invalid arguments: {_summarize(exc)}")

    report = find_matches(query, profile, opportunities, today)
    records = {opportunity.id: opportunity for opportunity in opportunities}
    payload = {
        "matches": [_match_view(match, records) for match in report.matches],
        "excluded": [_excluded_view(match, records) for match in report.excluded],
    }
    return ToolResult(
        name=SEARCH_TOOL_NAME,
        call_id=call.id,
        ok=True,
        content=_wrap(payload),
        report=report,
    )


def _match_view(match: Match, records: dict[str, Opportunity]) -> dict[str, Any]:
    """The compact view of a usable match that the model needs to explain it."""
    record = records[match.search.opportunity_id]
    return {
        "opportunity_id": record.id,
        "title": record.title,
        "organization": record.organization,
        "type": record.opportunity_type.value,
        "deadline": record.deadline.isoformat() if record.deadline else None,
        "eligibility": match.eligibility.status.value,
        "matched_terms": match.search.matched_terms,
        "notes": [
            check.reason
            for check in match.eligibility.checks
            if check.status is EligibilityStatus.UNKNOWN
        ],
        "description": _preview(record.description),
    }


def _excluded_view(match: Match, records: dict[str, Opportunity]) -> dict[str, Any]:
    """The compact view of an excluded match: only why it was excluded."""
    record = records[match.search.opportunity_id]
    return {
        "opportunity_id": record.id,
        "title": record.title,
        "reasons": [
            check.reason
            for check in match.eligibility.checks
            if check.status is EligibilityStatus.INELIGIBLE
        ],
    }


def _preview(text: str) -> str:
    """Shorten long descriptions to save tokens."""
    if len(text) <= _DESCRIPTION_PREVIEW_CHARS:
        return text
    return text[:_DESCRIPTION_PREVIEW_CHARS].rstrip() + "..."


def _summarize(exc: ValidationError) -> str:
    """A short, model-readable summary of what was wrong with the arguments."""
    problems = [
        f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
        for error in exc.errors()[:3]
    ]
    return "; ".join(problems)


def _error(call: ToolCall, message: str) -> ToolResult:
    """A failed tool result the model can read and react to."""
    return ToolResult(
        name=call.name,
        call_id=call.id,
        ok=False,
        content=_wrap({"error": message}),
    )


def _wrap(payload: dict[str, Any]) -> str:
    """Put JSON inside the untrusted-data envelope, blocking envelope breakout.

    Escaping "<" as \\u003c keeps the JSON valid and its meaning unchanged,
    but third-party text can never produce a real "</tool_result>" tag.
    """
    body = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return f'<tool_result name="{SEARCH_TOOL_NAME}">\n{body}\n</tool_result>'
