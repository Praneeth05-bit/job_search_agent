"""A deterministic, offline model for demos, the UI, and CI (USE_MOCK_MODEL=true).

``MockModelClient`` plays back a fixed script, which suits unit tests but
cannot answer a request nobody scripted. ``DemoModelClient`` follows the
same three-step plan for any request, so the whole app works without an API
key, without quota, and without outages:

1. First turn: ask for one search, using the request's keywords.
2. Tool results arrive: remember the matches, and say it is done searching.
3. Answer turn (a response schema is given): return a valid ``AgentAnswer``
   JSON built only from the remembered matches, best first.

It is not intelligent. It exists so the rest of the system (graph, API, UI)
can be shown and tested end to end. Real answers come from Gemini.
"""

import json
import threading
from collections import OrderedDict
from collections.abc import Sequence
from itertools import count
from typing import Any

from job_search_agent.llm.schemas import ModelResponse, ToolCall, ToolOutput, ToolSpec

SEARCH_TOOL = "search_opportunities"
MAX_RECOMMENDATIONS = 3

# How the agent's first-turn prompt introduces the request (see nodes.py).
_REQUEST_MARKER = "Student request:\n"

# Filler words removed so the keyword search sees only meaningful terms.
_FILLER_WORDS = frozenset(
    "a an and any are can find for get give i im i'm in is looking me my "
    "need of on please show some that the to want what which with".split()
)

# Remembered conversations are capped so memory cannot grow without limit.
_MAX_REMEMBERED = 1000


class DemoModelClient:
    """A rule-based ModelClient: search once, then answer from the results."""

    def __init__(self) -> None:
        self._ids = count(1)
        # response id -> the matches known at that point of the conversation
        self._matches: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        # The API serves requests on several threads at once.
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str = "",
        tools: Sequence[ToolSpec] = (),
        *,
        system_instruction: str | None = None,
        previous_response_id: str | None = None,
        tool_outputs: Sequence[ToolOutput] = (),
        response_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Return the next step of the plan for this conversation."""
        matches = self._recall(previous_response_id)

        if response_schema is not None:
            return self._reply(matches, text=_answer_json(matches))

        if tool_outputs:
            for output in tool_outputs:
                matches = matches + _matches_in(output)
            return self._reply(matches, text="I have the search results.")

        keywords = _keywords(prompt)
        if previous_response_id is None and keywords and _offers_search(tools):
            call = ToolCall(
                id=f"demo-call-{next(self._ids)}",
                name=SEARCH_TOOL,
                arguments={"text": keywords},
            )
            return self._reply(matches, tool_calls=[call])

        return self._reply(matches, text="I could not find search terms in that.")

    def _recall(self, response_id: str | None) -> list[dict[str, Any]]:
        if response_id is None:
            return []
        with self._lock:
            return list(self._matches.get(response_id, []))

    def _reply(
        self,
        matches: list[dict[str, Any]],
        *,
        text: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> ModelResponse:
        """Make a response with a new id, and remember the matches under it."""
        response_id = f"demo-{next(self._ids)}"
        with self._lock:
            self._matches[response_id] = matches
            while len(self._matches) > _MAX_REMEMBERED:
                self._matches.popitem(last=False)
        return ModelResponse(
            response_id=response_id, text=text, tool_calls=tool_calls or []
        )


def _keywords(prompt: str) -> str:
    """The request's meaningful words, taken from the first-turn prompt."""
    request = prompt.split(_REQUEST_MARKER, 1)[-1].split("\n\n", 1)[0]
    words = [word.strip(".,!?;:()\"'").lower() for word in request.split()]
    return " ".join(word for word in words if word and word not in _FILLER_WORDS)


def _offers_search(tools: Sequence[ToolSpec]) -> bool:
    return any(tool.name == SEARCH_TOOL for tool in tools)


def _matches_in(output: ToolOutput) -> list[dict[str, Any]]:
    """Read the usable matches from a search tool result (an error gives none)."""
    if output.is_error:
        return []
    body = output.result.strip()
    start, end = body.find("{"), body.rfind("}")  # inside the <tool_result> tags
    if start == -1 or end == -1:
        return []
    try:
        payload = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return []
    found = payload.get("matches", [])
    return [match for match in found if isinstance(match, dict)]


def _answer_json(matches: list[dict[str, Any]]) -> str:
    """An AgentAnswer as JSON: the first few unique matches, in search order."""
    seen: set[str] = set()
    ranked: list[dict[str, Any]] = []
    for match in matches:
        opportunity_id = match.get("opportunity_id")
        if not isinstance(opportunity_id, str) or opportunity_id in seen:
            continue
        seen.add(opportunity_id)
        ranked.append(
            {
                "opportunity_id": opportunity_id,
                "rank": len(ranked) + 1,
                "why": _why(match),
                "caveats": [str(note) for note in match.get("notes", [])],
            }
        )
        if len(ranked) == MAX_RECOMMENDATIONS:
            break

    if ranked:
        summary = f"Found {len(ranked)} opportunities that match your search."
    else:
        summary = "No matching opportunities were found for that search."
    return json.dumps({"summary": summary, "matches": ranked})


def _why(match: dict[str, Any]) -> str:
    terms = ", ".join(str(term) for term in match.get("matched_terms", []))
    status = match.get("eligibility", "unknown")
    if terms:
        return f"Matches your search terms ({terms}); eligibility: {status}."
    return f"Related to your search; eligibility: {status}."
