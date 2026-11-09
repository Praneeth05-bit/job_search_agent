"""Gemini adapter: the only module that talks to Google's SDK.

Written against google-genai 2.24.0, verified by introspecting the installed
SDK on 2026-11-09:

* ``client.interactions.create(*, timeout: float | httpx.Timeout, **body)``
  accepts a per-call timeout in seconds.
* The returned ``Interaction`` has ``id``, ``status``, ``output_text``,
  ``steps`` (tool requests are steps with ``type == "function_call"``), and
  ``usage.total_input_tokens`` / ``usage.total_output_tokens``.
* ``google.genai.errors.APIError`` carries the HTTP status in ``.code``.

Everything is translated into our own ModelResponse and ModelError types, so
no SDK object or exception escapes this file.
"""

import time
from collections.abc import Sequence
from typing import Any

from job_search_agent.config import Settings
from job_search_agent.llm.client import ModelClient
from job_search_agent.llm.errors import (
    ModelAuthError,
    ModelError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from job_search_agent.llm.retry import RetryingModelClient
from job_search_agent.llm.schemas import ModelResponse, ToolCall, ToolSpec

# Interaction statuses that mean the model did not produce a usable answer.
_FAILED_STATUSES = {"failed", "cancelled", "incomplete", "budget_exceeded"}


class GeminiClient:
    """A ModelClient backed by the Gemini Interactions API."""

    def __init__(self, sdk_client: Any, *, model: str, timeout_seconds: float) -> None:
        """Create the adapter.

        Args:
            sdk_client: A ``google.genai.Client``, or a fake with the same
                ``interactions.create`` method in tests. Typed as Any because
                the SDK's generated types are internal and change between
                versions; this file is the only place they are touched.
            model: Gemini model name, such as "gemini-3.6-flash".
            timeout_seconds: Per-call timeout passed to the SDK.
        """
        self._sdk = sdk_client
        self._model = model
        self._timeout_seconds = timeout_seconds

    def generate(self, prompt: str, tools: Sequence[ToolSpec] = ()) -> ModelResponse:
        """Send one prompt to Gemini and return a provider-neutral response.

        Raises:
            ModelError: A subclass describing what went wrong.
        """
        request: dict[str, Any] = {"model": self._model, "input": prompt}
        if tools:
            request["tools"] = [_to_gemini_tool(tool) for tool in tools]

        started = time.perf_counter()
        try:
            interaction = self._sdk.interactions.create(
                timeout=self._timeout_seconds, **request
            )
        except Exception as exc:
            translated = translate_error(exc)
            if translated is None:
                raise  # not a provider problem: likely a bug, so surface it
            raise translated from exc
        latency_ms = (time.perf_counter() - started) * 1000

        return to_model_response(interaction, latency_ms)


def _to_gemini_tool(tool: ToolSpec) -> dict[str, Any]:
    """Convert our ToolSpec into the Interactions API function format."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def _status_code(exc: BaseException) -> int | None:
    """Find an HTTP status code on an SDK or HTTP-library exception, if any."""
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def translate_error(exc: BaseException) -> ModelError | None:
    """Map a provider exception to one of our ModelError types.

    Returns None for exceptions that are not provider problems, so the caller
    can re-raise them unchanged. Messages never include request content or
    credentials.

    Timeout and connection errors are recognised by class name because the
    SDK may use either the ``httpx`` or ``httpx2`` package underneath.
    """
    name = type(exc).__name__.lower()
    if isinstance(exc, TimeoutError) or "timeout" in name:
        return ModelTimeoutError("Gemini did not respond in time")

    status = _status_code(exc)
    if status is not None:
        if status in (401, 403):
            return ModelAuthError(
                f"Gemini rejected the API key (HTTP {status}); check GEMINI_API_KEY"
            )
        if status == 429:
            return ModelRateLimitError("Gemini rate limit reached (HTTP 429)")
        if status == 408:
            return ModelTimeoutError("Gemini request timed out (HTTP 408)")
        if status >= 500:
            return ModelUnavailableError(f"Gemini server error (HTTP {status})")
        return ModelResponseError(f"Gemini rejected the request (HTTP {status})")

    if "connect" in name or "network" in name or "transport" in name:
        return ModelUnavailableError("Could not reach Gemini (network error)")
    return None


def to_model_response(interaction: Any, latency_ms: float) -> ModelResponse:
    """Convert an SDK Interaction into our ModelResponse.

    Raises:
        ModelResponseError: If the interaction failed or contains nothing usable.
    """
    status = getattr(interaction, "status", None)
    if status in _FAILED_STATUSES:
        raise ModelResponseError(f"Gemini interaction ended with status '{status}'")

    tool_calls: list[ToolCall] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "function_call":
            continue
        name = getattr(step, "name", None)
        arguments = getattr(step, "arguments", None) or {}
        if not isinstance(name, str) or not name:
            raise ModelResponseError("Gemini returned a function call with no name")
        if not isinstance(arguments, dict):
            raise ModelResponseError("Gemini returned non-object function arguments")
        call_id = getattr(step, "id", None)
        tool_calls.append(
            ToolCall(
                id=call_id if isinstance(call_id, str) else None,
                name=name,
                arguments=dict(arguments),
            )
        )

    text = getattr(interaction, "output_text", None) or None
    if text is None and not tool_calls:
        raise ModelResponseError("Gemini returned neither text nor a tool call")

    usage = getattr(interaction, "usage", None)
    response_id = getattr(interaction, "id", None)
    return ModelResponse(
        response_id=response_id if isinstance(response_id, str) else None,
        text=text,
        tool_calls=tool_calls,
        latency_ms=latency_ms,
        input_tokens=getattr(usage, "total_input_tokens", None),
        output_tokens=getattr(usage, "total_output_tokens", None),
    )


def create_gemini_client(settings: Settings) -> ModelClient:
    """Build the production model client: Gemini wrapped with our retries.

    The SDK's own retries are turned off (attempts=1) so retries happen in
    exactly one place, RetryingModelClient. Two retry layers would multiply
    attempts and burn quota.

    Raises:
        ModelAuthError: If no API key is configured.
    """
    if settings.gemini_api_key is None:
        raise ModelAuthError("GEMINI_API_KEY is not set (or set USE_MOCK_MODEL=true)")

    from google import genai  # imported here so tests never need the SDK
    from google.genai import types

    sdk = genai.Client(
        api_key=settings.gemini_api_key.get_secret_value(),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    gemini = GeminiClient(
        sdk,
        model=settings.gemini_model,
        timeout_seconds=settings.model_timeout_seconds,
    )
    return RetryingModelClient(gemini, max_retries=settings.model_max_retries)
