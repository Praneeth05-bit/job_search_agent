"""Bounded retries with exponential backoff and jitter for model calls.

Only errors marked ``retryable`` (timeouts, rate limits, temporary server
problems) are retried. Permanent model errors fail immediately, and any other
exception, such as a bug in our own code, is never caught here.

Waits grow exponentially (1s, 2s, 4s, ...), are capped, and get a small random
"jitter" so many clients do not retry at the same instant. ``sleep`` and
``jitter`` are parameters so tests can run instantly and predictably.
"""

import random
import time
from collections.abc import Callable, Sequence
from typing import Any

from job_search_agent.llm.client import ModelClient
from job_search_agent.llm.errors import ModelError
from job_search_agent.llm.schemas import ModelResponse, ToolOutput, ToolSpec


def backoff_delay(
    attempt: int,
    *,
    base_delay_seconds: float,
    max_delay_seconds: float,
    jitter: Callable[[], float],
) -> float:
    """Return how long to wait before the next attempt.

    Args:
        attempt: Zero-based index of the attempt that just failed.
        base_delay_seconds: Wait after the first failure, before jitter.
        max_delay_seconds: Upper limit for the exponential part.
        jitter: Returns a random fraction in [0, 1); scaled by the base delay.

    Returns:
        ``min(max, base * 2**attempt) + jitter() * base`` seconds.
    """
    exponential = min(max_delay_seconds, base_delay_seconds * 2.0**attempt)
    return exponential + jitter() * base_delay_seconds


def call_with_retries[T](
    call: Callable[[], T],
    *,
    max_retries: int,
    base_delay_seconds: float = 1.0,
    max_delay_seconds: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> T:
    """Run ``call``, retrying temporary model errors with backoff.

    Args:
        call: Zero-argument function that performs one attempt.
        max_retries: Extra attempts after the first. 2 means up to 3 tries.
        base_delay_seconds: Wait after the first failure, before jitter.
        max_delay_seconds: Upper limit for the exponential part of a wait.
        sleep: Function used to wait. Replaced by a recorder in tests.
        jitter: Random fraction in [0, 1). Replaced by a constant in tests.

    Returns:
        Whatever ``call`` returns on its first successful attempt.

    Raises:
        ModelError: The last error, if it is permanent or retries run out.
    """
    for attempt in range(max_retries + 1):
        try:
            return call()
        except ModelError as error:
            is_last_attempt = attempt == max_retries
            if not error.retryable or is_last_attempt:
                raise
            sleep(
                backoff_delay(
                    attempt,
                    base_delay_seconds=base_delay_seconds,
                    max_delay_seconds=max_delay_seconds,
                    jitter=jitter,
                )
            )
    # The loop always returns or raises; this line is unreachable.
    raise AssertionError("unreachable")


class RetryingModelClient:
    """Wraps any ModelClient and adds retries, without changing it.

    Because this class also satisfies ModelClient, the agent cannot tell
    whether retries are present (the decorator pattern).
    """

    def __init__(
        self,
        inner: ModelClient,
        *,
        max_retries: int,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        """Wrap ``inner`` so each generate() call is retried on temporary errors."""
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds
        self._sleep = sleep
        self._jitter = jitter

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
        """Call the wrapped client, retrying temporary failures."""
        return call_with_retries(
            lambda: self._inner.generate(
                prompt,
                tools,
                system_instruction=system_instruction,
                previous_response_id=previous_response_id,
                tool_outputs=tool_outputs,
                response_schema=response_schema,
            ),
            max_retries=self._max_retries,
            base_delay_seconds=self._base_delay_seconds,
            max_delay_seconds=self._max_delay_seconds,
            sleep=self._sleep,
            jitter=self._jitter,
        )
