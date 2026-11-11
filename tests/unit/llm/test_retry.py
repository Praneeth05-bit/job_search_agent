"""Tests for retries with backoff. They never really sleep."""

import pytest

from job_search_agent.llm.client import ModelClient
from job_search_agent.llm.errors import (
    ModelAuthError,
    ModelRateLimitError,
    ModelUnavailableError,
)
from job_search_agent.llm.mock import MockModelClient
from job_search_agent.llm.retry import (
    RetryingModelClient,
    backoff_delay,
    call_with_retries,
)
from job_search_agent.llm.schemas import ModelResponse, ToolOutput

OK = ModelResponse(text="ok")


class SleepRecorder:
    """Stands in for time.sleep: records each requested delay instantly."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def no_jitter() -> float:
    """Remove randomness so delays are exact and predictable."""
    return 0.0


def _retrying(
    mock: MockModelClient, sleep: SleepRecorder, max_retries: int = 2
) -> RetryingModelClient:
    """Build a retrying client around a mock, with fake sleep and no jitter."""
    return RetryingModelClient(
        mock, max_retries=max_retries, sleep=sleep, jitter=no_jitter
    )


def test_wrapper_satisfies_model_client_contract() -> None:
    """mypy checks this assignment: the wrapper is itself a ModelClient."""
    client: ModelClient = _retrying(MockModelClient([OK]), SleepRecorder())

    assert client.generate("hello") == OK


def test_succeeds_after_temporary_failures() -> None:
    """A 429 then a 503, then success: waits 1s, then 2s."""
    sleep = SleepRecorder()
    mock = MockModelClient(
        [ModelRateLimitError("429"), ModelUnavailableError("503"), OK]
    )

    result = _retrying(mock, sleep).generate("hello")

    assert result == OK
    assert sleep.delays == [1.0, 2.0]
    assert len(mock.prompts) == 3


def test_gives_up_after_max_retries() -> None:
    """With max_retries=2 there are exactly 3 attempts, then the error is raised."""
    sleep = SleepRecorder()
    mock = MockModelClient([ModelRateLimitError("429")] * 3)

    with pytest.raises(ModelRateLimitError):
        _retrying(mock, sleep).generate("hello")

    assert len(mock.prompts) == 3
    assert sleep.delays == [1.0, 2.0]


def test_permanent_error_is_not_retried() -> None:
    """A bad API key fails immediately: one call, no waiting."""
    sleep = SleepRecorder()
    mock = MockModelClient([ModelAuthError("401")])

    with pytest.raises(ModelAuthError):
        _retrying(mock, sleep).generate("hello")

    assert len(mock.prompts) == 1
    assert sleep.delays == []


def test_success_first_time_does_not_wait() -> None:
    """The common case costs nothing extra."""
    sleep = SleepRecorder()

    assert _retrying(MockModelClient([OK]), sleep).generate("hello") == OK
    assert sleep.delays == []


def test_zero_retries_means_single_attempt() -> None:
    """max_retries=0 turns retrying off."""
    sleep = SleepRecorder()
    mock = MockModelClient([ModelRateLimitError("429")])

    with pytest.raises(ModelRateLimitError):
        _retrying(mock, sleep, max_retries=0).generate("hello")

    assert len(mock.prompts) == 1


def test_delay_is_capped() -> None:
    """The exponential part never exceeds the maximum delay."""
    delays = [
        backoff_delay(
            attempt, base_delay_seconds=1.0, max_delay_seconds=8.0, jitter=no_jitter
        )
        for attempt in range(6)
    ]

    assert delays == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_jitter_adds_at_most_one_base_delay() -> None:
    """Jitter spreads retries out by up to one base delay."""
    delay = backoff_delay(
        0, base_delay_seconds=1.0, max_delay_seconds=8.0, jitter=lambda: 0.5
    )

    assert delay == 1.5


def test_non_model_exception_is_not_caught() -> None:
    """Bugs in our own code surface immediately instead of being retried."""
    sleep = SleepRecorder()
    calls: list[int] = []

    def buggy_call() -> str:
        calls.append(1)
        raise ValueError("bug in our code")

    with pytest.raises(ValueError, match="bug in our code"):
        call_with_retries(buggy_call, max_retries=2, sleep=sleep, jitter=no_jitter)

    assert len(calls) == 1
    assert sleep.delays == []


def test_wrapper_passes_multi_turn_arguments_through() -> None:
    """Retries must not drop the system instruction or tool outputs."""
    mock = MockModelClient([ModelRateLimitError("429"), OK])
    output = ToolOutput(call_id="c1", name="search_opportunities", result="{}")

    _retrying(mock, SleepRecorder()).generate(
        system_instruction="rules",
        previous_response_id="r1",
        tool_outputs=[output],
    )

    retried = mock.turns[1]  # the second attempt after the 429
    assert retried.system_instruction == "rules"
    assert retried.previous_response_id == "r1"
    assert retried.tool_outputs == [output]
