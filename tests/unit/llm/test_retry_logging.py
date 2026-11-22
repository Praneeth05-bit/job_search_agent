"""Retries leave a trace: one model_retry event per extra attempt."""

import pytest
from structlog.testing import capture_logs

from job_search_agent.llm.errors import ModelAuthError, ModelUnavailableError
from job_search_agent.llm.retry import call_with_retries


class FailingThenOk:
    """Raises the given errors in order, then returns "ok"."""

    def __init__(self, *errors: Exception) -> None:
        self._errors = list(errors)

    def __call__(self) -> str:
        if self._errors:
            raise self._errors.pop(0)
        return "ok"


def _no_sleep(seconds: float) -> None:
    """Skip the real wait."""


def test_each_retry_is_logged_with_error_type_and_wait() -> None:
    call = FailingThenOk(ModelUnavailableError("503"), ModelUnavailableError("503"))

    with capture_logs() as logs:
        result = call_with_retries(
            call, max_retries=2, sleep=_no_sleep, jitter=lambda: 0.0
        )

    assert result == "ok"
    assert [entry["event"] for entry in logs] == ["model_retry", "model_retry"]
    assert [entry["retry"] for entry in logs] == [1, 2]
    assert [entry["delay_s"] for entry in logs] == [1.0, 2.0]
    assert logs[0]["error_type"] == "ModelUnavailableError"
    assert logs[0]["max_retries"] == 2
    assert logs[0]["log_level"] == "warning"


def test_permanent_errors_are_not_logged_as_retries() -> None:
    call = FailingThenOk(ModelAuthError("401"))

    with capture_logs() as logs:
        with pytest.raises(ModelAuthError):
            call_with_retries(call, max_retries=2, sleep=_no_sleep)

    assert logs == []
