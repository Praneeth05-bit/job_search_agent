"""Structured logging used to trace agent runs.

Every important step of a run writes one log event: a name plus key=value
fields, for example ``model_call step=agent turn=1 latency_ms=1840``.
Together the events of one run form its trace, which shows what happened,
in what order, how long each step took, and how many tokens it used.

What may be logged: event names, counts, ids, timings, statuses, and error
type names. What must never be logged: the API key, the student profile,
the student's request text, prompts, model output text, tool arguments, or
validation messages (they can echo model output). Each call site follows
this rule, and tests check it.

Written against structlog 26.1.0 (pinned). APIs checked on 2026-11-17
against the structlog 26.1.0 docs and source: ``structlog.configure``,
``make_filtering_bound_logger``, ``PrintLoggerFactory``,
``structlog.contextvars.merge_contextvars`` / ``bound_contextvars``,
``structlog.processors.add_log_level`` / ``TimeStamper`` /
``JSONRenderer``, ``structlog.dev.ConsoleRenderer`` and
``structlog.testing.capture_logs``.
"""

import sys
import time
from typing import Literal

import structlog
from structlog.typing import FilteringBoundLogger, Processor

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def configure_logging(*, level: LogLevel = "INFO", json_output: bool = False) -> None:
    """Set up structlog once, at program start (scripts, the API).

    Library code never calls this; it only asks for a logger. Tests do not
    call it either: they capture events with ``structlog.testing.capture_logs``.

    Args:
        level: The lowest level that is written. DEBUG shows everything.
        json_output: True writes one JSON object per line, for machines and
            log tools. False writes readable ``key=value`` lines for people.
    """
    processors: list[Processor] = [
        # Adds fields bound for the whole run, such as thread_id.
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # stderr keeps logs apart from the program's normal output on stdout.
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        # Not cached, so capture_logs() in tests always sees every event.
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a logger for one module. Use ``get_logger(__name__)``."""
    logger: FilteringBoundLogger = structlog.get_logger(name)
    return logger


def elapsed_ms(started: float) -> float:
    """Milliseconds since ``started``, a value from ``time.perf_counter()``."""
    return round((time.perf_counter() - started) * 1000, 1)
