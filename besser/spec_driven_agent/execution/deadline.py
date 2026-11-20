"""A run's deadline: its wall-clock cap and its cancel signal, in one object.

Every LLM call and every command the run starts reads it. A call or command
gets at most the time left and is not started when too little is left; a
cancel stops the call or command that is running instead of waiting for it.
The run's entry points make the deadline current for their thread
(:func:`under_deadline`); a thread working for the run copies its context.
"""

import contextlib
import contextvars
import functools
import logging
import threading
from typing import Callable, Iterator

from besser.spec_driven_agent.errors import DeadlineReached

logger = logging.getLogger(__name__)

# Left free at the end of the run for packaging: formatting, the recipe and
# report, the checkpoint and the download archive, which take seconds. LLM
# calls and the agent's own commands stop this long before the cap.
PACKAGING_RESERVE_SECONDS = 30.0
# An LLM call is not started with less time than this: it would rarely finish,
# and a call cut short is paid for and gives nothing.
MIN_CALL_SECONDS = 20.0
# A validation round is not started closer to the cap than this; its commands
# are cut at the cap.
MIN_CHECK_SECONDS = 10.0

CANCELLED = "the run was cancelled"
OUT_OF_TIME = "the run's time limit is reached"

_current: contextvars.ContextVar["Deadline | None"] = contextvars.ContextVar(
    "besser_run_deadline", default=None)


class Deadline:
    """``time_left`` gives the seconds to the run's cap (None before its clock
    starts); ``should_continue`` is the caller's own cancel flag, if any."""

    def __init__(self, time_left: Callable[[], "float | None"],
                 should_continue: "Callable[[], bool] | None" = None):
        self._time_left = time_left
        self._should_continue = should_continue
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._on_cancel: list[Callable[[], None]] = []
        # LLM calls this deadline stopped, and why the last one was.
        self.cuts = 0
        self.last_cut = ""

    def cancel(self) -> None:
        """Cancel the run and stop what it is running now. Thread-safe."""
        self._cancelled.set()
        with self._lock:
            callbacks = list(self._on_cancel)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.warning("stopping work on cancel failed", exc_info=True)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set() or (
            self._should_continue is not None and not self._should_continue())

    def remaining(self) -> "float | None":
        """Seconds to the cap: the most a command may run."""
        return self._time_left()

    def work_remaining(self) -> "float | None":
        """Seconds left for LLM calls and the agent's commands."""
        left = self._time_left()
        return None if left is None else left - PACKAGING_RESERVE_SECONDS

    def out_of_time(self) -> bool:
        """No LLM call may start any more."""
        left = self.work_remaining()
        return left is not None and left < MIN_CALL_SECONDS

    def checks_out_of_time(self) -> bool:
        """No validation round may start any more."""
        left = self.remaining()
        return left is not None and left < MIN_CHECK_SECONDS

    def reached(self, reason: str) -> DeadlineReached:
        """The error for an LLM call this deadline stops, counted in ``cuts``."""
        self.cuts += 1
        self.last_cut = reason
        return DeadlineReached(reason, cancelled=reason == CANCELLED)

    def stop_reason(self) -> str:
        """Why no LLM call may start, or ""."""
        if self.cancelled:
            return CANCELLED
        return OUT_OF_TIME if self.out_of_time() else ""

    def call_timeout(self, default: float) -> float:
        """The timeout of an LLM call about to start; raises :class:`DeadlineReached`
        instead when it may not start."""
        reason = self.stop_reason()
        if reason:
            raise self.reached(reason)
        left = self.work_remaining()
        return default if left is None else min(default, left)

    def command_stop_reason(self) -> str:
        """Why a command may not start or must stop now, or ""."""
        if self.cancelled:
            return CANCELLED
        left = self.remaining()
        return OUT_OF_TIME if left is not None and left < 1 else ""

    def command_timeout(self, default: float) -> float:
        left = self.remaining()
        return default if left is None else max(1.0, min(default, left))

    def wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; True when the run was cancelled meanwhile."""
        return self._cancelled.wait(seconds) or self.cancelled

    @contextlib.contextmanager
    def on_cancel(self, callback: Callable[[], None]) -> Iterator[None]:
        """Call ``callback`` if the run is cancelled while the block runs."""
        with self._lock:
            self._on_cancel.append(callback)
        try:
            if self._cancelled.is_set():
                callback()
            yield
        finally:
            with self._lock:
                self._on_cancel.remove(callback)


def current_deadline() -> "Deadline | None":
    """The deadline of the run this thread works for, or None outside a run."""
    return _current.get()


def under_deadline(method):
    """Run a run entry point with ``self.deadline`` current for its thread."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        token = _current.set(getattr(self, "deadline", None))
        try:
            return method(self, *args, **kwargs)
        finally:
            _current.reset(token)
    return wrapper
