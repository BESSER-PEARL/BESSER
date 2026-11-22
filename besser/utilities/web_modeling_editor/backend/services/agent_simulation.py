"""Rate limiting and session ownership of the agent simulator endpoints.

``routers/agent_simulator_router.py`` serves the endpoints; the state here
belongs to one app: the backend's lifespan builds an :class:`AgentSimulatorState`
and stores it on ``app.state.agent_simulator``. It lives in the process, so
running several uvicorn workers would give each worker its own copy (the
backend runs one, see ``backend.py`` and the Dockerfile ``CMD``).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict, Optional, Set, Tuple

from fastapi import HTTPException

from besser.utilities.web_modeling_editor.backend.constants.constants import (
    DEFAULT_AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS,
)
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings, get_settings


class SlidingWindowRateLimiter:
    """Per-key sliding-window limiter with bounded memory.

    Expired timestamps are dropped on every check, idle keys are swept once per
    window, and at most ``max_keys`` keys are tracked (least recently used keys
    are evicted first).
    """

    def __init__(self, window_seconds: int, max_requests: int, max_keys: int):
        self._window_seconds = max(window_seconds, 1)
        self._max_requests = max(max_requests, 1)
        self._max_keys = max(max_keys, 1)
        self._events: "OrderedDict[str, Deque[float]]" = OrderedDict()
        self._last_sweep = time.monotonic()
        self._lock = Lock()

    def __len__(self) -> int:
        return len(self._events)

    def _sweep(self, threshold: float) -> None:
        for key in [k for k, events in self._events.items() if not events or events[-1] < threshold]:
            del self._events[key]

    def check(self, key: str) -> None:
        """Record one request for ``key``; raise HTTP 429 when over the limit."""
        now = time.monotonic()
        threshold = now - self._window_seconds
        with self._lock:
            if now - self._last_sweep >= self._window_seconds:
                self._sweep(threshold)
                self._last_sweep = now

            events = self._events.get(key)
            if events is None:
                if len(self._events) >= self._max_keys:
                    self._sweep(threshold)
                while len(self._events) >= self._max_keys:
                    self._events.popitem(last=False)
                events = deque()
                self._events[key] = events
            else:
                self._events.move_to_end(key)
                while events and events[0] < threshold:
                    events.popleft()

            if len(events) >= self._max_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded for agent simulator endpoints. Please retry shortly.",
                    headers={"Retry-After": str(self._window_seconds)},
                )
            events.append(now)


class SimulationSessionRegistry:
    """Maps each simulation session to the actor that created it.

    Also enforces the per-actor concurrent-session cap. Entries are forgotten
    when the session is stopped, when its WebSocket closes, or after the
    simulator's session lifetime.
    """

    def __init__(self, max_sessions_per_actor: int, ttl_seconds: int):
        self._max_sessions_per_actor = max(max_sessions_per_actor, 1)
        self._ttl_seconds = max(ttl_seconds, 1)
        self._sessions: Dict[str, Tuple[str, float]] = {}
        self._lock = Lock()

    def __len__(self) -> int:
        return len(self._sessions)

    def _prune(self, now: float) -> None:
        expired = [sid for sid, (_, created) in self._sessions.items() if now - created >= self._ttl_seconds]
        for sid in expired:
            del self._sessions[sid]

    def reserve(self, actor: str) -> str:
        """Allocate a new session id for ``actor``; raise HTTP 429 when at the cap."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            active = sum(1 for owner, _ in self._sessions.values() if owner == actor)
            if active >= self._max_sessions_per_actor:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"You already have {active} active agent simulation session(s), the maximum allowed. "
                        "Stop it before starting a new one."
                    ),
                )
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = (actor, now)
            return session_id

    def release(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def is_owner(self, session_id: str, actor: str) -> bool:
        with self._lock:
            self._prune(time.monotonic())
            entry = self._sessions.get(session_id)
            return entry is not None and entry[0] == actor

    def require_owner(self, session_id: str, actor: str) -> None:
        """Raise HTTP 404 unless ``actor`` created ``session_id``."""
        if not self.is_owner(session_id, actor):
            raise HTTPException(status_code=404, detail="Simulation session not found.")


@dataclass
class AgentSimulatorState:
    """The agent simulator endpoints' rate limiter, session registry and background tasks."""

    rate_limiter: SlidingWindowRateLimiter
    sessions: SimulationSessionRegistry
    #: Strong references to fire-and-forget tasks, so they are not garbage-collected
    #: before they finish (see the ``asyncio.create_task`` documentation).
    background_tasks: Set["asyncio.Task[None]"] = field(default_factory=set)

    @classmethod
    def create(cls, settings: Optional[BackendSettings] = None) -> "AgentSimulatorState":
        settings = settings or get_settings()
        return cls(
            rate_limiter=SlidingWindowRateLimiter(
                window_seconds=settings.agent_simulator_rate_limit_window_seconds,
                max_requests=settings.agent_simulator_rate_limit_max_requests,
                max_keys=settings.agent_simulator_rate_limit_max_keys,
            ),
            sessions=SimulationSessionRegistry(
                max_sessions_per_actor=settings.agent_simulator_max_sessions_per_actor,
                ttl_seconds=(settings.agent_simulator_session_lifetime_seconds
                             or DEFAULT_AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS),
            ),
        )

    def spawn(self, coro) -> None:
        """Run ``coro`` in the background while holding a reference to its task."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
