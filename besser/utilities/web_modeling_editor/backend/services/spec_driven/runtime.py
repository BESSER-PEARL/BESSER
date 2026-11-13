"""Process state of the spec-driven runs: one ``SpecDrivenRuntime`` per app.

The backend's lifespan builds it and stores it on ``app.state.spec_driven``;
routes receive it through ``dependencies.get_spec_driven_runtime`` and pass it
to each :class:`~.runner.SmartGenerationRunner`. Nothing here is a module
global, so two apps in one process (or two tests) never share runs, slots or
threads.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from besser.utilities.web_modeling_editor.backend.services.spec_driven.run_manager import (
    DurableRunManager,
    SqliteRunEventStore,
    default_store_path,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartRunRegistry,
    _locate_run_temp_dir,
)
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings, get_settings

logger = logging.getLogger(__name__)


class RunSlots:
    """The cap on spec-driven runs and GitHub imports in flight.

    Callers take a slot without waiting and answer HTTP 429 when none is left:
    a queued SSE connection would just look hung in the browser.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._held = 0

    def try_acquire(self) -> bool:
        """Take a slot if one is free; the caller must ``release`` it later."""
        # Called on the event loop thread only, with no await in between.
        if self._held >= self._limit:
            return False
        self._held += 1
        return True

    def release(self) -> None:
        """Return a slot; a release with none held is logged and ignored."""
        if self._held <= 0:
            logger.warning("RunSlots.release called with no slot held; ignored")
            return
        self._held -= 1


class ActiveRuns:
    """``run_id`` -> cancel event of each live run.

    The runner's orchestrator polls the event between turns; closing the SSE
    stream alone would not stop a run that is spending the user's key.
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def reserve(self, run_id: str) -> Optional[asyncio.Event]:
        """Reserve ``run_id`` for one live runner; None when it is already taken."""
        event = asyncio.Event()
        async with self._lock:
            if run_id in self._events:
                return None
            self._events[run_id] = event
        return event

    async def release(self, run_id: str, event: asyncio.Event) -> bool:
        """Release ``run_id`` only when ``event`` still owns it."""
        async with self._lock:
            if self._events.get(run_id) is not event:
                return False
            self._events.pop(run_id, None)
            return True

    async def request_cancellation(self, run_id: str) -> bool:
        """Signal a live run to stop at its next turn; False when no such run is live."""
        async with self._lock:
            event = self._events.get(run_id)
        if event is None:
            return False
        event.set()
        return True


@dataclass
class SpecDrivenRuntime:
    """Everything the spec-driven routes and runs share within one app."""

    registry: SmartRunRegistry
    durable_runs: DurableRunManager
    slots: RunSlots
    active_runs: ActiveRuns
    # Long runs stay off asyncio's default pool (``min(32, cpu + 4)`` threads),
    # which the quick ``to_thread`` endpoints (conversion, exports) also use.
    executor: ThreadPoolExecutor
    #: ``Idempotency-Key`` -> ``(run_id, created_at)`` of recent run starts.
    idempotent_runs: dict[str, tuple[str, float]] = field(default_factory=dict)
    idempotency_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    def create(
        cls,
        settings: Optional[BackendSettings] = None,
        *,
        store: Optional[SqliteRunEventStore] = None,
    ) -> "SpecDrivenRuntime":
        """Build the runtime; ``store`` defaults to the SQLite run store ``settings`` names."""
        settings = settings or get_settings()
        cap = settings.llm_max_concurrent_runs
        return cls(
            registry=SmartRunRegistry(),
            durable_runs=DurableRunManager(store or SqliteRunEventStore(
                default_store_path(settings),
                has_checkpoint=lambda run_id: _locate_run_temp_dir(run_id) is not None,
            )),
            slots=RunSlots(cap),
            active_runs=ActiveRuns(),
            # A little above the cap, so every admitted run gets its own thread.
            executor=ThreadPoolExecutor(max_workers=cap + 4, thread_name_prefix="smartgen"),
        )

    async def run_blocking(self, func, /, *args, **kwargs):
        """``asyncio.to_thread`` on this runtime's own pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, functools.partial(func, *args, **kwargs))

    def close(self) -> None:
        """Stop taking new work; threads still running finish at interpreter exit."""
        self.executor.shutdown(wait=False)
