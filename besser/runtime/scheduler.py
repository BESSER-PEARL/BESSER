"""Scheduler — asyncio-based tick loop for Digital Twin platforms.

The scheduler owns the Run/Pause/Step lifecycle. When running, it fires once
every ``dt`` simulated seconds (wall-clock), applies input bindings, dispatches
``step(dt=dt)`` on every instance that has a step method, records a history
snapshot, and notifies any registered delta listeners (WebSocket subscribers).

Usage
-----
The scheduler is constructed once by the generated platform's
``runtime/__init__.py`` bootstrap and exposed as the module-level ``scheduler``
singleton. FastAPI startup/shutdown hooks call ``scheduler.start()`` /
``scheduler.pause()`` and the ``/api/runtime/run``, ``/api/runtime/pause``,
``/api/runtime/step``, ``/api/runtime/reset`` endpoints delegate to it.

Delta subscribers
-----------------
Register a coroutine via :meth:`add_listener` to receive per-tick deltas. The
WebSocket route in ``routers/runtime_control.py`` uses this to push live
updates to connected browsers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


# Type for delta-listener callbacks.
# Called with (tick_count: int, deltas: list[dict]) after each tick.
DeltaListener = Callable[[int, list[dict]], Awaitable[None]]


class Scheduler:
    """Asyncio tick loop for a generated Digital Twin platform.

    Parameters
    ----------
    engine:
        The :class:`besser.runtime.engine.Engine` instance wired to the
        generated platform's InstanceManager.
    history:
        A :class:`besser.runtime.history.HistoryStore` that receives snapshots
        after each tick.
    bindings:
        A :class:`besser.runtime.bindings.InputBindings` that drives input
        attributes before each tick.
    instance_manager:
        The platform's InstanceManager singleton; used to discover steppable
        instances and to read the pre/post attribute state for delta computation.
    steppable_classes:
        Mapping of ``{class_name: method_name}`` for classes whose tick-step
        method is flagged with ``is_step=True`` in the domain model.  Computed
        by the platform bootstrap and passed in at construction time so the
        scheduler doesn't need to import generator-specific code.  A plain
        ``set`` of class names is also accepted for backward compatibility
        (the method name defaults to ``"step"`` in that case).
    dt:
        Simulated time step in seconds (default 1.0).
    """

    def __init__(
        self,
        engine,
        history,
        bindings,
        instance_manager,
        steppable_classes,
        dt: float = 1.0,
    ):
        self._engine = engine
        self._history = history
        self._bindings = bindings
        self._im = instance_manager
        # Accept both dict[str, str] (class→method) and set[str] (legacy)
        if isinstance(steppable_classes, dict):
            self._steppable: dict[str, str] = dict(steppable_classes)
        else:
            self._steppable = {cn: "step" for cn in steppable_classes}
        self.dt: float = dt

        self.running: bool = False
        self.tick_count: int = 0
        self._task: asyncio.Task | None = None
        self._listeners: list[DeltaListener] = []

    # ----------------------------------------------------------------- public API

    async def start(self) -> None:
        """Start the tick loop (no-op if already running)."""
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._loop(), name="dt-scheduler")
        logger.info("Scheduler started (dt=%.2fs)", self.dt)

    async def pause(self) -> None:
        """Pause the tick loop (no-op if not running)."""
        if not self.running:
            return
        self.running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Scheduler paused at tick %d", self.tick_count)

    async def step_once(self) -> dict:
        """Execute exactly one tick and return a state snapshot.

        Useful for the **Step** button — advances the simulation by one tick
        without starting the continuous loop.
        """
        was_running = self.running
        if was_running:
            await self.pause()
        deltas = await self._tick()
        return {"tick_count": self.tick_count, "deltas": deltas}

    async def reset(self) -> None:
        """Pause, clear history, and reset tick_count to 0."""
        await self.pause()
        self._history.clear()
        self.tick_count = 0
        logger.info("Scheduler reset")

    def state(self) -> dict:
        """Return current scheduler state as a serialisable dict."""
        return {"running": self.running, "tick_count": self.tick_count, "dt": self.dt}

    # ----------------------------------------------------------------- listeners

    def add_listener(self, coro: DeltaListener) -> None:
        """Register a coroutine that is called with (tick_count, deltas) after each tick."""
        self._listeners.append(coro)

    def remove_listener(self, coro: DeltaListener) -> None:
        """Unregister a previously registered listener."""
        try:
            self._listeners.remove(coro)
        except ValueError:
            pass

    # ----------------------------------------------------------------- internals

    async def _loop(self) -> None:
        while self.running:
            try:
                await self._tick()
            except Exception as exc:
                logger.exception("Error during scheduler tick %d: %s", self.tick_count, exc)
            await asyncio.sleep(self.dt)

    async def _tick(self) -> list[dict]:
        """Execute one tick: bind → step → snapshot → notify. Returns deltas."""
        # 1. Drive input attributes
        self._bindings.apply_pre_tick(self.tick_count, self.dt)

        # 2. Dispatch step() on every steppable instance
        deltas: list[dict] = []
        all_instances = self._im.get_all_instances()
        for inst in all_instances:
            cn = inst["class_name"]
            if cn not in self._steppable:
                continue
            method_name = self._steppable[cn]
            before = dict(inst.get("attributes", {}))
            try:
                self._engine.tick(cn, inst["id"], self.dt, method_name=method_name)
            except Exception as exc:
                logger.warning("step() failed for %s/%s: %s", cn, inst["id"], exc)
                continue
            # Capture attribute mutations for delta notification
            after_inst = self._im.get_instance(cn, inst["id"])
            after = after_inst.get("attributes", {})
            for attr, new_val in after.items():
                if before.get(attr) != new_val:
                    deltas.append({
                        "class_name": cn,
                        "instance_id": inst["id"],
                        "attribute": attr,
                        "value": new_val,
                    })

        # 3. Persist snapshot
        self._history.snapshot(self.tick_count)

        # 4. Advance tick counter
        self.tick_count += 1

        # 5. Notify WebSocket listeners
        if self._listeners and deltas:
            for listener in list(self._listeners):
                try:
                    await listener(self.tick_count - 1, deltas)
                except Exception as exc:
                    logger.debug("Listener error: %s", exc)

        return deltas
