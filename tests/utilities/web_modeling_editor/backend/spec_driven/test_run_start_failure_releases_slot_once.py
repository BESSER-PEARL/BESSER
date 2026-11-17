"""A store failure while a run starts returns its slot exactly once.

``DurableRunManager.start`` wrote the subscriber count after creating the
producer task. When that write failed, the router released the run's slot, and
the already-running task released it a second time when its stream ended,
freeing a slot another run still held (and the run went on although its start
had answered with an error).
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3

from besser.utilities.web_modeling_editor.backend.routers import (
    spec_driven_router as router_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.settings import get_settings
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _build_request,
    _FakeClient,
    _FakeOrchestrator,
)


def test_subscriber_count_failure_at_start_releases_the_slot_once(monkeypatch, spec_driven_runtime):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    runtime = spec_driven_runtime
    store = runtime.durable_runs.store

    def set_subscriber_count(run_id, count):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "set_subscriber_count", set_subscriber_count)

    async def exercise() -> None:
        assert runtime.slots.try_acquire(), "a slot held by another run"
        with contextlib.suppress(sqlite3.OperationalError):
            await router_module.smart_generate(
                _build_request(), runtime, get_settings().model_copy(), idempotency_key=None,
            )
        started = list(runtime.durable_runs._tasks.values())
        for task in started:
            await asyncio.wait({task}, timeout=10)
        assert runtime.slots._held == 1, "the other run's slot was released"
        assert runtime.active_runs._events == {}
        assert started == [], "a run whose start failed still ran"

    asyncio.run(exercise())
