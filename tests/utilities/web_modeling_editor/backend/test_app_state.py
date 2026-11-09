"""The app's state belongs to one app: built in its lifespan, gone with it.

Before, the spec-driven run registry, run slots, cancel events and thread pool,
the GitHub sign-in stores, the agent simulator's rate limiter and session
registry, and the KG consistency and SDMX import slots were module globals, so
every app (and every test) in a process shared them and nothing ever shut the
pool down.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
from fastapi import FastAPI

from besser.utilities.web_modeling_editor.backend import backend
from besser.utilities.web_modeling_editor.backend.routers import spec_driven_router
from besser.utilities.web_modeling_editor.backend.services.spec_driven import SpecDrivenRuntime
from besser.utilities.web_modeling_editor.backend.settings import get_settings

BASE_URL = "http://testserver"


@pytest.fixture
def quiet_lifespan(tmp_path, monkeypatch):
    """The real lifespan, minus the temp-dir cleanup, against a workspace in ``tmp_path``."""
    monkeypatch.setattr(backend, "cleanup_old_temp_files", lambda: None)
    monkeypatch.setattr(backend, "schedule_cleanup", lambda: asyncio.create_task(asyncio.sleep(3600)))
    monkeypatch.setattr(get_settings(), "llm_run_workspace_root", str(tmp_path / "runs"))
    monkeypatch.setattr(get_settings(), "llm_run_store_path", str(tmp_path / "runs.sqlite3"))
    monkeypatch.setattr(get_settings(), "oauth_sessions_path", str(tmp_path / "oauth_states.enc"))
    monkeypatch.setattr(get_settings(), "user_tokens_path", str(tmp_path / "user_tokens.enc"))


def _spec_driven_app() -> FastAPI:
    app = FastAPI(lifespan=backend.lifespan)
    app.include_router(spec_driven_router.router)
    return app


def test_lifespan_builds_the_runtime_and_shuts_its_pool_down(quiet_lifespan):
    app = _spec_driven_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            runtime = app.state.spec_driven
            thread = await runtime.run_blocking(lambda: threading.current_thread().name)
        return runtime, thread

    runtime, thread = asyncio.run(scenario())
    try:
        assert isinstance(runtime, SpecDrivenRuntime)
        assert thread.startswith("smartgen")
        assert runtime.executor._max_workers == get_settings().llm_max_concurrent_runs + 4
        with pytest.raises(RuntimeError, match="shutdown"):
            runtime.executor.submit(print)
    finally:
        runtime.durable_runs.store.close()


def test_two_apps_do_not_share_runs_or_slots(quiet_lifespan):
    first, second = _spec_driven_app(), _spec_driven_app()
    run_id = "c" * 32

    async def cancel(app: FastAPI) -> str:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.post(f"/besser_api/spec-driven/cancel/{run_id}")
        return response.json()["status"]

    async def scenario():
        async with first.router.lifespan_context(first), second.router.lifespan_context(second):
            one, two = first.state.spec_driven, second.state.spec_driven
            event = await one.active_runs.reserve(run_id)
            while one.slots.try_acquire():
                pass
            statuses = await cancel(second), await cancel(first)
            return one, two, event.is_set(), statuses, two.slots.try_acquire()

    one, two, cancelled, statuses, second_has_a_slot = asyncio.run(scenario())
    try:
        assert one is not two
        assert one.registry is not two.registry and one.executor is not two.executor
        # The run lives in the first app only: the second cannot cancel it.
        assert statuses == ("not_found", "cancelled")
        assert cancelled
        # Saturating the first app's slots leaves the second's untouched.
        assert second_has_a_slot
    finally:
        one.durable_runs.store.close()
        two.durable_runs.store.close()


def test_two_apps_do_not_share_sign_in_simulator_or_import_state(quiet_lifespan, tmp_path):
    first, second = _spec_driven_app(), _spec_driven_app()

    async def scenario():
        async with first.router.lifespan_context(first), second.router.lifespan_context(second):
            one, two = first.state, second.state
            while one.kg_consistency_slots.acquire(blocking=False):
                pass
            while one.sdmx_import_slots.acquire(blocking=False):
                pass
            session_id = one.agent_simulator.sessions.reserve("ip:1.2.3.4")
            return (
                one, two,
                two.kg_consistency_slots.acquire(blocking=False),
                two.sdmx_import_slots.acquire(blocking=False),
                two.agent_simulator.sessions.is_owner(session_id, "ip:1.2.3.4"),
            )

    one, two, kg_free, sdmx_free, second_knows_session = asyncio.run(scenario())
    try:
        assert one.github_sessions is not two.github_sessions
        assert one.agent_simulator is not two.agent_simulator
        # Built from the settings in the lifespan, not at import.
        assert one.github_sessions.user_tokens._store_path == tmp_path / "user_tokens.enc"
        assert one.github_sessions.oauth_states._store_path == tmp_path / "oauth_states.enc"
        # Saturating or using the first app's state leaves the second's untouched.
        assert kg_free and sdmx_free
        assert not second_knows_session
    finally:
        one.spec_driven.durable_runs.store.close()
        two.spec_driven.durable_runs.store.close()
