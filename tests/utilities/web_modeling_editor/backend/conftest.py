"""Shared fixtures for the backend endpoint tests.

The HTTP client lives in ``_asgi_client.py``; see its module docstring for why
it is not ``starlette.testclient.TestClient``.
"""

from __future__ import annotations

import pytest

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.agent_simulation import AgentSimulatorState
from besser.utilities.web_modeling_editor.backend.services.deployment.session_store import (
    GitHubSessions,
    SessionStore,
)
from besser.utilities.web_modeling_editor.backend.services.kg.validation import consistency_check_slots
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering import sdmx_import_slots
from besser.utilities.web_modeling_editor.backend.services.validators.alloy.semantic_consistency import (
    semantic_check_slots,
)
from besser.utilities.web_modeling_editor.backend.settings import get_settings
from tests.utilities.web_modeling_editor.backend._asgi_client import AsgiClient
from tests.utilities.web_modeling_editor.backend._runtime import make_runtime


@pytest.fixture(autouse=True)
def spec_driven_runtime():
    """A fresh spec-driven runtime on ``app.state`` per test, as the lifespan would build.

    The client below does not run the lifespan, and a runtime shared across
    tests would leak runs, slots and cancel events from one test into the next.
    """
    runtime = make_runtime()
    app.state.spec_driven = runtime
    yield runtime
    runtime.close()
    runtime.durable_runs.store.close()
    del app.state.spec_driven


@pytest.fixture(autouse=True)
def app_state(tmp_path_factory):
    """The rest of the per-app state the lifespan builds, fresh for each test.

    GitHub sign-in stores in a directory of their own (tests list ``tmp_path``),
    the agent simulator's limiter and sessions, and the KG consistency, SDMX
    import and semantic check slots.
    """
    state = app.state
    stores = tmp_path_factory.mktemp("github_sessions")
    state.github_sessions = GitHubSessions(
        oauth_states=SessionStore(store_path=str(stores / "oauth_states.enc"), ttl=600),
        user_tokens=SessionStore(store_path=str(stores / "user_tokens.enc"), ttl=86400),
    )
    state.agent_simulator = AgentSimulatorState.create()
    state.kg_consistency_slots = consistency_check_slots()
    state.sdmx_import_slots = sdmx_import_slots()
    state.semantic_check_slots = semantic_check_slots()
    yield state
    for name in ("github_sessions", "agent_simulator", "kg_consistency_slots", "sdmx_import_slots",
                 "semantic_check_slots"):
        delattr(state, name)


@pytest.fixture
def route_settings():
    """A copy of the process settings that the routes receive (``SettingsDep``); set fields on it."""
    settings = get_settings().model_copy()
    app.dependency_overrides[get_settings] = lambda: settings
    yield settings
    app.dependency_overrides.pop(get_settings, None)


@pytest.fixture(autouse=True)
def isolate_backend_test_artifacts(tmp_path, monkeypatch):
    """Keep generated test artifacts out of the repository root.

    Applies to every test in this directory. It used to live only in
    ``test_api_integration.py``, so the endpoint tests added alongside the KG
    work ran with the repository as their working directory.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="session")
def client() -> AsgiClient:
    """The backend app, callable over HTTP.

    Session-scoped because the client is stateless; module-scoped fixtures in
    the endpoint tests depend on it, which a function-scoped fixture could not
    satisfy.
    """
    return AsgiClient()
