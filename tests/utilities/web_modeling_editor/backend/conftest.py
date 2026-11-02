"""Shared fixtures for the backend endpoint tests.

The HTTP client lives in ``_asgi_client.py``; see its module docstring for why
it is not ``starlette.testclient.TestClient``.
"""

from __future__ import annotations

import pytest

from besser.utilities.web_modeling_editor.backend.backend import app
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
