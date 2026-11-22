"""The spec-driven runtime of the backend app, for tests.

``httpx.ASGITransport`` does not run the app's lifespan, so the
``spec_driven_runtime`` fixture in ``conftest.py`` installs a fresh runtime on
``app.state`` for every test, where the lifespan would. Tests that drive a
runner directly or seed the registry reach that same runtime here.
"""

from __future__ import annotations

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    SpecDrivenRuntime,
    SqliteRunEventStore,
)


def make_runtime() -> SpecDrivenRuntime:
    """A runtime with an in-memory run store, so tests never share run history."""
    return SpecDrivenRuntime.create(store=SqliteRunEventStore(":memory:"))


def current_runtime() -> SpecDrivenRuntime:
    """The runtime the backend app is using in this test."""
    return app.state.spec_driven
