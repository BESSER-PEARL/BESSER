"""Shared fixtures for the backend endpoint tests.

The HTTP client lives in ``_asgi_client.py``; see its module docstring for why
it is not ``starlette.testclient.TestClient``.
"""

from __future__ import annotations

import pytest

from tests.utilities.web_modeling_editor.backend._asgi_client import AsgiClient


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
