"""Shared ASGI test client for the backend endpoint tests.

Why not ``starlette.testclient.TestClient``: it forwards ``app=`` to
``httpx.Client``, and httpx removed that shortcut in 0.28 (it was deprecated
in 0.27, so the same code merely warned there). The versions are not free to
move -- ``bocl==1.0.1`` pins ``fastapi==0.110.0``, which pins
``starlette<0.37.0``, and starlette only switched its own ``TestClient`` over
to ``ASGITransport`` in 0.37.2. Driving the ASGI app through
``httpx.ASGITransport`` directly is therefore the only approach that works on
httpx 0.27 and 0.28 alike.

``test_api_integration.py`` builds its module-level client from the same class,
so there is exactly one implementation of this approach.

One behavioural note: ``ASGITransport`` does not run the app's lifespan, so
the periodic temp-directory cleanup task never starts. No endpoint under test
depends on it, and not starting a background task in a test process is the
better default anyway.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from besser.utilities.web_modeling_editor.backend.backend import app

BASE_URL = "http://testserver"


class AsgiClient:
    """Synchronous facade over ``httpx.AsyncClient`` bound to the ASGI app.

    Mirrors the slice of the ``TestClient`` API the endpoint tests use: the
    HTTP verb methods return an ``httpx.Response``, and ``json=``, ``data=``,
    ``files=``, ``headers=`` and ``params=`` pass straight through.

    A fresh ``AsyncClient`` per request keeps each call inside its own event
    loop, so the client carries no loop-bound state and can safely be shared
    across the whole session.
    """

    def __init__(self, application: Any = app, base_url: str = BASE_URL) -> None:
        self._app = application
        self._base_url = base_url

    async def _arequest(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(transport=transport, base_url=self._base_url) as ac:
            return await ac.request(method, url, **kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        return asyncio.run(self._arequest(method, url, **kwargs))

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


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
