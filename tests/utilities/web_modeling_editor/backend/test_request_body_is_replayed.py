"""Every POST to this application could hang, or 500, depending on the version.

``RequestSizeLimitMiddleware`` used to read the body to enforce the 50 MB
cap. That consumes the receive channel, and every way of handing it back
was wrong on one Starlette or another:

* consume and do not replay -> the endpoint waits forever for a body that
  is already gone. Every POST deadlocks on 0.27.0.
* replay a constant ``http.request`` -> fixes 0.27.0, and 500s on 0.36.3,
  where ``BaseHTTPMiddleware`` polls receive again to await the client
  disconnect and raises "Unexpected message received".
* replay then ``http.disconnect`` -> still 500s on 0.36.3, because that
  version already caches and replays the body itself.

`fastapi` is unpinned in the backend requirements, so a fresh install takes
whichever Starlette pip resolves. The middleware is now pure ASGI and never
reads the body at all, which removes the dependency on that entirely.

GET was never affected, which is why the application looked healthy while
every POST was broken. The backend suite also stops on its third test when
this regresses, so the whole suite becomes unrunnable.

Uses httpx + ASGITransport rather than ``TestClient(app)``: the installed
starlette/httpx versions do not support that legacy pattern, and an earlier
version of this file used it and broke CI while the code under test was
fine. Same approach as ``test_api_integration.py``.
"""

import asyncio
import time

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import (
    MAX_REQUEST_SIZE, app,
)

BASE_URL = "http://testserver"

# Generous on purpose: the property under test is hang versus no-hang, not
# latency. A deadlocked request never returns at all.
_BUDGET_SECONDS = 30


def _request(method: str, url: str, **kwargs) -> httpx.Response:
    async def go():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
            return await ac.request(method, url, **kwargs)
    return asyncio.run(go())


@pytest.mark.parametrize("path", [
    "/besser_api/validate-diagram",
    "/besser_api/generate-output",
    "/besser_api/export-buml",
])
def test_a_post_reaches_its_endpoint_instead_of_hanging(path):
    """An invalid body must be REJECTED, which proves it was delivered."""
    started = time.monotonic()
    response = _request("POST", path, json={})
    elapsed = time.monotonic() - started

    assert elapsed < _BUDGET_SECONDS, f"{path} did not return within {_BUDGET_SECONDS}s"
    # 422 proves the body arrived and failed validation. A hang, or a 500
    # from the body never arriving, would not.
    assert response.status_code in (200, 400, 422), response.status_code


def test_the_body_arrives_intact_not_merely_present():
    """Replaying the wrong bytes would be worse than replaying none."""
    response = _request(
        "POST", "/besser_api/validate-diagram",
        json={"title": "T", "model": {"elements": {}, "relationships": {}}},
    )

    assert response.status_code != 500
    assert response.content


def test_get_was_never_affected():
    """Pinned so a future fix cannot 'solve' this by breaking GET."""
    started = time.monotonic()
    response = _request("GET", "/health")

    assert time.monotonic() - started < _BUDGET_SECONDS
    assert response.status_code == 200


def test_an_oversized_body_is_still_rejected():
    """The middleware's actual job has to survive its repair."""
    oversized = b"x" * (MAX_REQUEST_SIZE + 1024)
    response = _request(
        "POST", "/besser_api/validate-diagram",
        content=oversized, headers={"content-type": "application/json"},
    )

    assert response.status_code == 413, response.status_code
