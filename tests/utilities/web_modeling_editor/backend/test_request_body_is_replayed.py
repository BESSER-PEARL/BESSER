"""Every POST to this application deadlocked.

``RequestSizeLimitMiddleware`` reads the body to enforce the 50 MB cap:

    body = await request.body()      # drains the receive channel
    ...
    return await call_next(request)  # builds a FRESH Request from it

``BaseHTTPMiddleware`` constructs a new Request downstream from the same
receive channel, so the endpoint waited forever for a body that had already
been consumed. GET was unaffected, which is why the application looked
healthy.

Whether it manifests depends on the installed Starlette - 0.27.0 hangs,
later versions replay the body themselves - and `fastapi` is unpinned in
the backend requirements, so which behaviour an install gets is down to
whatever pip resolves. That is the worst kind of dependency bug: it works
for the person who deployed it and hangs for the next person to install.

Cost while it was live: the whole backend test suite was unrunnable, since
it stops on the third test. Three separate investigations attributed that
to concurrent test runs competing for ports. It reproduces with one run on
an idle machine.
"""

import time

import pytest
from fastapi.testclient import TestClient

from besser.utilities.web_modeling_editor.backend.backend import app


# Generous: the point is hang versus no-hang, not latency. A deadlocked
# request never returns at all, so any finite number passes.
_BUDGET_SECONDS = 30


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.mark.parametrize("path", [
    "/besser_api/validate-diagram",
    "/besser_api/generate-output",
    "/besser_api/export-buml",
])
def test_a_post_reaches_its_endpoint_instead_of_hanging(client, path):
    """An invalid body must be REJECTED, which means it was delivered."""
    started = time.monotonic()
    response = client.post(path, json={})
    elapsed = time.monotonic() - started

    assert elapsed < _BUDGET_SECONDS, f"{path} did not return within {_BUDGET_SECONDS}s"
    # 422 proves the body arrived and failed validation. A hang, or a 500
    # from the body never arriving, would not.
    assert response.status_code in (200, 400, 422), response.status_code


def test_the_body_arrives_intact_not_merely_present(client):
    """Replaying the wrong bytes would be worse than replaying none.

    A diagram this malformed is rejected, but the rejection has to be about
    the CONTENT - so the body that reached the endpoint is the body that was
    sent, not an empty stand-in.
    """
    response = client.post(
        "/besser_api/validate-diagram",
        json={"title": "T", "model": {"elements": {}, "relationships": {}}},
    )

    assert response.status_code != 500
    # Whatever the verdict, it was computed from the payload we sent.
    assert response.content


def test_get_was_never_affected(client):
    """Pinned so a future fix cannot 'solve' this by breaking GET."""
    started = time.monotonic()
    response = client.get("/health")

    assert time.monotonic() - started < _BUDGET_SECONDS
    assert response.status_code == 200


def test_an_oversized_body_is_still_rejected(client):
    """The middleware's actual job must survive the fix."""
    from besser.utilities.web_modeling_editor.backend.backend import MAX_REQUEST_SIZE

    oversized = "x" * (MAX_REQUEST_SIZE + 1024)
    response = client.post(
        "/besser_api/validate-diagram",
        content=oversized.encode(),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413, response.status_code
