"""The two server-side halves of surviving a TLS-inspecting corporate proxy.

Such a proxy buffers a response body before releasing it, and an SSE stream
never finishes producing one. Behind it the REST endpoints and the agent
WebSocket work, and *only* the spec-driven stream fails — either the initial POST's headers are held so the fetch never settles
("Waiting for the first event…" forever), or the connection is torn down and
the client reports "Failed to fetch".

Two server-side pieces make the client able to cope:

``events.json``       a short, terminating JSON read of the same durable event
                      log, sharing the stream's sequence cursor.
``Idempotency-Key``   makes retrying a failed run start safe, so a transport
                      error before the run id is known no longer strands the
                      user with a run they cannot rejoin.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _FakeOrchestrator,
    _FakeClient,
    _clear_registry,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_spec_driven_router import (
    BASE_URL,
    _build_project_body,
)


@pytest.fixture(autouse=True)
def reset_state():
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


@pytest.fixture
def stub_backend(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **kw: _FakeClient())


async def _run_once(body: dict, headers: dict | None = None) -> tuple[int, str | None]:
    """Drive one generate stream to completion; return (status, run id)."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        async with ac.stream(
            "POST", "/besser_api/spec-driven/generate", json=body, headers=headers or {}
        ) as response:
            run_id = response.headers.get("X-BESSER-Run-Id")
            async for _ in response.aiter_bytes():
                pass
            return response.status_code, run_id


class TestPollingTransport:
    """``events.json`` must carry the same events as the stream."""

    def test_returns_events_and_cursor(self, stub_backend):
        async def scenario():
            status, run_id = await _run_once(_build_project_body())
            assert status == 200
            assert run_id

            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                return await ac.get(
                    f"/besser_api/spec-driven/runs/{run_id}/events.json"
                )

        response = asyncio.run(scenario())
        assert response.status_code == 200
        body = response.json()
        # Plain JSON, not a stream: a buffering proxy can release it.
        assert response.headers["content-type"].startswith("application/json")
        assert body["events"], "polling returned no events"
        assert [e["event"] for e in body["events"]][0] == "start"
        assert body["events"][-1]["event"] in {"done", "error"}
        assert body["cursor"] == body["events"][-1]["sequence"]
        assert body["hasMore"] is False
        # Payloads are decoded, so one reducer can serve both transports.
        assert body["events"][0]["data"]["event"] == "start"

    def test_after_cursor_replays_only_newer_events(self, stub_backend):
        async def scenario():
            _, run_id = await _run_once(_build_project_body())
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                first = await ac.get(
                    f"/besser_api/spec-driven/runs/{run_id}/events.json"
                )
                mid = first.json()["events"][0]["sequence"]
                rest = await ac.get(
                    f"/besser_api/spec-driven/runs/{run_id}/events.json?after={mid}"
                )
                return first.json(), rest.json()

        first, rest = asyncio.run(scenario())
        assert len(rest["events"]) == len(first["events"]) - 1
        assert all(e["sequence"] > first["events"][0]["sequence"]
                   for e in rest["events"])

    def test_limit_sets_has_more(self, stub_backend):
        async def scenario():
            _, run_id = await _run_once(_build_project_body())
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                return (await ac.get(
                    f"/besser_api/spec-driven/runs/{run_id}/events.json?limit=1"
                )).json()

        body = asyncio.run(scenario())
        assert len(body["events"]) == 1
        assert body["hasMore"] is True

    def test_unknown_run_is_404(self):
        async def scenario():
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                return await ac.get(
                    "/besser_api/spec-driven/runs/" + "a" * 32 + "/events.json"
                )

        assert asyncio.run(scenario()).status_code == 404


class TestIdempotentStart:
    """A retried start must rejoin the first run, never open a second."""

    def test_same_key_reuses_the_run(self, stub_backend):
        async def scenario():
            body = _build_project_body()
            headers = {"Idempotency-Key": "probe-key-1"}
            first = await _run_once(body, headers)
            second = await _run_once(body, headers)
            return first, second

        (s1, id1), (s2, id2) = asyncio.run(scenario())
        assert s1 == s2 == 200
        assert id1 and id2
        assert id1 == id2, "retry started a duplicate run"

    def test_replay_after_reuse_is_complete(self, stub_backend):
        """The retry must see the whole run, not just what came after it."""
        async def scenario():
            body = _build_project_body()
            headers = {"Idempotency-Key": "probe-key-2"}
            await _run_once(body, headers)
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                async with ac.stream(
                    "POST", "/besser_api/spec-driven/generate",
                    json=body, headers=headers,
                ) as response:
                    raw = b""
                    async for chunk in response.aiter_bytes():
                        raw += chunk
                    return raw.decode("utf-8")

        text = asyncio.run(scenario())
        assert "event: start" in text
        assert "event: done" in text or "event: error" in text

    def test_different_keys_start_different_runs(self, stub_backend):
        async def scenario():
            body = _build_project_body()
            a = await _run_once(body, {"Idempotency-Key": "key-a"})
            b = await _run_once(body, {"Idempotency-Key": "key-b"})
            return a, b

        (_, id_a), (_, id_b) = asyncio.run(scenario())
        assert id_a != id_b

    def test_no_key_keeps_old_behaviour(self, stub_backend):
        async def scenario():
            body = _build_project_body()
            a = await _run_once(body)
            b = await _run_once(body)
            return a, b

        (_, id_a), (_, id_b) = asyncio.run(scenario())
        assert id_a != id_b, "unkeyed requests must not be deduplicated"


class TestPollingKeepsTheRunAlive:
    """Polling is presence: behind a proxy that tears the POST stream down,
    the client follows the run through ``events.json`` and must not have it
    cancelled as abandoned while it keeps polling."""

    def test_polled_run_is_not_abandoned_until_polling_stops(self):
        from types import SimpleNamespace

        from besser.utilities.web_modeling_editor.backend.routers import spec_driven_router
        from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
            DurableRunManager,
            ErrorEvent,
            SqliteRunEventStore,
            format_sse,
        )

        grace = 0.3

        async def scenario():
            store = SqliteRunEventStore(":memory:")
            manager = DurableRunManager(store)
            runtime = SimpleNamespace(durable_runs=manager)
            run_id = "c" * 32
            cancel = asyncio.Event()

            async def source():
                await cancel.wait()
                yield format_sse(ErrorEvent(code="CANCELLED", message="cancelled"))

            await manager.start(
                run_id, source(), on_abandoned=cancel.set, disconnect_grace_seconds=grace,
            )
            loop = asyncio.get_running_loop()
            polling_until = loop.time() + 3 * grace
            while loop.time() < polling_until:
                await spec_driven_router.poll_smart_run_events(runtime, run_id, after=0, limit=250)
                await asyncio.sleep(grace / 5)
            polled_status = manager.get_run(run_id)

            # Polling stopped: the grace period now runs out and the run stops.
            await asyncio.wait_for(manager.wait(run_id), 10 * grace)
            final = manager.get_run(run_id)
            store.close()
            return polled_status, final

        polled_status, final = asyncio.run(scenario())
        assert polled_status.abandonment_requested_at is None, "a polled run was abandoned"
        assert polled_status.status == "running"
        assert final.status == "abandoned"
