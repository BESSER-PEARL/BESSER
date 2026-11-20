"""The preview's key-holding LLM proxy, served by ``serve_llm_proxy`` on a free loopback port.

The upstream is an ``httpx.MockTransport``: nothing leaves the machine. The model-written
agent in the preview can spend the key through the proxy but must never see it.
"""

import asyncio
import contextlib
import logging
import signal
import socket

import httpx
import pytest
from pydantic import SecretStr

from besser.utilities.web_modeling_editor.backend.services.lab import llm_proxy
from besser.utilities.web_modeling_editor.backend.services.lab.llm_proxy import (
    PREVIEW_PROXY_PLACEHOLDER_KEY,
    PROXIED_PATHS,
    create_llm_proxy_app,
    serve_llm_proxy,
)

KEY = "sk-proj-preview-0123456789abcdef"
BASE_URL = "https://upstream.test/v1/"
CALLER_SECRET = "sk-caller-own-key-fedcba9876543210"
REQUEST_BODY = b'{"model": "gpt-5", "messages": [{"role": "user", "content": "REQUEST-BODY-MARKER"}]}'
REPLY_BODY = b'{"choices": [{"message": {"content": "REPLY-BODY-MARKER"}}]}'
#: How long a test waits for the proxy to listen or for a chunk to arrive.
WAIT_SECONDS = 10


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.asynccontextmanager
async def _proxy(handler, *, max_requests_per_day=100):
    """A client of a running proxy whose upstream is ``handler``."""
    app = create_llm_proxy_app(api_key=SecretStr(KEY), base_url=BASE_URL,
                               max_requests_per_day=max_requests_per_day, transport=httpx.MockTransport(handler))
    port = _free_port()
    serving = asyncio.create_task(serve_llm_proxy(app, port))
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=WAIT_SECONDS,
                                 trust_env=False) as client:
        try:
            await _until_listening(client, serving, app)
            yield client
        finally:
            serving.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await serving


async def _until_listening(client: httpx.AsyncClient, serving: asyncio.Task, app) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + WAIT_SECONDS
    # The port opens right after the app's startup; on Windows a refused connect costs about a second.
    while not hasattr(app.state, "upstream") and not serving.done() and loop.time() < deadline:
        await asyncio.sleep(0.01)
    while True:
        await asyncio.sleep(0.02)
        if serving.done():
            serving.result()  # raises what stopped it
        try:
            await client.get("/")  # outside /v1/: a 404 that the cap does not count
            return
        except httpx.TransportError:
            if loop.time() > deadline:
                raise
            await asyncio.sleep(0.05)


async def _streamed(data: bytes):
    # A network reply's body is a stream; bytes content would be read already, unlike a real reply.
    yield data


def _reply(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=_streamed(REPLY_BODY), headers={"content-type": "application/json"})


def _recording(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request, request.read()))
        return _reply(request)
    return handler


def test_upstream_gets_the_key_and_never_the_callers_credentials():
    seen = []

    async def go():
        async with _proxy(_recording(seen)) as client:
            return await client.post("/v1/chat/completions?api-version=2", content=REQUEST_BODY, headers={
                "Authorization": f"Bearer {CALLER_SECRET}", "OpenAI-Project": "proj_other",
                "Content-Type": "application/json"})

    response = asyncio.run(go())

    assert (response.status_code, response.content) == (200, REPLY_BODY)
    ((request, body),) = seen
    assert str(request.url) == "https://upstream.test/v1/chat/completions?api-version=2"
    assert request.headers.get_list("authorization") == [f"Bearer {KEY}"]
    assert all(CALLER_SECRET not in value for value in request.headers.values())
    assert "openai-project" not in request.headers
    assert (body, request.headers["content-type"]) == (REQUEST_BODY, "application/json")


def test_every_proxied_path_is_forwarded_with_the_placeholder_key():
    seen = []

    async def go():
        async with _proxy(_recording(seen)) as client:
            return [(await client.post(f"/v1/{path}", content=b"{}",
                                       headers={"Authorization": f"Bearer {PREVIEW_PROXY_PLACEHOLDER_KEY}"})).status_code
                    for path in sorted(PROXIED_PATHS)]

    assert asyncio.run(go()) == [200] * len(PROXIED_PATHS)
    assert sorted(request.url.path for request, _ in seen) == sorted(f"/v1/{path}" for path in PROXIED_PATHS)
    assert {request.headers["authorization"] for request, _ in seen} == {f"Bearer {KEY}"}


@pytest.mark.parametrize("method, path", [("GET", "/v1/models"), ("POST", "/v1/files"), ("GET", "/v1/files"),
                                          ("GET", "/v1/chat/completions"), ("POST", "/v1/chat/completions/x")])
def test_other_paths_and_methods_answer_404_and_reach_nothing(method, path):
    seen = []

    async def go():
        async with _proxy(_recording(seen)) as client:
            return await client.request(method, path)

    response = asyncio.run(go())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_proxied"
    assert seen == []


def test_the_request_past_the_days_cap_answers_429_until_the_next_utc_day(monkeypatch):
    seen = []
    monkeypatch.setattr(llm_proxy, "_utc_day", lambda: "2026-11-20")

    async def go():
        async with _proxy(_recording(seen), max_requests_per_day=2) as client:
            statuses = [(await client.post("/v1/embeddings", content=b"{}")).status_code for _ in range(3)]
            refused = await client.post("/v1/embeddings", content=b"{}")
            monkeypatch.setattr(llm_proxy, "_utc_day", lambda: "2026-11-21")
            next_day = await client.post("/v1/embeddings", content=b"{}")
            return statuses, refused, next_day

    statuses, refused, next_day = asyncio.run(go())
    assert statuses == [200, 200, 429]
    assert refused.status_code == 429 and refused.json()["error"]["code"] == "daily_limit"
    assert next_day.status_code == 200
    assert len(seen) == 3  # the refused requests never reached the upstream


def test_a_streamed_reply_passes_through_chunk_by_chunk():
    async def go():
        second_chunk_may_go = asyncio.Event()

        async def events():
            yield b"data: first\n\n"
            await second_chunk_may_go.wait()
            yield b"data: second\n\n"

        def handler(request):
            return httpx.Response(200, content=events(), headers={"content-type": "text/event-stream"})

        async with _proxy(handler) as client:
            async with client.stream("POST", "/v1/responses", content=b"{}") as response:
                chunks = response.aiter_raw()
                # Only arrives before the upstream has finished if the proxy does not buffer the reply.
                first = await asyncio.wait_for(chunks.__anext__(), WAIT_SECONDS)
                second_chunk_may_go.set()
                rest = b"".join([chunk async for chunk in chunks])
                return response, first, rest

    response, first, rest = asyncio.run(go())
    assert response.headers["transfer-encoding"] == "chunked"
    assert response.headers["content-type"] == "text/event-stream"
    assert (first, rest) == (b"data: first\n\n", b"data: second\n\n")


def test_an_unreachable_upstream_answers_502():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    async def go():
        async with _proxy(handler) as client:
            return await client.post("/v1/chat/completions", content=b"{}")

    response = asyncio.run(go())
    assert response.status_code == 502 and response.json()["error"]["code"] == "upstream_unreachable"


def test_logs_name_the_path_and_status_and_never_the_key_or_a_body(caplog):
    def failing(request):
        raise httpx.ReadTimeout(f"timed out {KEY}", request=request)

    async def go():
        async with _proxy(_reply) as client:
            await client.post("/v1/chat/completions", content=REQUEST_BODY,
                              headers={"Authorization": f"Bearer {CALLER_SECRET}"})
        async with _proxy(failing) as client:
            await client.post("/v1/chat/completions", content=REQUEST_BODY)

    with caplog.at_level(logging.DEBUG):
        asyncio.run(go())

    assert "/v1/chat/completions -> 200" in caplog.text
    assert "/v1/chat/completions -> 502" in caplog.text
    for secret in (KEY, CALLER_SECRET, "REQUEST-BODY-MARKER", "REPLY-BODY-MARKER"):
        assert secret not in caplog.text


def test_the_backends_signal_handlers_stay_and_a_cancel_closes_the_proxy():
    # Inside the backend's lifespan: uvicorn's own handlers would take SIGINT/SIGTERM from the backend's server.
    app = create_llm_proxy_app(api_key=SecretStr(KEY), base_url=BASE_URL, max_requests_per_day=1,
                               transport=httpx.MockTransport(_reply))
    port = _free_port()

    async def go():
        before = signal.getsignal(signal.SIGINT)
        serving = asyncio.create_task(serve_llm_proxy(app, port))
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
            await _until_listening(client, serving, app)
            during = signal.getsignal(signal.SIGINT)
        serving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await serving
        # Checked while the backend's loop still runs: closing the loop would close both anyway.
        closed = app.state.upstream.is_closed  # the app's shutdown ran
        with socket.socket() as probe:
            refused = probe.connect_ex(("127.0.0.1", port)) != 0
        return before, during, closed, refused

    before, during, closed, refused = asyncio.run(go())
    assert during is before
    assert closed and refused


def test_a_port_in_use_is_an_error_not_an_exit():
    app = create_llm_proxy_app(api_key=SecretStr(KEY), base_url=BASE_URL, max_requests_per_day=1)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        with pytest.raises(RuntimeError, match="could not listen"):
            asyncio.run(serve_llm_proxy(app, taken.getsockname()[1]))
