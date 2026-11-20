"""The key-holding LLM proxy of the lab's preview.

The preview's modeling agent runs model-written code in a sandbox, so it must be
able to spend the preview's OpenAI key without being able to read it. It gets
``OPENAI_BASE_URL=http://127.0.0.1:<port>/v1`` and :data:`PREVIEW_PROXY_PLACEHOLDER_KEY`;
this proxy, served by the lab backend outside the sandbox, replaces the caller's
``Authorization`` with the real key, forwards only :data:`PROXIED_PATHS` and caps
the requests of each UTC day. It logs each request's path and status, never a
header or a body.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Iterator

import httpx
import uvicorn
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

#: The OpenAI endpoints the modeling agent calls (chat, responses, embeddings, voice input).
PROXIED_PATHS = frozenset({"chat/completions", "responses", "embeddings", "audio/transcriptions"})

#: The key the preview's agent is given; the proxy never checks or forwards it.
PREVIEW_PROXY_PLACEHOLDER_KEY = "lab-preview-proxy"

#: Only the host may call the proxy: sandboxes reach it through pasta's ``-T <port>``.
PROXY_HOST = "127.0.0.1"

#: As the OpenAI SDK's own default: a reasoning model may send nothing for minutes.
UPSTREAM_TIMEOUT = httpx.Timeout(600.0, connect=10.0)

#: Open streams get this long to finish when the backend stops, then are cut.
SHUTDOWN_GRACE_SECONDS = 5

#: Hop-by-hop headers (RFC 9110 7.6.1) belong to one connection, never forwarded.
_HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
                         "trailer", "transfer-encoding", "upgrade"})

#: Also not forwarded upstream: the caller's credentials, its host, and the account
#: selectors, so a caller cannot move the key's spending to another organization or project.
_DROPPED_REQUEST_HEADERS = _HOP_BY_HOP | {"authorization", "host", "openai-organization", "openai-project"}

#: Every method reaches the handler, so a path that is not proxied is a 404 whatever the method.
_ANY_METHOD = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


def _utc_day() -> str:
    """Today's date in UTC, the period of the request cap."""
    return datetime.now(timezone.utc).date().isoformat()


class _DailyCap:
    """Counts the requests of the current UTC day, in memory (a restart starts the day again)."""

    def __init__(self, limit: int):
        self.limit = limit
        self.day = ""
        self.count = 0

    def take(self) -> bool:
        """Count one request; False once the day's ``limit`` is used up."""
        today = _utc_day()
        if today != self.day:
            self.day, self.count = today, 0
        if self.count >= self.limit:
            return False
        self.count += 1
        if self.count == self.limit:
            logger.warning("Preview LLM proxy: the daily cap of %d requests is reached until 00:00 UTC", self.limit)
        return True


def _error(status: int, message: str, code: str) -> JSONResponse:
    """An error in OpenAI's shape, so the agent's SDK reports ``message``."""
    return JSONResponse({"error": {"message": message, "type": "lab_preview_proxy", "code": code}},
                        status_code=status)


def create_llm_proxy_app(*, api_key: SecretStr, base_url: str, max_requests_per_day: int,
                         transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    """The proxy as an ASGI app: ``POST /v1/<path>`` for each of :data:`PROXIED_PATHS`.

    Each request goes to ``<base_url>/<path>`` (query string kept) with
    ``Authorization: Bearer <api_key>`` in place of the caller's, and the response,
    status, headers and body, streams back as it arrives. Any other path or method
    answers 404; past ``max_requests_per_day`` requests in a UTC day, 429; an
    unreachable upstream, 502. ``transport`` replaces the network (tests).
    """
    upstream_base = base_url.rstrip("/")
    cap = _DailyCap(max_requests_per_day)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT, transport=transport) as client:
            app.state.upstream = client
            yield

    async def forward(request: Request) -> Response:
        path = request.path_params["path"]
        response = await respond(request, path)
        logger.info("Preview LLM proxy: /v1/%s -> %d", path, response.status_code)
        return response

    async def respond(request: Request, path: str) -> Response:
        if path not in PROXIED_PATHS or request.method != "POST":
            return _error(404, f"The lab's preview proxy does not forward {request.method} /v1/{path}.",
                          "not_proxied")
        if not cap.take():
            return _error(429, f"The lab's preview assistant reached its limit of {cap.limit} requests "
                               "today; it resets at 00:00 UTC.", "daily_limit")
        headers = [(name, value) for name, value in request.headers.items()
                   if name not in _DROPPED_REQUEST_HEADERS]
        headers.append(("authorization", f"Bearer {api_key.get_secret_value()}"))
        client: httpx.AsyncClient = request.app.state.upstream
        url = httpx.URL(f"{upstream_base}/{path}", query=request.url.query.encode())
        outgoing = client.build_request("POST", url, headers=headers, content=request.stream())
        try:
            upstream = await client.send(outgoing, stream=True)
        except httpx.HTTPError as exc:
            # The exception names the URL, never the headers; its type is enough to tell timeouts from refusals.
            logger.warning("Preview LLM proxy: /v1/%s could not reach the provider (%s)", path,
                           type(exc).__name__)
            return _error(502, "The lab's preview proxy could not reach the model provider.", "upstream_unreachable")
        return StreamingResponse(_relay(upstream), status_code=upstream.status_code,
                                 headers={name: value for name, value in upstream.headers.items()
                                          if name.lower() not in _HOP_BY_HOP})

    return Starlette(routes=[Route("/v1/{path:path}", forward, methods=_ANY_METHOD)], lifespan=lifespan)


async def _relay(upstream: httpx.Response) -> AsyncIterator[bytes]:
    """The upstream body as it arrives, still encoded; the connection is released however it ends."""
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    finally:
        await upstream.aclose()


class _EmbeddedServer(uvicorn.Server):
    """A uvicorn server inside the backend's own: signals stay with the backend's server."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


async def serve_llm_proxy(app: Starlette, port: int) -> None:
    """Serve ``app`` on ``127.0.0.1:<port>`` until this task is cancelled (a lifespan task of the backend).

    Raises ``RuntimeError`` when the port cannot be bound, rather than uvicorn's ``SystemExit``,
    which would stop the whole backend.
    """
    # log_config=None: uvicorn would otherwise replace the backend's logging; requests are logged above.
    server = _EmbeddedServer(uvicorn.Config(app, host=PROXY_HOST, port=port, log_config=None, access_log=False,
                                            timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS))
    try:
        await server.serve()
    except SystemExit:
        raise RuntimeError(f"the preview LLM proxy could not listen on {PROXY_HOST}:{port}") from None
    except asyncio.CancelledError:
        # Cancelled in its main loop, the server skipped its own shutdown: close the port and the client.
        if server.started:
            await server.shutdown()
        raise
