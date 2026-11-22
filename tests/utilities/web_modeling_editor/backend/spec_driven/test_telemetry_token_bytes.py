"""A non-ASCII X-Telemetry-Token is a 403, not a 500.

``secrets.compare_digest`` raises TypeError on a non-ASCII ``str``; Starlette
decodes header bytes as latin-1, so any byte >= 0x80 used to crash the report.
"""

import asyncio

import httpx
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app


def _get_report(token: bytes) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            return await ac.get(
                "/besser_api/telemetry/report", headers={"X-Telemetry-Token": token},
            )

    return asyncio.run(_run())


def test_non_ascii_token_is_403(monkeypatch):
    monkeypatch.setenv("BESSER_TELEMETRY_ADMIN_TOKEN", "admin-token-123")
    assert _get_report("töken".encode("utf-8")).status_code == 403
