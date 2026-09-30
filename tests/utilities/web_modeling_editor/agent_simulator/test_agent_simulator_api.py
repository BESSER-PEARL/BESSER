"""Simulator API: token authentication (HTTP + WebSocket) and error mapping.

HTTP goes through ``httpx.ASGITransport`` and the WebSocket handshake through
a minimal ASGI harness, so the tests do not depend on the installed
starlette/httpx TestClient compatibility.
"""
import asyncio
import subprocess
import uuid
from typing import Dict, List, Optional

import httpx
import pytest

from besser.utilities.web_modeling_editor.agent_simulator import agent_simulator_api as api
from besser.utilities.web_modeling_editor.agent_simulator.sandbox import SandboxUnavailable
from besser.utilities.web_modeling_editor.agent_simulator.session_manager import SessionCapacityError

TOKEN = "test-simulator-token"
AUTH = {api.API_TOKEN_HEADER: TOKEN}


class Client:
    """Synchronous facade over an ASGI transport to the simulator app."""

    def request(self, method: str, path: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> httpx.Response:
        async def _go():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://simulator") as ac:
                return await ac.request(method, path, headers=headers or {}, **kwargs)

        return asyncio.run(_go())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)

    def delete(self, path, **kwargs):
        return self.request("DELETE", path, **kwargs)

    def websocket_handshake(self, path: str, headers: Optional[Dict[str, str]] = None) -> List[dict]:
        """Open a WebSocket, then disconnect; return every ASGI message the app sent."""
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            "client": ("127.0.0.1", 5000),
            "server": ("simulator", 8001),
            "subprotocols": [],
        }
        sent: List[dict] = []

        async def _go():
            inbox: asyncio.Queue = asyncio.Queue()
            await inbox.put({"type": "websocket.connect"})

            async def receive():
                return await inbox.get()

            async def send(message):
                sent.append(message)
                if message["type"] in ("websocket.close", "websocket.http.response.start"):
                    await inbox.put({"type": "websocket.disconnect", "code": 1000})

            await asyncio.wait_for(api.app(scope, receive, send), timeout=10)

        asyncio.run(_go())
        return sent


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(api.API_TOKEN_ENV, TOKEN)
    monkeypatch.setattr(api, "sandbox_selftest_error", lambda uid: None)
    return Client()


def _session_payload(**overrides):
    payload = {"session_id": str(uuid.uuid4()), "agent_code": "", "config_yaml": ""}
    payload.update(overrides)
    return payload


class TestTokenAuthentication:
    def test_header_and_env_names(self):
        assert api.API_TOKEN_HEADER == "X-Agent-Simulator-Token"
        assert api.API_TOKEN_ENV == "AGENT_SIMULATOR_API_TOKEN"

    def test_valid_token_is_accepted(self, client):
        response = client.get("/health", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["sandbox"] == "ok"

    @pytest.mark.parametrize("headers", [{}, {api.API_TOKEN_HEADER: "wrong"}, {api.API_TOKEN_HEADER: ""}])
    def test_missing_or_wrong_token_is_rejected(self, client, headers):
        assert client.get("/health", headers=headers).status_code == 401

    @pytest.mark.parametrize(
        "method, path",
        [
            ("post", "/sessions"),
            ("get", f"/sessions/{uuid.uuid4()}/files"),
            ("delete", f"/sessions/{uuid.uuid4()}"),
        ],
    )
    def test_every_http_endpoint_requires_the_token(self, client, method, path):
        response = getattr(client, method)(path)
        assert response.status_code == 401

    def test_unset_token_fails_closed_with_503(self, client, monkeypatch):
        monkeypatch.delenv(api.API_TOKEN_ENV)
        # Even a request carrying some token is refused: nothing to compare to.
        assert client.get("/health", headers=AUTH).status_code == 503
        assert client.get("/health").status_code == 503

    def test_websocket_without_token_is_refused(self, client):
        sent = client.websocket_handshake(f"/sessions/{uuid.uuid4()}/ws")
        assert sent[0] == {"type": "websocket.close", "code": 1008, "reason": sent[0]["reason"]}
        assert not any(m["type"] == "websocket.accept" for m in sent)

    def test_websocket_with_wrong_token_is_refused(self, client):
        sent = client.websocket_handshake(f"/sessions/{uuid.uuid4()}/ws", headers={api.API_TOKEN_HEADER: "x"})
        assert sent[0]["type"] == "websocket.close" and sent[0]["code"] == 1008

    def test_websocket_without_configured_token_is_refused(self, client, monkeypatch):
        monkeypatch.delenv(api.API_TOKEN_ENV)
        sent = client.websocket_handshake(f"/sessions/{uuid.uuid4()}/ws", headers=AUTH)
        assert sent[0]["type"] == "websocket.close" and sent[0]["code"] == 1008

    def test_websocket_with_token_reaches_the_handler(self, client):
        sent = client.websocket_handshake(f"/sessions/{uuid.uuid4()}/ws", headers=AUTH)
        assert sent[0]["type"] == "websocket.accept"
        assert '"Session not found"' in sent[1]["text"]
        assert sent[2] == {"type": "websocket.close", "code": 4004, "reason": sent[2].get("reason")}

    def test_no_openapi_schema_is_served(self, client):
        assert client.get("/openapi.json", headers=AUTH).status_code == 404


class TestCreateSessionErrors:
    @pytest.mark.parametrize(
        "error, status",
        [
            (SessionCapacityError("full"), 429),
            (SandboxUnavailable("bubblewrap (bwrap) is not installed"), 503),
            (ValueError("Path '../x' must not contain '..' segments."), 400),
            (subprocess.SubprocessError("Exception occurred in preexec_fn."), 500),
            (OSError("no space"), 500),
        ],
    )
    def test_error_mapping(self, client, monkeypatch, error, status):
        def raise_error(*args):
            raise error

        monkeypatch.setattr(api.session_manager, "create_session", raise_error)
        response = client.post("/sessions", json=_session_payload(), headers=AUTH)
        assert response.status_code == status

    def test_sandbox_detail_is_not_leaked(self, client, monkeypatch):
        def raise_error(*args):
            raise SandboxUnavailable("bubblewrap cannot create a sandbox here (internal detail)")

        monkeypatch.setattr(api.session_manager, "create_session", raise_error)
        response = client.post("/sessions", json=_session_payload(), headers=AUTH)
        assert "internal detail" not in response.text

    def test_invalid_session_id(self, client):
        response = client.post("/sessions", json=_session_payload(session_id="../../etc"), headers=AUTH)
        assert response.status_code == 400
