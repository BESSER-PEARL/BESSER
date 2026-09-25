"""
Tests for the live agent simulation router (/besser_api/simulation/*).

The simulator service is never contacted: HTTP calls go through an
``httpx.MockTransport`` installed on the router's transport seam, and the
simulator WebSocket is replaced by an in-memory fake. The frontend WebSocket
is driven by a small ASGI harness so the tests do not depend on the
installed starlette/httpx TestClient compatibility.
"""

import asyncio
import importlib.util
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    AGENT_SIMULATOR_TOKEN_ENV_VAR,
    AGENT_SIMULATOR_TOKEN_HEADER,
)
from besser.utilities.web_modeling_editor.backend.routers import agent_simulator_router as asr
from besser.utilities.web_modeling_editor.backend.routers import auth as auth_module

BASE = "/besser_api/simulation"
SIM_TOKEN = "test-simulator-token"
ALICE = "alice-session"
BOB = "bob-session"


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

def _agent_model(actions: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Minimal AgentDiagram: initial node -> one state, with optional action elements."""
    elements: Dict[str, Any] = {
        "init-node": {"id": "init-node", "type": "StateInitialNode", "name": ""},
        "state-1": {
            "id": "state-1",
            "type": "AgentState",
            "name": "Greet",
            "actions": list((actions or {}).keys()),
        },
    }
    elements.update(actions or {})
    return {
        "type": "AgentDiagram",
        "elements": elements,
        "relationships": {
            "trans-init": {
                "type": "AgentStateTransitionInit",
                "name": "",
                "source": {"element": "init-node"},
                "target": {"element": "state-1"},
            },
        },
    }


def _text_reply_model() -> Dict[str, Any]:
    return _agent_model({"a1": {"id": "a1", "type": "AgentStateBody", "actionType": "TextReplyAction",
                                "name": "Hello there"}})


def _custom_code_model(code: str) -> Dict[str, Any]:
    return _agent_model({"a1": {"id": "a1", "type": "AgentStateBody", "actionType": "CustomCodeAction",
                                "name": code}})


def _session_payload(model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"title": "Agent", "model": model or _text_reply_model()}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeSimulator:
    """Records every HTTP request the router sends to the simulator."""

    def __init__(self):
        self.requests: List[httpx.Request] = []
        self.create_status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/sessions":
            body = json.loads(request.content)
            return httpx.Response(self.create_status, json={"session_id": body["session_id"], "port": 7700})
        if request.method == "GET" and path.endswith("/files"):
            return httpx.Response(200, json={"files": [{"path": "out.txt", "content": "hi"}], "directories": []})
        if request.method == "DELETE":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"detail": "not found"})


@pytest.fixture
def simulator(monkeypatch):
    """Fresh in-process state, a mocked simulator, sign-in on, custom code restricted."""
    fake = _FakeSimulator()
    monkeypatch.setenv(AGENT_SIMULATOR_TOKEN_ENV_VAR, SIM_TOKEN)
    monkeypatch.setattr(asr, "_http_transport", httpx.MockTransport(fake.handler))
    monkeypatch.setattr(asr, "_rate_limiter", asr._SlidingWindowRateLimiter(60, 100, 1000))
    monkeypatch.setattr(asr, "_session_registry", asr._SimulationSessionRegistry(1, 900))
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_REQUIRE_AUTH", True)
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE", True)
    monkeypatch.setattr(auth_module, "get_user_token", lambda s: "gh-token" if s in {ALICE, BOB} else None)
    return fake


@pytest.fixture
def fast_generation(monkeypatch):
    """Skip real BAF generation for tests that only exercise the router plumbing."""
    async def _fake_generate(diagram_data, config, config_yaml):
        return "print('agent')", "platforms: {}\n", ["ReceiveTextEvent"], {}, []

    monkeypatch.setattr(asr, "_generate_agent_code_and_config", _fake_generate)


def _request(method: str, url: str, github_session: Optional[str] = None,
             client: Tuple[str, int] = ("127.0.0.1", 123), **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    if github_session:
        headers["X-GitHub-Session"] = github_session

    async def _go():
        transport = ASGITransport(app=app, client=client)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            return await ac.request(method, url, headers=headers, **kwargs)

    return _run(_go())


def _create_session(github_session: Optional[str] = ALICE, **kwargs) -> httpx.Response:
    return _request("POST", f"{BASE}/sessions", github_session, json=_session_payload(), **kwargs)


# ---------------------------------------------------------------------------
# /limits and auth
# ---------------------------------------------------------------------------

def test_limits_requires_github_session(simulator):
    resp = _request("GET", f"{BASE}/limits")
    assert resp.status_code == 401
    assert "GitHub" in resp.json()["detail"]


def test_limits_rejects_expired_session(simulator):
    resp = _request("GET", f"{BASE}/limits", "unknown-session")
    assert resp.status_code == 401
    assert "expired" in resp.json()["detail"]


def test_limits_returns_configured_values(simulator, monkeypatch):
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_MEMORY_MB", 512)
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_CPU_CORES", 0.5)
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_QUOTA_ENABLED", True)
    resp = _request("GET", f"{BASE}/limits", ALICE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["memoryMb"] == 512
    assert body["cpuCores"] == 0.5
    assert body["editorQuotaEnabled"] is True


def test_auth_can_be_disabled(simulator, monkeypatch):
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_REQUIRE_AUTH", False)
    assert _request("GET", f"{BASE}/limits").status_code == 200


# ---------------------------------------------------------------------------
# /validate
# ---------------------------------------------------------------------------

def test_validate_generates_agent_code(simulator):
    resp = _request("POST", f"{BASE}/validate", ALICE, json=_session_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["errors"] == []
    assert "Greet" in body["agentCode"]
    # Validation never contacts the simulator.
    assert simulator.requests == []


def _load_fresh_constants():
    """Import a private copy of constants.py so env defaults are re-evaluated."""
    from besser.utilities.web_modeling_editor.backend.constants import constants

    spec = importlib.util.spec_from_file_location("_fresh_simulator_constants", constants.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("env_value, expected", [(None, True), ("false", False), ("1", True)])
def test_restrict_custom_code_defaults_to_true(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE", raising=False)
    else:
        monkeypatch.setenv("AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE", env_value)
    assert _load_fresh_constants().AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE is expected


def test_simulator_url_default_and_override(monkeypatch):
    monkeypatch.delenv("AGENT_SIMULATOR_URL", raising=False)
    assert _load_fresh_constants().AGENT_SIMULATOR_URL == "http://besser-wme-agent-simulator:8001"
    monkeypatch.setenv("AGENT_SIMULATOR_URL", "http://sim.internal:9001/")
    assert _load_fresh_constants().AGENT_SIMULATOR_URL == "http://sim.internal:9001"


def test_validate_reports_restricted_custom_code(simulator):
    model = _custom_code_model("def act(session):\n    session.reply('hi')\n")
    resp = _request("POST", f"{BASE}/validate", ALICE, json=_session_payload(model))
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert "custom Python code" in body["errors"][0]


def test_validate_reports_custom_code_lint_errors_when_unrestricted(simulator, monkeypatch):
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE", False)
    model = _custom_code_model("def act(session):\n    import subprocess\n")
    resp = _request("POST", f"{BASE}/validate", ALICE, json=_session_payload(model))
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert "subprocess" in body["errors"][0]


# ---------------------------------------------------------------------------
# /sessions
# ---------------------------------------------------------------------------

def test_create_session_sends_simulator_token(simulator, fast_generation):
    resp = _create_session()
    assert resp.status_code == 200, resp.text
    session_id = resp.json()["sessionId"]
    uuid.UUID(session_id)
    assert resp.json()["eventList"] == ["ReceiveTextEvent"]

    [sent] = simulator.requests
    assert sent.method == "POST"
    assert sent.headers[AGENT_SIMULATOR_TOKEN_HEADER] == SIM_TOKEN
    assert json.loads(sent.content)["session_id"] == session_id


def test_create_session_with_real_generation(simulator):
    resp = _create_session()
    assert resp.status_code == 200, resp.text
    payload = json.loads(simulator.requests[0].content)
    assert "Greet" in payload["agent_code"]
    assert payload["config_yaml"]


def test_create_session_forwards_credentials_as_env_vars(simulator, fast_generation):
    body = _session_payload()
    body["credentials"] = {"openAiApiKey": "sk-test", "huggingFaceToken": ""}
    resp = _request("POST", f"{BASE}/sessions", ALICE, json=body)
    assert resp.status_code == 200
    assert json.loads(simulator.requests[0].content)["env_vars"] == {"OPENAI_API_KEY": "sk-test"}


def test_create_session_without_simulator_token_is_503(simulator, fast_generation, monkeypatch):
    monkeypatch.delenv(AGENT_SIMULATOR_TOKEN_ENV_VAR)
    resp = _create_session()
    assert resp.status_code == 503
    assert simulator.requests == []
    # The failed attempt does not hold the actor's session slot.
    monkeypatch.setenv(AGENT_SIMULATOR_TOKEN_ENV_VAR, SIM_TOKEN)
    assert _create_session().status_code == 200


def test_create_session_requires_auth(simulator, fast_generation):
    assert _create_session(github_session=None).status_code == 401
    assert simulator.requests == []


def test_create_session_rejects_restricted_custom_code(simulator):
    model = _custom_code_model("def act(session):\n    session.reply('hi')\n")
    resp = _request("POST", f"{BASE}/sessions", ALICE, json=_session_payload(model))
    assert resp.status_code == 403
    assert simulator.requests == []


def test_create_session_maps_code_validation_error_to_400(simulator, monkeypatch):
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE", False)
    model = _custom_code_model("def act(ctx):\n    pass\n")
    resp = _request("POST", f"{BASE}/sessions", ALICE, json=_session_payload(model))
    assert resp.status_code == 400
    assert "first parameter must be named 'session'" in resp.json()["detail"]
    assert simulator.requests == []


def test_create_session_simulator_rejection_is_502_without_leaking_body(simulator, fast_generation):
    simulator.create_status = 500
    resp = _create_session()
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Agent simulator service rejected session creation."
    # The slot is released on failure.
    simulator.create_status = 200
    assert _create_session().status_code == 200


def test_per_actor_session_cap(simulator, fast_generation):
    assert _create_session(ALICE).status_code == 200
    second = _create_session(ALICE)
    assert second.status_code == 429
    assert "active agent simulation session" in second.json()["detail"]
    # Another actor is not affected by Alice's cap.
    assert _create_session(BOB).status_code == 200


def test_session_cap_frees_up_after_stop(simulator, fast_generation):
    session_id = _create_session(ALICE).json()["sessionId"]
    assert _request("DELETE", f"{BASE}/sessions/{session_id}", ALICE).status_code == 200
    assert _create_session(ALICE).status_code == 200


def test_rate_limit_returns_429(simulator, fast_generation, monkeypatch):
    monkeypatch.setattr(asr, "_rate_limiter", asr._SlidingWindowRateLimiter(60, 2, 1000))
    for _ in range(2):
        assert _request("POST", f"{BASE}/validate", ALICE, json=_session_payload()).status_code == 200
    resp = _request("POST", f"{BASE}/validate", ALICE, json=_session_payload())
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "60"
    # Budgets are per actor.
    assert _request("POST", f"{BASE}/validate", BOB, json=_session_payload()).status_code == 200


def test_rate_limiter_bounds_tracked_keys():
    limiter = asr._SlidingWindowRateLimiter(window_seconds=60, max_requests=5, max_keys=3)
    for i in range(10):
        limiter.check(f"actor-{i}")
    assert len(limiter) == 3


def test_rate_limiter_prunes_idle_keys(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(asr.time, "monotonic", lambda: clock[0])
    limiter = asr._SlidingWindowRateLimiter(window_seconds=10, max_requests=1, max_keys=100)
    for i in range(5):
        limiter.check(f"actor-{i}")
    assert len(limiter) == 5
    clock[0] += 11
    limiter.check("actor-new")
    assert len(limiter) == 1
    # An expired window lets the same actor through again.
    limiter.check("actor-0")


def test_session_registry_forgets_expired_sessions(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(asr.time, "monotonic", lambda: clock[0])
    registry = asr._SimulationSessionRegistry(max_sessions_per_actor=1, ttl_seconds=900)
    session_id = registry.reserve("actor")
    assert registry.is_owner(session_id, "actor")
    clock[0] += 901
    assert not registry.is_owner(session_id, "actor")
    registry.reserve("actor")  # the expired session no longer counts toward the cap


# ---------------------------------------------------------------------------
# Ownership: files / delete
# ---------------------------------------------------------------------------

def test_owner_can_list_files(simulator, fast_generation):
    session_id = _create_session(ALICE).json()["sessionId"]
    resp = _request("GET", f"{BASE}/sessions/{session_id}/files", ALICE)
    assert resp.status_code == 200
    assert resp.json()["files"] == [{"path": "out.txt", "content": "hi"}]
    assert simulator.requests[-1].headers[AGENT_SIMULATOR_TOKEN_HEADER] == SIM_TOKEN


def test_other_actor_gets_404_on_files_and_delete(simulator, fast_generation):
    session_id = _create_session(ALICE).json()["sessionId"]
    requests_before = len(simulator.requests)

    assert _request("GET", f"{BASE}/sessions/{session_id}/files", BOB).status_code == 404
    assert _request("DELETE", f"{BASE}/sessions/{session_id}", BOB).status_code == 404
    # Nothing was forwarded to the simulator, and Alice still owns the session.
    assert len(simulator.requests) == requests_before
    assert _request("GET", f"{BASE}/sessions/{session_id}/files", ALICE).status_code == 200


def test_other_ip_gets_404_when_auth_is_off(simulator, fast_generation, monkeypatch):
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_REQUIRE_AUTH", False)
    session_id = _create_session(None, client=("10.0.0.1", 1)).json()["sessionId"]
    url = f"{BASE}/sessions/{session_id}/files"
    assert _request("GET", url, client=("10.0.0.2", 1)).status_code == 404
    assert _request("GET", url, client=("10.0.0.1", 1)).status_code == 200


def test_unknown_session_is_404(simulator):
    resp = _request("DELETE", f"{BASE}/sessions/{uuid.uuid4()}", ALICE)
    assert resp.status_code == 404


def test_invalid_session_id_is_400(simulator):
    assert _request("GET", f"{BASE}/sessions/not-a-uuid/files", ALICE).status_code == 400


def test_delete_forwards_to_simulator_with_token(simulator, fast_generation):
    session_id = _create_session(ALICE).json()["sessionId"]
    resp = _request("DELETE", f"{BASE}/sessions/{session_id}", ALICE)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    last = simulator.requests[-1]
    assert last.method == "DELETE"
    assert last.url.path == f"/sessions/{session_id}"
    assert last.headers[AGENT_SIMULATOR_TOKEN_HEADER] == SIM_TOKEN


# ---------------------------------------------------------------------------
# CodeValidationError on the generation endpoint
# ---------------------------------------------------------------------------

def test_generate_output_maps_code_validation_error_to_400():
    body = {
        "title": "Agent",
        "generator": "agent",
        "model": _custom_code_model("def act(ctx):\n    pass\n"),
    }
    resp = _request("POST", "/besser_api/generate-output", json=body)
    assert resp.status_code == 400, resp.text
    assert "first parameter must be named 'session'" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# WebSocket relay
# ---------------------------------------------------------------------------

class _FakeSimulatorWS:
    """In-memory stand-in for the simulator WebSocket: echoes what it receives."""

    def __init__(self):
        self.sent: List[str] = []
        self.closed = False
        self._incoming: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

    async def send(self, message: str) -> None:
        self.sent.append(message)
        await self._incoming.put(json.dumps({"echo": message}))

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            await self._incoming.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        message = await self._incoming.get()
        if message is None:
            raise StopAsyncIteration
        return message


class _WebSocketClient:
    """Minimal ASGI WebSocket client driving the real app."""

    def __init__(self, path: str, client: Tuple[str, int] = ("127.0.0.1", 5000)):
        self._to_app: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self._from_app: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self._scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": [(b"host", b"testserver")],
            "client": client,
            "server": ("testserver", 80),
            "subprotocols": [],
        }
        self._task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        await self._to_app.put({"type": "websocket.connect"})
        self._task = asyncio.create_task(app(self._scope, self._to_app.get, self._from_app.put))
        message = await self.receive()
        assert message["type"] == "websocket.accept", message

    async def send_json(self, data: Any) -> None:
        await self._to_app.put({"type": "websocket.receive", "text": json.dumps(data)})

    async def send_text(self, text: str) -> None:
        await self._to_app.put({"type": "websocket.receive", "text": text})

    async def receive(self) -> Dict[str, Any]:
        return await asyncio.wait_for(self._from_app.get(), timeout=5)

    async def receive_json(self) -> Any:
        message = await self.receive()
        assert message["type"] == "websocket.send", message
        return json.loads(message["text"])

    async def expect_close(self, code: int) -> Dict[str, Any]:
        error = await self.receive_json()
        assert error["type"] == "error"
        closing = await self.receive()
        assert closing == {"type": "websocket.close", "code": code, "reason": ""} or (
            closing["type"] == "websocket.close" and closing["code"] == code
        ), closing
        await asyncio.wait_for(self._task, timeout=5)
        return error

    async def disconnect(self) -> None:
        await self._to_app.put({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(self._task, timeout=5)


@pytest.fixture
def fake_ws_connect(monkeypatch):
    captured: Dict[str, Any] = {}

    async def _connect(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        captured["ws"] = _FakeSimulatorWS()
        return captured["ws"]

    monkeypatch.setattr(asr, "_ws_connect", _connect)
    return captured


def test_ws_relays_after_first_frame_auth(simulator, fast_generation, fake_ws_connect):
    session_id = _create_session(ALICE).json()["sessionId"]

    async def _scenario():
        ws = _WebSocketClient(f"{BASE}/{session_id}/ws")
        await ws.connect()
        await ws.send_json({"type": "auth", "githubSession": ALICE})
        assert await ws.receive_json() == {"type": "auth_ok"}

        await ws.send_text("hello agent")
        assert await ws.receive_json() == {"echo": "hello agent"}

        await ws.disconnect()
        await asyncio.gather(*list(asr._background_tasks))

    _run(_scenario())

    assert fake_ws_connect["url"].endswith(f"/sessions/{session_id}/ws")
    assert fake_ws_connect["url"].startswith("ws://")
    assert fake_ws_connect["kwargs"]["additional_headers"] == {AGENT_SIMULATOR_TOKEN_HEADER: SIM_TOKEN}
    assert fake_ws_connect["ws"].sent == ["hello agent"]
    assert fake_ws_connect["ws"].closed is True
    # The session is cleaned up on the simulator and forgotten locally.
    last = simulator.requests[-1]
    assert last.method == "DELETE" and last.headers[AGENT_SIMULATOR_TOKEN_HEADER] == SIM_TOKEN
    assert len(asr._background_tasks) == 0
    assert len(asr._session_registry) == 0


@pytest.mark.parametrize(
    "first_frame",
    [
        {"type": "auth", "githubSession": ""},
        {"type": "auth", "githubSession": "unknown-session"},
        {"type": "auth"},
        {"type": "hello", "githubSession": ALICE},
        "not json",
    ],
)
def test_ws_bad_or_missing_auth_closes_4401(simulator, fast_generation, fake_ws_connect, first_frame):
    session_id = _create_session(ALICE).json()["sessionId"]

    async def _scenario():
        ws = _WebSocketClient(f"{BASE}/{session_id}/ws")
        await ws.connect()
        if isinstance(first_frame, str):
            await ws.send_text(first_frame)
        else:
            await ws.send_json(first_frame)
        await ws.expect_close(4401)

    _run(_scenario())
    assert "url" not in fake_ws_connect


def test_ws_auth_frame_timeout_closes_4401(simulator, fast_generation, fake_ws_connect, monkeypatch):
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_WS_AUTH_TIMEOUT_SECONDS", 0.05)
    session_id = _create_session(ALICE).json()["sessionId"]

    async def _scenario():
        ws = _WebSocketClient(f"{BASE}/{session_id}/ws")
        await ws.connect()
        error = await ws.expect_close(4401)
        assert "in time" in error["message"]

    _run(_scenario())
    assert "url" not in fake_ws_connect


def test_ws_query_param_token_is_ignored(simulator, fast_generation, fake_ws_connect, monkeypatch):
    monkeypatch.setattr(asr, "AGENT_SIMULATOR_WS_AUTH_TIMEOUT_SECONDS", 0.05)
    session_id = _create_session(ALICE).json()["sessionId"]

    async def _scenario():
        ws = _WebSocketClient(f"{BASE}/{session_id}/ws")
        ws._scope["query_string"] = f"github_session={ALICE}".encode()
        await ws.connect()
        await ws.expect_close(4401)

    _run(_scenario())


def test_ws_other_actor_closes_4404(simulator, fast_generation, fake_ws_connect):
    session_id = _create_session(ALICE).json()["sessionId"]

    async def _scenario():
        ws = _WebSocketClient(f"{BASE}/{session_id}/ws")
        await ws.connect()
        await ws.send_json({"type": "auth", "githubSession": BOB})
        await ws.expect_close(4404)

    _run(_scenario())
    assert "url" not in fake_ws_connect
    # Alice's session survives Bob's attempt.
    assert len(asr._session_registry) == 1


def test_ws_invalid_session_id_closes_4400(simulator, fake_ws_connect):
    async def _scenario():
        ws = _WebSocketClient(f"{BASE}/not-a-uuid/ws")
        await ws.connect()
        await ws.expect_close(4400)

    _run(_scenario())
