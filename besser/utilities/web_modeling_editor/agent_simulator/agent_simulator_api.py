"""
BESSER Agent Simulator API

Isolated execution environment for user-designed BAF agents. Runs as a
separate container on its own Docker network, reachable only by the backend.
Each POST /sessions starts one sandboxed agent process (see
:mod:`.session_manager` and :mod:`.sandbox`); the /sessions/{id}/ws endpoint
bridges the backend WebSocket to the running agent.

Every HTTP and WebSocket endpoint requires the ``X-Agent-Simulator-Token``
header to equal ``AGENT_SIMULATOR_API_TOKEN``. Agent processes share the
simulator's network namespace and can reach this API on localhost, so the
token (which agent code cannot read) is what keeps them out. Without the
variable the API refuses every request with 503 (fail closed).
"""
import asyncio
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from typing import Dict, List

import uvicorn
import websockets
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, WebSocketException, status
from pydantic import BaseModel, Field
from starlette.requests import HTTPConnection
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import WebSocketException as AgentWebSocketError

from besser.utilities.web_modeling_editor.agent_simulator import session_manager as session_module
from besser.utilities.web_modeling_editor.agent_simulator.sandbox import SandboxUnavailable, sandbox_selftest_error
from besser.utilities.web_modeling_editor.agent_simulator.session_manager import (
    Session,
    SessionCapacityError,
    session_manager,
)

logging.basicConfig(
    level=logging.INFO,
    format="{levelname} - {asctime}: {message}",
    style="{",
)
logger = logging.getLogger(__name__)

API_TOKEN_ENV = "AGENT_SIMULATOR_API_TOKEN"
API_TOKEN_HEADER = "X-Agent-Simulator-Token"
API_PORT = 8001

_STATE_BODY_LOG_RE = re.compile(r"\[(?P<state>[^]]+)\]\s+Running body\b")

# How long the relay waits for the agent's WebSocket server: 150 x 0.5 s.
_AGENT_CONNECT_ATTEMPTS = 150
_AGENT_CONNECT_RETRY_SECONDS = 0.5

# Errors that mean "the agent's WebSocket server is not (yet) accepting".
_AGENT_CONNECT_ERRORS = (OSError, asyncio.TimeoutError, AgentWebSocketError)
# Errors that mean "the backend side of the relay is gone".
_RELAY_SEND_ERRORS = (WebSocketDisconnect, RuntimeError, OSError, ConnectionClosed)

_missing_token_logged = False


def _reject(connection: HTTPConnection, http_status: int, reason: str) -> None:
    if connection.scope["type"] == "websocket":
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=reason)
    raise HTTPException(status_code=http_status, detail=reason)


async def require_api_token(connection: HTTPConnection) -> None:
    """Reject any request whose ``X-Agent-Simulator-Token`` does not match.

    Applied to every route (HTTP and WebSocket) through the app dependencies.
    The comparison is constant-time.
    """
    global _missing_token_logged
    expected = os.environ.get(API_TOKEN_ENV, "")
    if not expected:
        if not _missing_token_logged:
            _missing_token_logged = True
            logger.error(
                "AGENT_SIMULATOR_API_TOKEN is not set: refusing every request (fail closed). "
                "Set the same value on the backend and on the simulator."
            )
        _reject(connection, status.HTTP_503_SERVICE_UNAVAILABLE, "Agent simulator API token is not configured")
    supplied = connection.headers.get(API_TOKEN_HEADER, "")
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        _reject(connection, status.HTTP_401_UNAUTHORIZED, "Invalid or missing agent simulator token")


def _is_valid_session_id(session_id: str) -> bool:
    try:
        uuid.UUID(session_id)
        return True
    except (ValueError, TypeError):
        return False


app = FastAPI(
    title="BESSER Agent Simulator API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    dependencies=[Depends(require_api_token)],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    session_id: str
    agent_code: str
    config_yaml: str
    env_vars: Dict[str, str] = {}
    event_list: List[str] = []
    support_files: Dict[str, str] = {}
    workspace_paths: List[str] = []


class CreateSessionResponse(BaseModel):
    session_id: str
    port: int


class SessionFile(BaseModel):
    path: str
    content: str


class SessionFilesResponse(BaseModel):
    files: List[SessionFile]
    directories: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Start a new agent simulator session."""
    if not _is_valid_session_id(request.session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    try:
        session = await asyncio.to_thread(
            session_manager.create_session,
            request.session_id,
            request.agent_code,
            request.config_yaml,
            request.env_vars,
            request.event_list,
            request.support_files,
            request.workspace_paths,
        )
    except SessionCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SandboxUnavailable as exc:
        logger.error("Refusing session %s: %s", request.session_id, exc)
        raise HTTPException(status_code=503, detail="Agent sandbox is unavailable on this server") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        logger.exception("Failed to start session %s", request.session_id)
        raise HTTPException(status_code=500, detail="Agent process could not be started") from exc
    return CreateSessionResponse(session_id=session.session_id, port=session.port)


@app.get("/sessions/{session_id}/files", response_model=SessionFilesResponse)
async def list_session_files(session_id: str):
    """Return the regular files and directories in the session work directory."""
    if not _is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    payload = await asyncio.to_thread(session_manager.get_session_files, session_id)
    return SessionFilesResponse(
        files=[SessionFile(**f) for f in payload["files"]],
        directories=payload["directories"],
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Terminate a simulator session and clean up its resources."""
    if not _is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    await asyncio.to_thread(session_manager.terminate_session, session_id)
    return {"ok": True}


@app.get("/health")
async def health():
    """Health check: current session count and whether the sandbox can start."""
    sandbox_error = await asyncio.to_thread(sandbox_selftest_error, session_module.UID_POOL[0])
    return {
        "status": "ok" if sandbox_error is None else "degraded",
        "sessions": session_manager.get_session_count(),
        "sandbox": "ok" if sandbox_error is None else f"unavailable: {sandbox_error}",
    }


# ---------------------------------------------------------------------------
# WebSocket relay
# ---------------------------------------------------------------------------


async def _send_json(websocket: WebSocket, payload: dict) -> bool:
    """Send one JSON message to the backend; False once the backend is gone."""
    try:
        await websocket.send_text(json.dumps(payload))
    except _RELAY_SEND_ERRORS as exc:
        logger.debug("Backend WebSocket closed while sending: %s", exc)
        return False
    return True


@app.websocket("/sessions/{session_id}/ws")
async def session_ws(websocket: WebSocket, session_id: str):
    """
    Relay WebSocket between the backend and the running agent subprocess.

    Message routing:
    - Backend -> agent:  user messages forwarded to agent's WebSocket server.
    - Agent -> backend:  agent replies forwarded to backend.
    - Agent stdout:      state-body log lines are parsed via regex and
                         forwarded as structured events; other lines are
                         forwarded as ``{"type":"stdout","line":"..."}`` messages.
    """
    await websocket.accept()

    if not _is_valid_session_id(session_id):
        await _send_json(websocket, {"type": "error", "message": "Invalid session id"})
        await websocket.close(code=4400)
        return

    session = session_manager.get_session(session_id)
    if not session:
        await _send_json(websocket, {"type": "error", "message": "Session not found"})
        await websocket.close(code=4004)
        return

    agent_ws_url = f"ws://{session_module.AGENT_WS_HOST}:{session.port}/"
    agent_ws = None
    for _attempt in range(_AGENT_CONNECT_ATTEMPTS):
        try:
            agent_ws = await websockets.connect(agent_ws_url, open_timeout=2)
            logger.info("[%s] Connected to agent WS on port %d", session_id, session.port)
            break
        except _AGENT_CONNECT_ERRORS as exc:
            logger.debug("[%s] agent WS not ready: %s", session_id, exc)
            if not session.is_alive():
                break
            await asyncio.sleep(_AGENT_CONNECT_RETRY_SECONDS)

    if agent_ws is None:
        await _report_start_failure(websocket, session)
        await asyncio.to_thread(session_manager.terminate_session, session_id)
        return

    stop_event = asyncio.Event()

    async def _read_stdout():
        """Read agent stdout; forward inferred state-change events and lines."""
        loop = asyncio.get_running_loop()
        try:
            while not stop_event.is_set():
                line_bytes = await loop.run_in_executor(None, session.process.stdout.readline)
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                state_match = _STATE_BODY_LOG_RE.search(line)
                if state_match:
                    event = {"type": "state_change", "state": state_match.group("state"), "_source": "log_state"}
                    if not await _send_json(websocket, event):
                        break
                if not await _send_json(websocket, {"type": "stdout", "line": line}):
                    break
        except (OSError, ValueError) as exc:
            # ValueError: the pipe was closed by terminate_session mid-read.
            logger.debug("[%s] stdout reader ended: %s", session_id, exc)
        finally:
            stop_event.set()

    async def _frontend_to_agent():
        """Forward messages from the backend/frontend to the agent."""
        try:
            while not stop_event.is_set():
                msg = await websocket.receive_text()
                await agent_ws.send(msg)
        except WebSocketDisconnect:
            logger.debug("[%s] backend disconnected", session_id)
        except (ConnectionClosed, RuntimeError, OSError) as exc:
            logger.debug("[%s] frontend->agent relay ended: %s", session_id, exc)
        finally:
            stop_event.set()
            # Close agent WS so _agent_to_frontend's async-for exits immediately
            # instead of blocking until the agent sends its next message.
            await agent_ws.close()

    async def _agent_to_frontend():
        """Forward messages from the agent to the backend/frontend."""
        try:
            async for raw in agent_ws:
                text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                await websocket.send_text(text)
        except _RELAY_SEND_ERRORS as exc:
            logger.debug("[%s] agent->frontend relay ended: %s", session_id, exc)
        finally:
            stop_event.set()

    try:
        results = await asyncio.gather(
            _read_stdout(), _frontend_to_agent(), _agent_to_frontend(), return_exceptions=True,
        )
    finally:
        await agent_ws.close()
    for result in results:
        if isinstance(result, Exception):
            logger.warning("[%s] relay task failed: %r", session_id, result)


async def _report_start_failure(websocket: WebSocket, session: Session) -> None:
    """Forward what the agent printed before dying, then close the relay."""
    # Give the process a moment to flush remaining output, then read what's available.
    await asyncio.sleep(0.3)
    stdout_text = ""
    try:
        # read1 returns available bytes without blocking on EOF (safe even if process is alive)
        raw = await asyncio.wait_for(asyncio.to_thread(session.process.stdout.read1, 65536), timeout=2.0)
        stdout_text = raw.decode("utf-8", errors="replace") if raw else ""
    except (asyncio.TimeoutError, OSError, ValueError) as exc:
        logger.debug("[%s] could not read agent output: %s", session.session_id, exc)

    # Forward each stdout line to the terminal pane so the user can see the traceback
    for line in stdout_text.splitlines():
        if not await _send_json(websocket, {"type": "stdout", "line": line}):
            return
    detail = stdout_text.strip() or "No output captured - the process may have crashed silently."
    message = f"Agent subprocess failed to start its WebSocket server.\n\n{detail}"
    if await _send_json(websocket, {"type": "error", "message": message}):
        await websocket.close()


# ---------------------------------------------------------------------------
# Startup and background cleanup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _start_cleanup_task():
    _ensure_sessions_root_permissions()
    _remove_stale_session_dirs()
    if not os.environ.get(API_TOKEN_ENV):
        logger.error("AGENT_SIMULATOR_API_TOKEN is not set: every request will be refused with 503")
    sandbox_error = sandbox_selftest_error(session_module.UID_POOL[0])
    if sandbox_error:
        logger.error("Agent sandbox unavailable (%s): sessions will be refused", sandbox_error)
    asyncio.create_task(_periodic_cleanup())


def _ensure_sessions_root_permissions() -> None:
    """Sessions root: root-owned, 0711 so session UIDs can only reach their own dir."""
    sessions_root = session_module.SESSIONS_ROOT
    os.makedirs(sessions_root, exist_ok=True)
    os.chown(sessions_root, 0, 0)
    os.chmod(sessions_root, 0o711)


def _remove_stale_session_dirs() -> None:
    """Remove any session directories left over from a previous run."""
    sessions_root = session_module.SESSIONS_ROOT
    for entry in os.listdir(sessions_root):
        entry_path = os.path.join(sessions_root, entry)
        if os.path.isdir(entry_path) and not os.path.islink(entry_path):
            shutil.rmtree(entry_path, ignore_errors=True)
            logger.info("Removed stale session dir on startup: %s", entry_path)


async def _periodic_cleanup():
    while True:
        await asyncio.sleep(60)
        await asyncio.to_thread(session_manager.cleanup_expired)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")
