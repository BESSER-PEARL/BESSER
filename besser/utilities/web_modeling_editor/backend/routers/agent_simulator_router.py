"""
Agent Simulator Router.

Provides endpoints for live agent simulation in the BESSER web modeling editor.
Generates agent code from a diagram, forwards it to the isolated agent
simulator service, and relays the WebSocket connection between the frontend
and the running agent.

Security model:
  * Untrusted agent code only ever runs inside the simulator sandbox. The
    Python code lint applied here (``python_code_validator``) gives users early
    feedback; it is NOT a security boundary.
  * Every call to the simulator carries the shared secret
    ``X-Agent-Simulator-Token`` (env ``AGENT_SIMULATOR_API_TOKEN``).
  * Each session is bound to the actor that created it (GitHub session when
    sign-in is required, client IP otherwise); another actor gets 404.
  * The rate limiter, the per-actor session cap and the ownership map are
    in-process state. This is correct because the backend runs as a single
    uvicorn process (see ``backend.py`` and the Dockerfile ``CMD``); running
    several workers would give each worker its own copy.
"""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from collections import OrderedDict, deque
from threading import Lock
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import httpx
import websockets
from fastapi import APIRouter, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.datastructures import Address

from besser.BUML.metamodel.state_machine.state_machine import CustomCodeAction
from besser.generators.agents.baf_generator import GenerationMode
from besser.utilities.web_modeling_editor.backend.config import get_generator_info
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    AGENT_SIMULATOR_CREATE_TIMEOUT_SECONDS,
    AGENT_SIMULATOR_CPU_CORES,
    AGENT_SIMULATOR_DELETE_TIMEOUT_SECONDS,
    AGENT_SIMULATOR_DISK_MB,
    AGENT_SIMULATOR_FALLBACK_WS_PORT,
    AGENT_SIMULATOR_FILES_TIMEOUT_SECONDS,
    AGENT_SIMULATOR_MAX_SESSIONS_PER_ACTOR,
    AGENT_SIMULATOR_MEMORY_MB,
    AGENT_SIMULATOR_QUOTA_ENABLED,
    AGENT_SIMULATOR_RATE_LIMIT_MAX_KEYS,
    AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS,
    AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS,
    AGENT_SIMULATOR_REQUIRE_AUTH,
    AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE,
    AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS,
    AGENT_SIMULATOR_TOKEN_ENV_VAR,
    AGENT_SIMULATOR_TOKEN_HEADER,
    AGENT_SIMULATOR_URL,
    AGENT_SIMULATOR_WS_AUTH_TIMEOUT_SECONDS,
    AGENT_SIMULATOR_WS_OPEN_TIMEOUT_SECONDS,
    AGENT_TEMP_DIR_PREFIX,
    DEFAULT_AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS,
    OUTPUT_DIR_NAME,
    WS_CLOSE_BAD_REQUEST,
    WS_CLOSE_NOT_FOUND,
    WS_CLOSE_RATE_LIMITED,
    WS_CLOSE_UNAUTHORIZED,
)
from besser.utilities.web_modeling_editor.backend.models import (
    SimulationLimitsResponse,
    SimulationSessionFilesResponse,
    SimulationSessionInput,
    SimulationSessionResponse,
    SimulationSessionStopResponse,
    SimulationValidationResponse,
)
from besser.utilities.web_modeling_editor.backend.routers.auth import require_github_session
from besser.utilities.web_modeling_editor.backend.routers.error_handler import handle_endpoint_errors
from besser.utilities.web_modeling_editor.backend.services.converters import process_agent_diagram
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
    GenerationError,
    ValidationError,
)
from besser.utilities.web_modeling_editor.backend.services.validators.python_code_validator import (
    lint_simulation_tool_code,
    validate_custom_code_action,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/besser_api/simulation", tags=["agent-simulator"])

_CUSTOM_CODE_RESTRICTED_MESSAGE = (
    "Agent simulation is not available for agents with custom Python code state bodies."
)
_SIMULATOR_UNAVAILABLE_MESSAGE = "Agent simulator service is unavailable. Please try again later."

# Test seams: an httpx transport for simulator HTTP calls (None = real network)
# and the coroutine used to open the simulator WebSocket.
_http_transport: Optional[httpx.AsyncBaseTransport] = None
_ws_connect = websockets.connect

# Strong references to fire-and-forget tasks so they are not garbage-collected
# before they finish (see the asyncio.create_task documentation).
_background_tasks: Set["asyncio.Task[None]"] = set()


# ---------------------------------------------------------------------------
# In-process rate limiting and session ownership
# ---------------------------------------------------------------------------

class _SlidingWindowRateLimiter:
    """Per-key sliding-window limiter with bounded memory.

    Expired timestamps are dropped on every check, idle keys are swept once per
    window, and at most ``max_keys`` keys are tracked (least recently used keys
    are evicted first). Single-process only, see the module docstring.
    """

    def __init__(self, window_seconds: int, max_requests: int, max_keys: int):
        self._window_seconds = max(window_seconds, 1)
        self._max_requests = max(max_requests, 1)
        self._max_keys = max(max_keys, 1)
        self._events: "OrderedDict[str, Deque[float]]" = OrderedDict()
        self._last_sweep = time.monotonic()
        self._lock = Lock()

    def __len__(self) -> int:
        return len(self._events)

    def _sweep(self, threshold: float) -> None:
        for key in [k for k, events in self._events.items() if not events or events[-1] < threshold]:
            del self._events[key]

    def check(self, key: str) -> None:
        """Record one request for ``key``; raise HTTP 429 when over the limit."""
        now = time.monotonic()
        threshold = now - self._window_seconds
        with self._lock:
            if now - self._last_sweep >= self._window_seconds:
                self._sweep(threshold)
                self._last_sweep = now

            events = self._events.get(key)
            if events is None:
                if len(self._events) >= self._max_keys:
                    self._sweep(threshold)
                while len(self._events) >= self._max_keys:
                    self._events.popitem(last=False)
                events = deque()
                self._events[key] = events
            else:
                self._events.move_to_end(key)
                while events and events[0] < threshold:
                    events.popleft()

            if len(events) >= self._max_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded for agent simulator endpoints. Please retry shortly.",
                    headers={"Retry-After": str(self._window_seconds)},
                )
            events.append(now)


class _SimulationSessionRegistry:
    """Maps each simulation session to the actor that created it.

    Also enforces the per-actor concurrent-session cap. Entries are forgotten
    when the session is stopped, when its WebSocket closes, or after the
    simulator's session lifetime. Single-process only, see the module docstring.
    """

    def __init__(self, max_sessions_per_actor: int, ttl_seconds: int):
        self._max_sessions_per_actor = max(max_sessions_per_actor, 1)
        self._ttl_seconds = max(ttl_seconds, 1)
        self._sessions: Dict[str, Tuple[str, float]] = {}
        self._lock = Lock()

    def __len__(self) -> int:
        return len(self._sessions)

    def _prune(self, now: float) -> None:
        expired = [sid for sid, (_, created) in self._sessions.items() if now - created >= self._ttl_seconds]
        for sid in expired:
            del self._sessions[sid]

    def reserve(self, actor: str) -> str:
        """Allocate a new session id for ``actor``; raise HTTP 429 when at the cap."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            active = sum(1 for owner, _ in self._sessions.values() if owner == actor)
            if active >= self._max_sessions_per_actor:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"You already have {active} active agent simulation session(s), the maximum allowed. "
                        "Stop it before starting a new one."
                    ),
                )
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = (actor, now)
            return session_id

    def release(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def is_owner(self, session_id: str, actor: str) -> bool:
        with self._lock:
            self._prune(time.monotonic())
            entry = self._sessions.get(session_id)
            return entry is not None and entry[0] == actor

    def require_owner(self, session_id: str, actor: str) -> None:
        """Raise HTTP 404 unless ``actor`` created ``session_id``."""
        if not self.is_owner(session_id, actor):
            raise HTTPException(status_code=404, detail="Simulation session not found.")


_rate_limiter = _SlidingWindowRateLimiter(
    window_seconds=AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS,
    max_requests=AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS,
    max_keys=AGENT_SIMULATOR_RATE_LIMIT_MAX_KEYS,
)

_session_registry = _SimulationSessionRegistry(
    max_sessions_per_actor=AGENT_SIMULATOR_MAX_SESSIONS_PER_ACTOR,
    ttl_seconds=AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS or DEFAULT_AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _authenticate(github_session: Optional[str], client: Optional[Address]) -> str:
    """Apply the sign-in gate and return the actor key the caller is tracked by.

    The actor is the GitHub session when sign-in is required (hashed, so the
    bearer value is never kept as a key or logged) and the client IP otherwise.
    """
    if AGENT_SIMULATOR_REQUIRE_AUTH:
        require_github_session(github_session)
        return "github:" + hashlib.sha256(github_session.encode("utf-8")).hexdigest()
    if client and client.host:
        return f"ip:{client.host}"
    return "anonymous"


def _require_valid_session_id(session_id: str) -> None:
    try:
        uuid.UUID(session_id)
    except (ValueError, TypeError) as exc:
        raise ValidationError("Invalid session id") from exc


def _simulator_headers() -> Dict[str, str]:
    """Return the auth header for the simulator; HTTP 503 when the token is not configured."""
    token = os.environ.get(AGENT_SIMULATOR_TOKEN_ENV_VAR, "").strip()
    if not token:
        logger.error("%s is not set; agent simulation is disabled.", AGENT_SIMULATOR_TOKEN_ENV_VAR)
        raise HTTPException(
            status_code=503,
            detail="Agent simulation is not configured on this server.",
        )
    return {AGENT_SIMULATOR_TOKEN_HEADER: token}


def _simulator_client(timeout: float) -> httpx.AsyncClient:
    """Build an HTTP client for the simulator that sends the auth header."""
    return httpx.AsyncClient(
        base_url=AGENT_SIMULATOR_URL,
        timeout=timeout,
        headers=_simulator_headers(),
        transport=_http_transport,
    )


def _spawn_background(coro) -> None:
    """Run ``coro`` in the background while holding a reference to its task."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _model_has_custom_code_action(model: Dict[str, Any]) -> bool:
    """Return True if any element in the diagram model is a custom Python code action."""
    for element in (model.get("elements") or {}).values():
        if not isinstance(element, dict):
            continue
        if element.get("actionType") == "CustomCodeAction" or element.get("replyType") == "code":
            return True
    return False


def _collect_support_files(output_dir: str) -> Dict[str, str]:
    """Read tools.py plus the skills/ and guis/ files the BAF generator emitted."""
    support_files: Dict[str, str] = {}

    tools_path = os.path.join(output_dir, "tools.py")
    if os.path.exists(tools_path):
        with open(tools_path, encoding="utf-8") as f:
            support_files["tools.py"] = f.read()

    for sub_dir in ("skills", "guis"):
        dir_path = os.path.join(output_dir, sub_dir)
        if not os.path.isdir(dir_path):
            continue
        for file_name in sorted(os.listdir(dir_path)):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, encoding="utf-8") as f:
                    support_files[f"{sub_dir}/{file_name}"] = f.read()

    return support_files


def _lint_agent_code(agent_model) -> None:
    """Lint every CustomCodeAction and Tool.code before anything reaches the simulator.

    This is early user feedback, not a security boundary: the simulator sandbox
    is what isolates untrusted code.
    """
    for state in getattr(agent_model, "states", []):
        for body in filter(None, [getattr(state, "body", None), getattr(state, "fallback_body", None)]):
            for action in getattr(body, "actions", []):
                if isinstance(action, CustomCodeAction):
                    validate_custom_code_action(action.code, simulation=True)
    for tool in getattr(agent_model, "tools", []) or []:
        if getattr(tool, "code", None):
            lint_simulation_tool_code(tool.code)


async def _generate_agent_code_and_config(
    diagram_data: Dict[str, Any],
    config: Dict[str, Any],
    config_yaml: Optional[str],
) -> Tuple[str, str, List[str], Dict[str, str], List[str]]:
    """Convert diagram JSON into agent code and generated config.yaml.

    Returns:
        (agent_code, config_yaml, event_list, support_files, workspace_paths)
    """
    # Merge config into diagram_data so process_agent_diagram reads
    # default_llm_name (and the other config fields) and sets the default LLM;
    # otherwise the BAF template falls back to the first registered LLM.
    agent_model = process_agent_diagram({**diagram_data, "config": config or {}})
    _lint_agent_code(agent_model)

    with tempfile.TemporaryDirectory(prefix=f"{AGENT_TEMP_DIR_PREFIX}simulation_") as temp_dir:
        # The in-memory agent model is handed to the generator directly; the
        # generated code is never exec'd in the backend process, only inside
        # the simulator sandbox.
        generator_class = get_generator_info("agent").generator_class
        generator_output_dir = os.path.join(temp_dir, OUTPUT_DIR_NAME)
        generator = generator_class(
            agent_model,
            output_dir=generator_output_dir,
            config=config or {},
            generation_mode=GenerationMode.CODE_ONLY,
            config_yaml=config_yaml,
            test_mode=True,
        )
        await asyncio.to_thread(generator.generate)

        agent_name = getattr(agent_model, "name", None) or "agent"
        agent_py_path = os.path.join(generator_output_dir, f"{agent_name}.py")
        if not os.path.exists(agent_py_path):
            skip_files = {"tools.py", "personalized_agent_model.py"}
            for fname in sorted(os.listdir(generator_output_dir)):
                if fname.endswith(".py") and fname not in skip_files:
                    agent_py_path = os.path.join(generator_output_dir, fname)
                    break

        if not os.path.exists(agent_py_path):
            raise GenerationError("BAFGenerator did not produce an agent Python file")

        with open(agent_py_path, encoding="utf-8") as f:
            agent_code = f.read()

        config_yaml_path = os.path.join(generator_output_dir, "config.yaml")
        if os.path.exists(config_yaml_path):
            with open(config_yaml_path, encoding="utf-8") as f:
                generated_config_yaml = f.read()
        else:
            generated_config_yaml = f"platforms:\n  websocket:\n    port: {AGENT_SIMULATOR_FALLBACK_WS_PORT}\n"

        event_list: List[str] = []
        for state in getattr(agent_model, "states", []):
            for transition in getattr(state, "transitions", []):
                event = getattr(transition, "event", None)
                if event is None:
                    continue
                event_name = type(event).__name__
                if event_name not in event_list:
                    event_list.append(event_name)

        support_files = _collect_support_files(generator_output_dir)

    workspace_paths = [
        ws.path
        for ws in getattr(agent_model, "workspaces", []) or []
        if isinstance(getattr(ws, "path", None), str) and ws.path.strip()
    ]

    return agent_code, generated_config_yaml, event_list, support_files, workspace_paths


def _credentials_to_env(credentials: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Map the optional LLM credentials to the env vars the BAF agent reads."""
    mapping = {
        "openAiApiKey": "OPENAI_API_KEY",
        "huggingFaceToken": "HUGGINGFACEHUB_API_TOKEN",
        "replicateApiKey": "REPLICATE_API_TOKEN",
    }
    return {env: credentials[key] for key, env in mapping.items() if credentials and credentials.get(key)}


async def _delete_simulator_session(session_id: str) -> None:
    """Best-effort DELETE of a simulator session; failures are logged, not raised."""
    try:
        async with _simulator_client(AGENT_SIMULATOR_DELETE_TIMEOUT_SECONDS) as client:
            resp = await client.delete(f"/sessions/{session_id}")
        if resp.status_code not in (200, 204, 404):
            logger.warning("Simulator refused to delete session %s (HTTP %s)", session_id, resp.status_code)
    except httpx.HTTPError as exc:
        logger.warning("Could not delete simulator session %s: %s", session_id, exc)
    except HTTPException as exc:
        logger.warning("Could not delete simulator session %s: %s", session_id, exc.detail)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@router.get("/limits", response_model=SimulationLimitsResponse)
@handle_endpoint_errors("get_simulation_limits")
async def get_simulation_limits(
    request: Request,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Return the configured resource limits and quota settings for agent simulation."""
    _authenticate(github_session, request.client)
    return SimulationLimitsResponse(
        memoryMb=AGENT_SIMULATOR_MEMORY_MB,
        cpuCores=AGENT_SIMULATOR_CPU_CORES,
        diskMb=AGENT_SIMULATOR_DISK_MB,
        sessionLifetimeSeconds=AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS,
        editorQuotaEnabled=AGENT_SIMULATOR_QUOTA_ENABLED,
    )


@router.post("/validate", response_model=SimulationValidationResponse)
@handle_endpoint_errors("validate_agent_simulation")
async def validate_agent(
    input_data: SimulationSessionInput,
    request: Request,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Check that an agent diagram can be simulated, without creating a session.

    User errors in the diagram (conversion, validation, custom code lint) are
    returned as ``valid=False`` with their messages; unexpected failures
    surface as HTTP errors through ``@handle_endpoint_errors``.
    """
    actor = _authenticate(github_session, request.client)
    _rate_limiter.check(actor)

    if AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE and _model_has_custom_code_action(input_data.model):
        return SimulationValidationResponse(
            valid=False, agentCode="", eventList=[], errors=[_CUSTOM_CODE_RESTRICTED_MESSAGE],
        )

    try:
        agent_code, _config_yaml, event_list, _support_files, _workspaces = await _generate_agent_code_and_config(
            {"title": input_data.title, "model": input_data.model},
            input_data.config or {},
            input_data.configYaml,
        )
    except (ConversionError, ValidationError, ValueError) as exc:
        logger.info("Agent simulation validation failed: %s", exc)
        return SimulationValidationResponse(valid=False, agentCode="", eventList=[], errors=[str(exc)])

    return SimulationValidationResponse(valid=True, agentCode=agent_code, eventList=event_list, errors=[])


@router.post("/sessions", response_model=SimulationSessionResponse)
@handle_endpoint_errors("start_simulation_session")
async def start_simulation_session(
    input_data: SimulationSessionInput,
    request: Request,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Generate agent code and start a simulator session owned by the caller."""
    actor = _authenticate(github_session, request.client)
    _rate_limiter.check(actor)

    if AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE and _model_has_custom_code_action(input_data.model):
        raise HTTPException(status_code=403, detail=_CUSTOM_CODE_RESTRICTED_MESSAGE)

    session_id = _session_registry.reserve(actor)
    try:
        agent_code, config_yaml, event_list, support_files, workspace_paths = await _generate_agent_code_and_config(
            {"title": input_data.title, "model": input_data.model},
            input_data.config or {},
            input_data.configYaml,
        )
        payload = {
            "session_id": session_id,
            "agent_code": agent_code,
            "config_yaml": config_yaml,
            "env_vars": _credentials_to_env(input_data.credentials),
            "event_list": event_list,
            "support_files": support_files,
            "workspace_paths": workspace_paths,
        }
        try:
            async with _simulator_client(AGENT_SIMULATOR_CREATE_TIMEOUT_SECONDS) as client:
                resp = await client.post("/sessions", json=payload)
        except httpx.RequestError as exc:
            logger.error("Cannot reach agent simulator service at %s: %s", AGENT_SIMULATOR_URL, exc)
            raise HTTPException(status_code=503, detail=_SIMULATOR_UNAVAILABLE_MESSAGE) from exc

        if resp.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="The agent simulator is at capacity. Please retry in a few minutes.",
            )
        if resp.status_code != 200:
            logger.error("Simulator rejected session creation (HTTP %s): %s", resp.status_code, resp.text[:500])
            raise HTTPException(status_code=502, detail="Agent simulator service rejected session creation.")
    except BaseException:
        _session_registry.release(session_id)
        raise

    return SimulationSessionResponse(sessionId=session_id, eventList=event_list)


@router.get("/sessions/{session_id}/files", response_model=SimulationSessionFilesResponse)
@handle_endpoint_errors("get_simulation_session_files")
async def get_session_files(
    session_id: str,
    request: Request,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Return the files the caller's running agent produced in its workspace."""
    actor = _authenticate(github_session, request.client)
    _require_valid_session_id(session_id)
    _session_registry.require_owner(session_id, actor)

    try:
        async with _simulator_client(AGENT_SIMULATOR_FILES_TIMEOUT_SECONDS) as client:
            resp = await client.get(f"/sessions/{session_id}/files")
    except httpx.RequestError as exc:
        logger.error("Cannot reach agent simulator service at %s: %s", AGENT_SIMULATOR_URL, exc)
        raise HTTPException(status_code=503, detail=_SIMULATOR_UNAVAILABLE_MESSAGE) from exc

    if resp.status_code == 404:
        _session_registry.release(session_id)
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    if resp.status_code != 200:
        logger.error("Simulator file listing failed (HTTP %s): %s", resp.status_code, resp.text[:500])
        raise HTTPException(status_code=502, detail="Agent simulator service error.")
    try:
        return SimulationSessionFilesResponse.model_validate(resp.json())
    except ValueError as exc:
        logger.error("Simulator returned an invalid file listing: %s", exc)
        raise HTTPException(status_code=502, detail="Agent simulator service error.") from exc


@router.delete("/sessions/{session_id}", response_model=SimulationSessionStopResponse)
@handle_endpoint_errors("stop_simulation_session")
async def stop_simulation_session(
    session_id: str,
    request: Request,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Stop one of the caller's simulation sessions."""
    actor = _authenticate(github_session, request.client)
    _require_valid_session_id(session_id)
    _session_registry.require_owner(session_id, actor)
    _session_registry.release(session_id)
    await _delete_simulator_session(session_id)
    return SimulationSessionStopResponse(ok=True)


# ---------------------------------------------------------------------------
# WebSocket relay
# ---------------------------------------------------------------------------

async def _close_with_error(websocket: WebSocket, code: int, message: str) -> None:
    await websocket.send_text(json.dumps({"type": "error", "message": message}))
    await websocket.close(code=code)


async def _receive_auth_frame(websocket: WebSocket) -> str:
    """Wait for the ``{"type": "auth", "githubSession": ...}`` first frame.

    Returns the GitHub session ('' when none was sent). Raises ValueError when
    the frame is missing, late, or malformed.
    """
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=AGENT_SIMULATOR_WS_AUTH_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise ValueError("Authentication frame not received in time.") from exc
    except KeyError as exc:  # a binary frame has no "text" key
        raise ValueError("Authentication frame must be a text frame.") from exc
    try:
        frame = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Authentication frame must be JSON.") from exc
    if not isinstance(frame, dict) or frame.get("type") != "auth":
        raise ValueError('The first frame must be {"type": "auth", "githubSession": "..."}.')
    github_session = frame.get("githubSession") or ""
    if not isinstance(github_session, str):
        raise ValueError("githubSession must be a string.")
    return github_session


@router.websocket("/{session_id}/ws")
async def simulator_session_ws(websocket: WebSocket, session_id: str):
    """
    Relay WebSocket messages between the frontend and the agent simulator service.

    Protocol: the client's first frame must be
    ``{"type": "auth", "githubSession": "<session or empty>"}`` within
    ``AGENT_SIMULATOR_WS_AUTH_TIMEOUT_SECONDS``. The backend answers
    ``{"type": "auth_ok"}`` and starts relaying, or closes with 4401
    (unauthorized), 4400 (bad session id), 4404 (not the caller's session)
    or 4429 (rate limited).

    Frontend -> simulator service -> agent subprocess
    Agent subprocess -> simulator service -> frontend
    """
    await websocket.accept()

    try:
        _require_valid_session_id(session_id)
    except ValidationError as exc:
        await _close_with_error(websocket, WS_CLOSE_BAD_REQUEST, str(exc))
        return

    try:
        github_session = await _receive_auth_frame(websocket)
    except WebSocketDisconnect:
        return
    except ValueError as exc:
        await _close_with_error(websocket, WS_CLOSE_UNAUTHORIZED, str(exc))
        return

    try:
        actor = _authenticate(github_session, websocket.client)
    except HTTPException as exc:
        await _close_with_error(websocket, WS_CLOSE_UNAUTHORIZED, str(exc.detail))
        return
    try:
        _rate_limiter.check(actor)
    except HTTPException as exc:
        await _close_with_error(websocket, WS_CLOSE_RATE_LIMITED, str(exc.detail))
        return
    if not _session_registry.is_owner(session_id, actor):
        await _close_with_error(websocket, WS_CLOSE_NOT_FOUND, "Simulation session not found.")
        return

    await websocket.send_text(json.dumps({"type": "auth_ok"}))

    ws_base = AGENT_SIMULATOR_URL.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    try:
        simulator_ws = await _ws_connect(
            f"{ws_base}/sessions/{session_id}/ws",
            additional_headers=_simulator_headers(),
            open_timeout=AGENT_SIMULATOR_WS_OPEN_TIMEOUT_SECONDS,
            proxy=None,
        )
    except HTTPException as exc:
        await _close_with_error(websocket, 1011, str(exc.detail))
        return
    except (websockets.exceptions.WebSocketException, OSError, asyncio.TimeoutError) as exc:
        logger.error("Cannot connect to simulator WS for session %s: %s", session_id, exc)
        await _close_with_error(websocket, 1011, "Cannot connect to the agent simulator service.")
        return

    stop = asyncio.Event()

    async def _close_simulator_ws() -> None:
        try:
            await simulator_ws.close()
        except (websockets.exceptions.WebSocketException, OSError) as exc:
            logger.debug("[%s] closing simulator WS failed: %s", session_id, exc)

    async def _frontend_to_simulator():
        try:
            while not stop.is_set():
                msg = await websocket.receive_text()
                await simulator_ws.send(msg)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # relay teardown: any failure just ends this direction
            logger.debug("[%s] frontend->simulator relay ended: %s", session_id, exc)
        finally:
            stop.set()
            # Close the simulator WebSocket so the other direction exits immediately.
            await _close_simulator_ws()

    async def _simulator_to_frontend():
        try:
            async for raw in simulator_ws:
                text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                await websocket.send_text(text)
        except Exception as exc:  # relay teardown: any failure just ends this direction
            logger.debug("[%s] simulator->frontend relay ended: %s", session_id, exc)
        finally:
            stop.set()

    try:
        await asyncio.gather(_frontend_to_simulator(), _simulator_to_frontend(), return_exceptions=True)
    finally:
        await _close_simulator_ws()
        _session_registry.release(session_id)
        _spawn_background(_delete_simulator_session(session_id))
