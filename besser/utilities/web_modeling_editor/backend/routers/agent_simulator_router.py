"""
Agent Simulator Router.

Provides endpoints for live agent simulation in the BESSER web modeling editor.
Generates agent code from a diagram, forwards it to the agent simulator service,
and relays the WebSocket connection between the frontend and the running agent.
"""

import asyncio
import importlib.util
import json
import logging
import os
import tempfile
import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import httpx
import websockets
from fastapi import APIRouter, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from besser.generators.agents.baf_generator import GenerationMode
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.config import get_generator_info
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    AGENT_MODEL_FILENAME,
    AGENT_TEMP_DIR_PREFIX,
    OUTPUT_DIR_NAME,
)
from besser.utilities.web_modeling_editor.backend.services.converters import process_agent_diagram
from besser.utilities.web_modeling_editor.backend.services.deployment.github_oauth import get_user_token
from besser.utilities.web_modeling_editor.backend.services.exceptions import GenerationError

logger = logging.getLogger(__name__)

# Agent simulator base URL. Keep legacy env var names for compatibility.
SIMULATOR_URL = (
    os.environ.get("AGENT_SIMULATOR_URL")
    or os.environ.get("AGENT_SIMULATOR_URL")
    or os.environ.get("AGENT_SANDBOX_URL")
    or "http://besser-agent-simulator:8001"
)

_AGENT_SIMULATOR_REQUIRE_AUTH = (
    os.environ.get("AGENT_SIMULATOR_REQUIRE_AUTH")
    or os.environ.get("AGENT_SIMULATION_REQUIRE_AUTH")
    or os.environ.get("AGENT_TEST_REQUIRE_AUTH")
    or "true"
).strip().lower() in {"1", "true", "yes", "on"}

_RATE_LIMIT_WINDOW_SECONDS = int(
    os.environ.get("AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS")
    or os.environ.get("AGENT_SIMULATION_RATE_LIMIT_WINDOW_SECONDS")
    or os.environ.get("AGENT_TEST_RATE_LIMIT_WINDOW_SECONDS")
    or "60"
)

_RATE_LIMIT_MAX_REQUESTS = int(
    os.environ.get("AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS")
    or os.environ.get("AGENT_SIMULATION_RATE_LIMIT_MAX_REQUESTS")
    or os.environ.get("AGENT_TEST_RATE_LIMIT_MAX_REQUESTS")
    or "12"
)


class _SlidingWindowRateLimiter:
    def __init__(self, window_seconds: int, max_requests: int):
        self._window_seconds = max(window_seconds, 1)
        self._max_requests = max(max_requests, 1)
        self._events: Dict[str, List[float]] = {}
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        threshold = now - self._window_seconds
        with self._lock:
            events = [ts for ts in self._events.get(key, []) if ts >= threshold]
            if len(events) >= self._max_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded for agent simulator endpoints. Please retry shortly.",
                )
            events.append(now)
            self._events[key] = events


_rate_limiter = _SlidingWindowRateLimiter(
    window_seconds=_RATE_LIMIT_WINDOW_SECONDS,
    max_requests=_RATE_LIMIT_MAX_REQUESTS,
)


def _is_valid_session_id(session_id: str) -> bool:
    try:
        uuid.UUID(session_id)
        return True
    except (ValueError, TypeError):
        return False


def _resolve_actor_key(
    github_session: Optional[str],
    request: Optional[Request],
    websocket: Optional[WebSocket] = None,
) -> str:
    if github_session:
        return f"github:{github_session}"
    if request and request.client and request.client.host:
        return f"ip:{request.client.host}"
    if websocket and websocket.client and websocket.client.host:
        return f"ip:{websocket.client.host}"
    return "anonymous"


def _enforce_agent_simulator_auth(github_session: Optional[str]) -> None:
    if not _AGENT_SIMULATOR_REQUIRE_AUTH:
        return
    if not github_session:
        raise HTTPException(
            status_code=401,
            detail="GitHub authentication required for agent simulation. Please sign in first.",
        )
    if not get_user_token(github_session):
        raise HTTPException(
            status_code=401,
            detail="GitHub session expired. Please sign in again.",
        )


router = APIRouter(prefix="/besser_api/simulation", tags=["agent-simulator"])
legacy_router = APIRouter(prefix="/besser_api/test", tags=["agent-simulator-legacy"])


class StartSimulationSessionRequest(BaseModel):
    title: str
    model: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    configYaml: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None


class StartSimulationSessionResponse(BaseModel):
    sessionId: str
    eventList: List[str]


class ValidateResponse(BaseModel):
    valid: bool
    agentCode: str
    eventList: List[str]
    errors: List[str]


class SessionFile(BaseModel):
    path: str
    content: str


class SessionFilesResponse(BaseModel):
    files: List[SessionFile]
    directories: List[str] = Field(default_factory=list)


async def _generate_agent_code_and_config(
    diagram_data: dict,
    config: dict,
    config_yaml: Optional[str],
) -> Tuple[str, str, List[str], Dict[str, str], List[str]]:
    """Convert diagram JSON into agent code and generated config.yaml."""

    def _collect_support_files(output_dir: str) -> Dict[str, str]:
        support_files: Dict[str, str] = {}

        tools_path = os.path.join(output_dir, "tools.py")
        if os.path.exists(tools_path):
            with open(tools_path, encoding="utf-8") as f:
                support_files["tools.py"] = f.read()

        skills_dir = os.path.join(output_dir, "skills")
        if os.path.isdir(skills_dir):
            for skill_name in sorted(os.listdir(skills_dir)):
                skill_path = os.path.join(skills_dir, skill_name)
                if not os.path.isfile(skill_path):
                    continue
                rel_path = os.path.join("skills", skill_name).replace("\\", "/")
                with open(skill_path, encoding="utf-8") as f:
                    support_files[rel_path] = f.read()

        return support_files

    agent_model = process_agent_diagram(diagram_data)

    with tempfile.TemporaryDirectory(prefix=f"{AGENT_TEMP_DIR_PREFIX}simulation_") as temp_dir:
        agent_file = os.path.join(temp_dir, AGENT_MODEL_FILENAME)
        agent_model_to_code(agent_model, agent_file)

        spec = importlib.util.spec_from_file_location("_simulation_agent_model", agent_file)
        if spec is None or spec.loader is None:
            raise GenerationError("Failed to prepare dynamic import for generated agent model")
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        the_agent = getattr(agent_module, "agent", agent_model)

        generator_info = get_generator_info("agent")
        generator_class = generator_info.generator_class
        generator_output_dir = os.path.join(temp_dir, OUTPUT_DIR_NAME)

        generator = generator_class(
            the_agent,
            output_dir=generator_output_dir,
            config=config or {},
            generation_mode=GenerationMode.CODE_ONLY,
            config_yaml=config_yaml,
            test_mode=True,
        )
        await asyncio.to_thread(generator.generate)

        agent_name = getattr(the_agent, "name", None) or "agent"
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
            generated_config_yaml = "platforms:\n  websocket:\n    port: 7700\n"

        event_list: List[str] = []
        if hasattr(the_agent, "states"):
            for state in the_agent.states:
                for transition in getattr(state, "transitions", []):
                    event = getattr(transition, "event", None)
                    if event is None:
                        continue
                    event_name = type(event).__name__
                    if event_name not in event_list:
                        event_list.append(event_name)

        support_files = _collect_support_files(generator_output_dir)

        workspace_paths: List[str] = []
        for ws in getattr(the_agent, "workspaces", []) or []:
            ws_path = getattr(ws, "path", None)
            if isinstance(ws_path, str) and ws_path.strip():
                workspace_paths.append(ws_path)

        return agent_code, generated_config_yaml, event_list, support_files, workspace_paths


async def _cleanup_simulator_session(session_id: str) -> None:
    """Best-effort DELETE of an simulator session (called on disconnect)."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.delete(f"{SIMULATOR_URL}/sessions/{session_id}")
    except Exception:
        pass


@router.post("/sessions", response_model=StartSimulationSessionResponse)
@legacy_router.post("/sessions", response_model=StartSimulationSessionResponse)
async def start_simulation_session(
    request: StartSimulationSessionRequest,
    http_request: Request,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Generate agent code and create an simulator session."""
    _enforce_agent_simulator_auth(github_session)
    _rate_limiter.check(_resolve_actor_key(github_session, http_request))
    session_id = str(uuid.uuid4())

    diagram_data = {"title": request.title, "model": request.model}

    try:
        agent_code, config_yaml, event_list, support_files, workspace_paths = await _generate_agent_code_and_config(
            diagram_data,
            request.config or {},
            request.configYaml,
        )
    except Exception as exc:
        logger.exception("Code generation failed for simulator session")
        raise HTTPException(status_code=400, detail=f"Failed to generate agent code: {exc}") from exc

    env_vars: Dict[str, str] = {}
    if request.credentials:
        if request.credentials.get("openAiApiKey"):
            env_vars["OPENAI_API_KEY"] = request.credentials["openAiApiKey"]
        if request.credentials.get("huggingFaceToken"):
            env_vars["HUGGINGFACEHUB_API_TOKEN"] = request.credentials["huggingFaceToken"]
        if request.credentials.get("replicateApiKey"):
            env_vars["REPLICATE_API_TOKEN"] = request.credentials["replicateApiKey"]

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{SIMULATOR_URL}/sessions",
                json={
                    "session_id": session_id,
                    "agent_code": agent_code,
                    "config_yaml": config_yaml,
                    "env_vars": env_vars,
                    "event_list": event_list,
                    "support_files": support_files,
                    "workspace_paths": workspace_paths,
                },
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Agent simulator service rejected session creation: {resp.text}",
                )
    except httpx.RequestError as exc:
        logger.exception("Cannot reach agent simulation service at %s", SIMULATOR_URL)
        raise HTTPException(
            status_code=503,
            detail="Agent simulator service is unavailable. Please try again later.",
        ) from exc

    return StartSimulationSessionResponse(sessionId=session_id, eventList=event_list)


@router.post("/validate", response_model=ValidateResponse)
@legacy_router.post("/validate", response_model=ValidateResponse)
async def validate_agent(
    request: StartSimulationSessionRequest,
    http_request: Request,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Validate agent code generation without creating a simulation session."""
    _enforce_agent_simulator_auth(github_session)
    _rate_limiter.check(_resolve_actor_key(github_session, http_request))
    diagram_data = {"title": request.title, "model": request.model}

    try:
        agent_code, _config_yaml, event_list, _support_files, _workspace_paths = await _generate_agent_code_and_config(
            diagram_data,
            request.config or {},
            request.configYaml,
        )
    except Exception as exc:
        logger.exception("Code generation failed during validation")
        return ValidateResponse(valid=False, agentCode="", eventList=[], errors=[str(exc)])

    return ValidateResponse(valid=True, agentCode=agent_code, eventList=event_list, errors=[])


@router.get("/sessions/{session_id}/files", response_model=SessionFilesResponse)
@legacy_router.get("/sessions/{session_id}/files", response_model=SessionFilesResponse)
async def get_session_files(
    session_id: str,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Proxy file listing from the simulation service for the given session."""
    _enforce_agent_simulator_auth(github_session)
    if not _is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{SIMULATOR_URL}/sessions/{session_id}/files")
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Session not found in simulation service")
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Simulation service error: {resp.text}")
            return resp.json()
    except HTTPException:
        raise
    except httpx.RequestError as exc:
        logger.exception("Cannot reach agent simulation service at %s", SIMULATOR_URL)
        raise HTTPException(status_code=503, detail="Agent simulator service is unavailable.") from exc


@router.delete("/sessions/{session_id}")
@legacy_router.delete("/sessions/{session_id}")
async def stop_simulation_session(
    session_id: str,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Terminate an agent simulation session."""
    _enforce_agent_simulator_auth(github_session)
    if not _is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    await _cleanup_simulator_session(session_id)
    return {"ok": True}


@router.websocket("/{session_id}/ws")
@legacy_router.websocket("/{session_id}/ws")
async def simulator_session_ws(websocket: WebSocket, session_id: str):
    """
    Relay WebSocket messages between the frontend and the agent simulator service.

    Frontend -> simulator service -> agent subprocess
    Agent subprocess -> simulator service -> frontend
    Agent stdout events -> frontend (via simulator service)
    """
    github_session = websocket.query_params.get("github_session")
    actor_key = _resolve_actor_key(github_session, request=None, websocket=websocket)

    try:
        _enforce_agent_simulator_auth(github_session)
        _rate_limiter.check(actor_key)
        if not _is_valid_session_id(session_id):
            raise HTTPException(status_code=400, detail="Invalid session id")
    except HTTPException as exc:
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "error", "message": exc.detail}))
        await websocket.close(code=4401 if exc.status_code == 401 else 4400)
        return

    await websocket.accept()

    simulator_ws_url = SIMULATOR_URL.replace("https://", "wss://").replace("http://", "ws://")
    simulator_ws_url = f"{simulator_ws_url}/sessions/{session_id}/ws"

    try:
        simulator_ws = await websockets.connect(simulator_ws_url, open_timeout=15)
    except Exception as exc:
        logger.exception("Cannot connect to simulator WS for session %s", session_id)
        await websocket.send_text(
            json.dumps({"type": "error", "message": f"Cannot connect to agent simulator service: {exc}"})
        )
        await websocket.close()
        return

    stop = asyncio.Event()

    async def _frontend_to_simulator():
        try:
            while not stop.is_set():
                msg = await websocket.receive_text()
                await simulator_ws.send(msg)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.debug("[%s] frontend->simulator relay ended: %s", session_id, exc)
        finally:
            stop.set()
            # Close simulator service websocket so receiver exits immediately.
            try:
                await simulator_ws.close()
            except Exception:
                pass

    async def _simulator_to_frontend():
        try:
            async for raw in simulator_ws:
                text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                await websocket.send_text(text)
        except Exception as exc:
            logger.debug("[%s] simulator->frontend relay ended: %s", session_id, exc)
        finally:
            stop.set()

    try:
        await asyncio.gather(
            _frontend_to_simulator(),
            _simulator_to_frontend(),
            return_exceptions=True,
        )
    finally:
        try:
            await simulator_ws.close()
        except Exception:
            pass
        asyncio.create_task(_cleanup_simulator_session(session_id))
