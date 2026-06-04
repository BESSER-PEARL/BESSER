"""
BESSER Agent Sandbox API

Isolated execution environment for user-designed BAF agents.
Runs as a separate container reachable only from the backend on an internal
Docker network.  Each POST /sessions spawns a subprocess with resource limits
and a dedicated WebSocket port.  The /sessions/{id}/ws endpoint bridges the
backend WebSocket to the running agent.
"""
import asyncio
import json
import logging
import re
from typing import Dict, List

import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from session_manager import session_manager

logging.basicConfig(
    level=logging.INFO,
    format="{levelname} - {asctime}: {message}",
    style="{",
)
logger = logging.getLogger(__name__)

_STATE_BODY_LOG_RE = re.compile(r"\[(?P<state>[^]]+)\]\s+Running body\b")

app = FastAPI(title="BESSER Agent Sandbox API", docs_url=None, redoc_url=None)

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


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Start a new agent sandbox session."""
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
        return CreateSessionResponse(session_id=session.session_id, port=session.port)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create session %s", request.session_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Terminate a sandbox session and clean up its resources."""
    await asyncio.to_thread(session_manager.terminate_session, session_id)
    return {"ok": True}


@app.get("/health")
async def health():
    """Health check — also exposes current session count."""
    return {
        "status": "ok",
        "sessions": session_manager.get_session_count(),
    }


# ---------------------------------------------------------------------------
# WebSocket relay
# ---------------------------------------------------------------------------

@app.websocket("/sessions/{session_id}/ws")
async def session_ws(websocket: WebSocket, session_id: str):
    """
    Relay WebSocket between the backend and the running agent subprocess.

    Message routing:
    - Backend → agent:    user messages forwarded to agent's WebSocket server.
    - Agent → backend:    agent replies forwarded to backend.
    - Agent stdout:       state-body log lines are parsed via regex and
                          forwarded as structured events; other lines are
                          forwarded as ``{"type":"stdout","line":"..."}`` messages.
    """
    await websocket.accept()

    session = session_manager.get_session(session_id)
    if not session:
        await websocket.send_text(json.dumps({"type": "error", "message": "Session not found"}))
        await websocket.close(code=4004)
        return

    # Wait up to 15 seconds for the agent's WebSocket server to start
    agent_ws_url = f"ws://localhost:{session.port}/"
    agent_ws = None
    for attempt in range(30):
        try:
            agent_ws = await websockets.connect(agent_ws_url, open_timeout=2)
            logger.info("[%s] Connected to agent WS on port %d", session_id, session.port)
            break
        except Exception:
            if not session.is_alive():
                break
            await asyncio.sleep(0.5)

    if agent_ws is None:
        # Give the process a moment to flush remaining output, then read what's available
        await asyncio.sleep(0.3)
        stdout_text = ""
        try:
            # read1 returns available bytes without blocking on EOF (safe even if process is alive)
            raw = await asyncio.wait_for(
                asyncio.to_thread(session.process.stdout.read1, 65536),  # type: ignore[attr-defined]
                timeout=2.0,
            )
            stdout_text = raw.decode("utf-8", errors="replace") if raw else ""
        except Exception:
            pass

        # Forward each stdout line to the terminal pane so the user can see the traceback
        if stdout_text:
            for line in stdout_text.splitlines():
                try:
                    await websocket.send_text(json.dumps({"type": "stdout", "line": line}))
                except Exception:
                    break

        detail = stdout_text.strip() or "No output captured — the process may have crashed silently."
        await websocket.send_text(
            json.dumps({
                "type": "error",
                "message": f"Agent subprocess failed to start its WebSocket server.\n\n{detail}",
            })
        )
        await websocket.close()
        await asyncio.to_thread(session_manager.terminate_session, session_id)
        return

    stop_event = asyncio.Event()

    async def _read_stdout():
        """Read agent stdout; forward inferred state-change events and lines."""
        loop = asyncio.get_event_loop()
        try:
            while not stop_event.is_set():
                line_bytes = await loop.run_in_executor(None, session.process.stdout.readline)
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                state_match = _STATE_BODY_LOG_RE.search(line)
                if state_match:
                    try:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "state_change",
                                    "state": state_match.group("state"),
                                    "_source": "log_state",
                                }
                            )
                        )
                    except Exception:
                        break
                try:
                    await websocket.send_text(json.dumps({"type": "stdout", "line": line}))
                except Exception:
                    break
        except Exception as exc:
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
            pass
        except Exception as exc:
            logger.debug("[%s] frontend→agent relay ended: %s", session_id, exc)
        finally:
            stop_event.set()
            # Close agent WS so _agent_to_frontend's async-for exits immediately
            # instead of blocking until the agent sends its next message.
            try:
                await agent_ws.close()
            except Exception:
                pass

    async def _agent_to_frontend():
        """Forward messages from the agent to the backend/frontend."""
        try:
            async for raw in agent_ws:
                text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                await websocket.send_text(text)
        except Exception as exc:
            logger.debug("[%s] agent→frontend relay ended: %s", session_id, exc)
        finally:
            stop_event.set()

    try:
        await asyncio.gather(
            _read_stdout(),
            _frontend_to_agent(),
            _agent_to_frontend(),
            return_exceptions=True,
        )
    finally:
        try:
            await agent_ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Background cleanup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _start_cleanup_task():
    _remove_stale_session_dirs()
    asyncio.create_task(_periodic_cleanup())


def _remove_stale_session_dirs() -> None:
    """Remove any /tmp/sessions/* directories left over from a previous run."""
    import os as _os, shutil as _shutil
    sessions_root = "/tmp/sessions"
    if not _os.path.isdir(sessions_root):
        return
    for entry in _os.listdir(sessions_root):
        entry_path = _os.path.join(sessions_root, entry)
        if _os.path.isdir(entry_path):
            _shutil.rmtree(entry_path, ignore_errors=True)
            logger.info("Removed stale session dir on startup: %s", entry_path)


async def _periodic_cleanup():
    while True:
        await asyncio.sleep(60)
        session_manager.cleanup_expired()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
