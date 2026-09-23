"""
Study Deploy Router

Handles the study-deploy endpoint: generates a BAF agent from an agent diagram,
writes the files to a persistent directory, kills any previously running study
agent, and starts the new one.  Returns the Streamlit URL so participants can
open it directly.

This feature is intentionally separate from the main deployment router so that
it can be toggled (or removed) without touching any other endpoint.
"""

import ast
import importlib.util
import json as _json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from besser.generators.agents.baf_generator import GenerationMode
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.web_modeling_editor.backend.config import get_generator_info
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    AGENT_MODEL_FILENAME,
    AGENT_TEMP_DIR_PREFIX,
)
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_agent_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
    extract_openai_api_key,
    normalize_personalization_mapping,
)
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    generate_user_profile_document as _generate_user_profile_document,
)

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────

_STUDY_AGENT_DIR = os.environ.get(
    "STUDY_AGENT_DIR",
    os.path.join(tempfile.gettempdir(), "besser_study_agent"),
)
_STUDY_SUBMISSIONS_DIR = os.environ.get(
    "STUDY_SUBMISSIONS_DIR",
    os.path.join(tempfile.gettempdir(), "besser_study_submissions"),
)
_STUDY_PID_FILE = os.path.join(tempfile.gettempdir(), "besser_study_agent.pid")
_STREAMLIT_PORT = int(os.environ.get("STUDY_AGENT_STREAMLIT_PORT", "5000"))
# When set (e.g. "/study-agent"), Streamlit is served under that subpath via a
# reverse proxy and the returned URL uses the deployment host + path instead of
# a raw port.  Leave empty for local/direct access.
_STUDY_AGENT_URL_PATH = os.environ.get("STUDY_AGENT_URL_PATH", "").rstrip("/")

router = APIRouter(prefix="/besser_api", tags=["study"])


# ── request model ──────────────────────────────────────────────────────────


class StudyDeployRequest(BaseModel):
    """Payload sent by the frontend deploy_study_agent hook."""

    agent_model: Dict[str, Any]
    agent_config: Optional[Dict[str, Any]] = None
    agent_config_yaml: Optional[str] = None
    personalization_mapping: Optional[List[Any]] = None


# ── process management helpers ─────────────────────────────────────────────


def _kill_study_agent() -> None:
    """Kill the currently running study agent (and its children) if any.

    On Windows, waits up to 5 s for the process tree to exit so that file
    handles on the log files are released before the caller clears the
    directory.
    """
    if not os.path.exists(_STUDY_PID_FILE):
        return
    pid = None
    try:
        with open(_STUDY_PID_FILE) as fh:
            pid = int(fh.read().strip())
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
            )
            # Wait for the process tree to actually exit (handles released)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if str(pid) not in result.stdout:
                    break
                time.sleep(0.2)
        else:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # already gone
    except (ValueError, OSError):
        pass
    finally:
        try:
            os.remove(_STUDY_PID_FILE)
        except OSError:
            pass


def _start_study_agent(script_name: str) -> int:
    """Launch the agent as a detached subprocess and return its PID."""
    log_path = os.path.join(_STUDY_AGENT_DIR, "application.log")
    logger.info("[study-deploy] Agent log: %s", log_path)

    # Prepend the Scripts dirs of the running Python so BAF's subprocess.run(["streamlit",...])
    # finds the correct streamlit.exe (shell=False on Windows only resolves .exe).
    env = os.environ.copy()
    scripts_dirs = _python_scripts_dirs()
    # Also include /baf_packages/bin for Docker deployments where BAF is
    # installed via pip --target /baf_packages (scripts land in its bin/ subdir).
    baf_bin = os.path.join(os.environ.get("BAF_PACKAGES_DIR", "/baf_packages"), "bin")
    env["PATH"] = os.pathsep.join([baf_bin] + scripts_dirs) + os.pathsep + env.get("PATH", "")
    # Force Streamlit to bind on all interfaces (not just localhost) so it is
    # reachable from outside the Docker container.
    env.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    env.setdefault("STREAMLIT_SERVER_PORT", str(_STREAMLIT_PORT))
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    if _STUDY_AGENT_URL_PATH:
        env.setdefault("STREAMLIT_SERVER_BASE_URL_PATH", _STUDY_AGENT_URL_PATH)
    logger.info("[study-deploy] Using Python: %s", sys.executable)

    kwargs: Dict[str, Any] = dict(
        cwd=_STUDY_AGENT_DIR,
        env=env,
    )
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    log_f = open(log_path, "a")  # noqa: SIM115  append so prior runs aren't lost
    try:
        proc = subprocess.Popen(
            [sys.executable, script_name],
            stdout=log_f,
            stderr=log_f,
            **kwargs,
        )
    finally:
        # Close the parent's handle — the subprocess keeps its own.
        log_f.close()

    time.sleep(2)
    if proc.poll() is not None:
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                err_tail = fh.read()[-2000:]
        except OSError:
            err_tail = "(could not read log)"
        logger.error(
            "[study-deploy] Agent process exited immediately (rc=%d):\n%s",
            proc.returncode, err_tail,
        )
        raise RuntimeError(
            f"Agent process exited immediately (rc={proc.returncode}): {err_tail}"
        )

    logger.info("[study-deploy] Agent process alive after 2 s (PID=%d)", proc.pid)
    return proc.pid


# ── generation helper ──────────────────────────────────────────────────────


def _clear_study_agent_dir() -> None:
    """Remove only the *contents* of _STUDY_AGENT_DIR, never the directory itself.

    Files that are still locked by the previous agent process (e.g.
    ``application.log`` written by BAF at runtime) are skipped — they will be
    overwritten when the new agent starts.  ``agent_stdout.log`` /
    ``agent_stderr.log`` now live outside this directory so they never block
    the clear.
    """
    os.makedirs(_STUDY_AGENT_DIR, exist_ok=True)
    for entry in os.scandir(_STUDY_AGENT_DIR):
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
        except PermissionError:
            logger.warning(
                "[study-deploy] Skipping locked file during dir clear: %s", entry.path
            )


def _python_scripts_dirs() -> list:
    """Return the Scripts directories for the running Python, user-install first."""
    import site as _site
    dirs = []
    if _site.ENABLE_USER_SITE:
        # e.g. C:\Users\foo\AppData\Roaming\Python\Python312\Scripts
        user_base = _site.getuserbase()
        ver = f"Python{sys.version_info.major}{sys.version_info.minor}"
        dirs.append(os.path.join(user_base, ver, "Scripts"))
    # e.g. C:\Python312\Scripts
    dirs.append(os.path.join(os.path.dirname(sys.executable), "Scripts"))
    dirs.append(os.path.dirname(sys.executable))
    logger.info("[study-deploy] Resolved Scripts dirs: %s", dirs)
    return dirs


def _generate_study_agent_files(json_data: dict, config: dict, config_yaml: Optional[str] = None) -> str:
    """Generate agent files to the persistent study directory.

    Clears the *contents* of ``_STUDY_AGENT_DIR`` (never the directory itself),
    then writes all BAF agent files (``{agent.name}.py``, ``config.yaml``,
    optional ``tools.py`` / skills, and ``user_profiles.json`` when
    personalization is present).

    Returns the agent entry-point script name, e.g. ``my_agent.py``.
    """
    _clear_study_agent_dir()

    with tempfile.TemporaryDirectory(
        prefix=f"{AGENT_TEMP_DIR_PREFIX}{uuid.uuid4().hex}_"
    ) as tmp:
        agent_file = os.path.join(tmp, AGENT_MODEL_FILENAME)

        # Convert diagram JSON → BUML Agent object
        agent_model_obj = process_agent_diagram(json_data)
        agent_model_to_code(agent_model_obj, agent_file)

        # Validate syntax before executing
        with open(agent_file, encoding="utf-8") as fh:
            src = fh.read()
        try:
            ast.parse(src, filename=agent_file)
        except SyntaxError as exc:
            raise HTTPException(
                status_code=400,
                detail="Generated agent model code has syntax errors",
            ) from exc

        # Import the module so BAFGenerator can access the live Agent object
        spec = importlib.util.spec_from_file_location("study_agent_model", agent_file)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)

        agent_obj = getattr(agent_module, "agent", agent_model_obj)
        agent_name: str = agent_obj.name

        # Run BAFGenerator writing directly to the persistent directory
        generator_class = get_generator_info("agent").generator_class
        generator = generator_class(
            agent_obj,
            output_dir=_STUDY_AGENT_DIR,
            config=config,
            openai_api_key=extract_openai_api_key(config),
            generation_mode=GenerationMode.FULL,
            config_yaml=config_yaml,
        )
        generator.generate()
        logger.info("[study-deploy] Generated agent '%s' in %s", agent_name, _STUDY_AGENT_DIR)

    # Write user_profiles.json alongside the agent files so the BAF runtime can
    # load them for per-user personalisation.
    if isinstance(config.get("personalizationMapping"), list):
        profiles = []
        for idx, entry in enumerate(config["personalizationMapping"]):
            if not isinstance(entry, dict):
                continue
            profile_data = entry.get("user_profile")
            profile_name = entry.get("name") or f"variant_{idx + 1}"
            if profile_data is not None:
                profiles.append({"name": profile_name, "user_profile": profile_data})
        if profiles:
            profiles_path = os.path.join(_STUDY_AGENT_DIR, "user_profiles.json")
            with open(profiles_path, "w", encoding="utf-8") as fh:
                _json.dump(profiles, fh, indent=2)
            logger.info(
                "[study-deploy] Wrote %d user profile(s) to %s",
                len(profiles), profiles_path,
            )

    return f"{agent_name}.py"


# ── endpoint ───────────────────────────────────────────────────────────────


@router.post("/deploy_study_agent")
@handle_endpoint_errors("deploy_study_agent")
async def deploy_study_agent(request: StudyDeployRequest):
    """Deploy a personalised BAF agent for a user study.

    1. Generates agent files to a persistent directory on the server.
    2. Kills any previously running study agent (and its Streamlit child).
    3. Starts the new agent as a detached process.
    4. Returns the Streamlit URL participants can open to test the agent.

    Returns:
        {success: bool, url: str, message: str}
    """
    config: dict = dict(request.agent_config) if request.agent_config else {}
    json_data: dict = {
        "model": request.agent_model,
        "config": config,
    }

    if request.personalization_mapping:
        config["personalizationMapping"] = request.personalization_mapping
        normalize_personalization_mapping(config, json_data, _generate_user_profile_document)
        logger.info(
            "[study-deploy] Personalization mapping with %d variant(s)",
            len(request.personalization_mapping),
        )

    # Kill the previous instance first so its log file handles are released
    # before _generate_study_agent_files clears the directory.
    await asyncio.to_thread(_kill_study_agent)

    agent_script = await asyncio.to_thread(_generate_study_agent_files, json_data, config, request.agent_config_yaml)

    try:
        pid = await asyncio.to_thread(_start_study_agent, agent_script)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info("[study-deploy] Started agent PID=%d  script=%s", pid, agent_script)

    with open(_STUDY_PID_FILE, "w") as fh:
        fh.write(str(pid))

    deployment_url = os.environ.get("DEPLOYMENT_URL", "").rstrip("/")
    if _STUDY_AGENT_URL_PATH and deployment_url:
        # Served through a reverse proxy: use the deployment origin + subpath.
        agent_url = f"{deployment_url}{_STUDY_AGENT_URL_PATH}/"
    else:
        # Direct access: plain HTTP on the Streamlit port.
        host = urlparse(deployment_url).hostname or "localhost" if deployment_url else "localhost"
        agent_url = f"http://{host}:{_STREAMLIT_PORT}"
    agent_name = agent_script[:-3] if agent_script.endswith(".py") else agent_script

    return {
        "success": True,
        "url": agent_url,
        "message": f"Agent '{agent_name}' deployed successfully",
    }


# ── study model upload ─────────────────────────────────────────────────────

_PARTICIPANT_ID_RE = re.compile(r'^[a-f0-9\-]{8,64}$', re.IGNORECASE)


class StudyModelUploadRequest(BaseModel):
    participant_id: str
    project: Dict[str, Any]


@router.post("/upload_study_model")
@handle_endpoint_errors("upload_study_model")
async def upload_study_model(request: StudyModelUploadRequest):
    """Save a participant's project JSON to the study submissions directory.

    The file is named ``<participant_id>.json`` so submissions are trivially
    linked to questionnaire responses that carry the same ID.

    Returns:
        {success: bool, message: str}
    """
    pid = request.participant_id.strip()
    if not _PARTICIPANT_ID_RE.match(pid):
        raise HTTPException(status_code=400, detail="Invalid participant ID format.")

    os.makedirs(_STUDY_SUBMISSIONS_DIR, exist_ok=True)

    # Prevent path traversal: derive the filename purely from the validated ID.
    filename = f"{pid}.json"
    target = os.path.realpath(os.path.join(_STUDY_SUBMISSIONS_DIR, filename))
    if not target.startswith(os.path.realpath(_STUDY_SUBMISSIONS_DIR)):
        raise HTTPException(status_code=400, detail="Invalid participant ID.")

    def _write() -> None:
        with open(target, "w", encoding="utf-8") as fh:
            _json.dump(request.project, fh, indent=2)

    await asyncio.to_thread(_write)
    logger.info("[study-upload] Saved submission for participant %s → %s", pid, target)

    return {"success": True, "message": "Model uploaded successfully."}
