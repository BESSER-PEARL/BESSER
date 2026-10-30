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
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput

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

# Default (non-personalized) museum agent: bundled with the backend and started
# once at boot whenever study mode is on (STUDY_AGENT_URL_PATH set), independent
# of participant deploys.  Uses its own ports/subpath so both agents coexist.
_DEFAULT_AGENT_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "study_default_agent"
)
_DEFAULT_AGENT_DIR = os.path.join(tempfile.gettempdir(), "besser_study_default_agent")
_DEFAULT_AGENT_PID_FILE = os.path.join(tempfile.gettempdir(), "besser_study_default_agent.pid")
_DEFAULT_AGENT_STREAMLIT_PORT = 5001
_DEFAULT_AGENT_WS_PORT = 8767
_DEFAULT_AGENT_URL_PATH = "/study-agent-default"

router = APIRouter(prefix="/besser_api", tags=["study"])


# ── request model ──────────────────────────────────────────────────────────


class StudyDeployRequest(BaseModel):
    """Payload sent by the frontend deploy_study_agent hook."""

    agent_model: Dict[str, Any]
    agent_config: Optional[Dict[str, Any]] = None
    agent_config_yaml: Optional[str] = None
    personalization_mapping: Optional[List[Any]] = None
    web_app_project: ProjectInput
    web_app_versions: List[Dict[str, Any]]


async def _deploy_study_website(request: StudyDeployRequest) -> str:
    """Generate the normal React export and ask the existing frontend to build it."""
    # Defaults fit the docker-compose study setup: the frontend container is
    # reached internally by service name and published through the same
    # reverse proxy (and host) as the editor, under /study-site/.
    internal_url = (os.environ.get("STUDY_FRONTEND_INTERNAL_URL")
                    or os.environ.get("WME_NODE_SERVER_URL") or "http://besser-wme-frontend:8080").rstrip("/")
    public_url = (os.environ.get("STUDY_FRONTEND_PUBLIC_URL") or os.environ.get("DEPLOYMENT_URL", "")).rstrip("/")
    token = os.environ.get("STUDY_DEPLOY_TOKEN", "")
    if not public_url or not token:
        raise HTTPException(status_code=503, detail="Study website deployment requires STUDY_DEPLOY_TOKEN (and DEPLOYMENT_URL or STUDY_FRONTEND_PUBLIC_URL)")
    if len(token) < 32:
        raise HTTPException(status_code=503, detail="STUDY_DEPLOY_TOKEN must contain at least 32 characters")
    internal_origin = urlparse(internal_url)
    public_origin = urlparse(public_url)
    if (internal_origin.scheme not in {"http", "https"}
            or not internal_origin.hostname or internal_origin.path
            or public_origin.scheme not in {"http", "https"}
            or not public_origin.hostname or public_origin.path):
        raise HTTPException(status_code=503, detail="Study frontend URLs must be HTTP(S) origins without a path")

    from besser.utilities.web_modeling_editor.backend.routers.generation_router import _handle_web_app_project_generation

    versions = request.web_app_versions
    if not versions or len(versions) > 8 or any(
            not isinstance(v, dict)
            or not isinstance(v.get("slug"), str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", v["slug"])
            or not isinstance(v.get("guiModel"), dict)
            for v in versions
    ):
        raise HTTPException(status_code=400, detail="Invalid study website versions")
    slugs = [v["slug"] for v in versions]
    if len(slugs) != len(set(slugs)) or "base" not in slugs:
        raise HTTPException(status_code=400, detail="Study website versions must include one Base version and unique names")

    generated = await _handle_web_app_project_generation(
        request.web_app_project,
        get_generator_info("web_app"),
        {"webAppVersions": versions},
    )
    archive = b"".join([chunk async for chunk in generated.body_iterator])
    if len(archive) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Generated study website is too large")

    def send_archive() -> Dict[str, Any]:
        post = urllib.request.Request(
            f"{internal_url}/api/study-site",
            data=archive,
            headers={"Content-Type": "application/zip", "X-Study-Deploy-Token": token},
            method="POST",
        )
        try:
            with urllib.request.urlopen(post, timeout=600) as response:
                return _json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(f"Study website build failed: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Study frontend is unreachable: {exc.reason}") from exc

    try:
        published = await asyncio.to_thread(send_archive)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    published_versions = published.get("versions")
    if (published.get("success") is not True
            or not isinstance(published_versions, list)
            or any(not isinstance(v, str) for v in published_versions)
            or set(published_versions) != set(slugs)):
        raise HTTPException(status_code=502, detail="Study frontend returned an invalid website deployment result")
    return f"{public_url}/study-site/"


# ── process management helpers ─────────────────────────────────────────────


def _kill_study_agent(pid_file: Optional[str] = None) -> None:
    """Kill the currently running study agent (and its children) if any.

    On Windows, waits up to 5 s for the process tree to exit so that file
    handles on the log files are released before the caller clears the
    directory.
    """
    pid_file = pid_file or _STUDY_PID_FILE
    if not os.path.exists(pid_file):
        return
    pid = None
    try:
        with open(pid_file) as fh:
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
            os.remove(pid_file)
        except OSError:
            pass


def _start_study_agent(
    script_name: str,
    agent_dir: Optional[str] = None,
    streamlit_port: Optional[int] = None,
    url_path: Optional[str] = None,
) -> int:
    """Launch the agent as a detached subprocess and return its PID."""
    agent_dir = agent_dir or _STUDY_AGENT_DIR
    streamlit_port = streamlit_port or _STREAMLIT_PORT
    url_path = _STUDY_AGENT_URL_PATH if url_path is None else url_path
    log_path = os.path.join(agent_dir, "application.log")
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
    env.setdefault("STREAMLIT_SERVER_PORT", str(streamlit_port))
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    if url_path:
        env["STREAMLIT_SERVER_BASE_URL_PATH"] = url_path
    logger.info("[study-deploy] Using Python: %s", sys.executable)

    kwargs: Dict[str, Any] = dict(
        cwd=agent_dir,
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
        _inject_server_openai_key(os.path.join(_STUDY_AGENT_DIR, "config.yaml"))
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


# ── default agent ──────────────────────────────────────────────────────────


def _inject_server_openai_key(config_path: str) -> None:
    """Write the server's OPENAI_API_KEY into an agent ``config.yaml``.

    Keeps the key out of the editor/browser: participants never enter it, and
    whatever placeholder the generated config carries is replaced here.
    """
    import yaml

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not os.path.isfile(config_path):
        return
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    config.setdefault("nlp", {}).setdefault("openai", {})["api_key"] = api_key
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)


def _prepare_default_agent_dir() -> str:
    """Copy the default agent to a runtime dir, patching ports and the API key.

    Returns the entry-point script name.  The source directory is never
    modified; ports are rewritten so the default agent does not collide with the
    participant agent, and the OpenAI key is taken from ``OPENAI_API_KEY`` so
    no secret has to live in the bundled ``config.yaml``.
    """
    import yaml

    shutil.rmtree(_DEFAULT_AGENT_DIR, ignore_errors=True)
    shutil.copytree(_DEFAULT_AGENT_SRC, _DEFAULT_AGENT_DIR)

    config_path = os.path.join(_DEFAULT_AGENT_DIR, "config.yaml")
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    ws = config.setdefault("platforms", {}).setdefault("websocket", {})
    ws["host"] = "localhost"
    ws["port"] = _DEFAULT_AGENT_WS_PORT
    streamlit = ws.setdefault("streamlit", {})
    streamlit["host"] = "0.0.0.0"
    streamlit["port"] = _DEFAULT_AGENT_STREAMLIT_PORT
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    _inject_server_openai_key(config_path)

    return "Museum_Agent.py"


def start_default_study_agent() -> Optional[int]:
    """Start the always-on default agent when study mode is on.

    Failures are logged, not raised, so a broken default agent never prevents
    the backend from starting.
    """
    if not _STUDY_AGENT_URL_PATH:
        return None
    try:
        _kill_study_agent(_DEFAULT_AGENT_PID_FILE)
        script = _prepare_default_agent_dir()
        pid = _start_study_agent(
            script,
            agent_dir=_DEFAULT_AGENT_DIR,
            streamlit_port=_DEFAULT_AGENT_STREAMLIT_PORT,
            url_path=_DEFAULT_AGENT_URL_PATH,
        )
        with open(_DEFAULT_AGENT_PID_FILE, "w") as fh:
            fh.write(str(pid))
        logger.info("[study-deploy] Default agent started PID=%d script=%s", pid, script)
        return pid
    except Exception:
        logger.exception("[study-deploy] Could not start the default study agent")
        return None


def stop_default_study_agent() -> None:
    _kill_study_agent(_DEFAULT_AGENT_PID_FILE)


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

    website_url = await _deploy_study_website(request)

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
        "websiteUrl": website_url,
        "websiteVersions": [v["slug"] for v in request.web_app_versions],
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
