"""Render deploy files for an app in its own repository.

"Go live" deploys the code that is in the project's repo, which is the code
the Spec-Driven Agent wrote. The agent does not always lay an app out the way
the deterministic web-app deploy expects (``backend/`` and ``frontend/`` at
the root). It puts a web app under ``web_app/``, and its frontends build to
``build`` or ``dist``. So the layout is read from the files, and a layout BESSER
cannot deploy is refused with a reason instead of guessed.

The same service writer backs the deterministic deploy
(``github_deploy_api._add_deployment_configs``), so both produce one format.
"""

import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

PYTHON_VERSION = "3.12.3"
# Render's one-click Blueprint deploy; takes ``?repo=<GitHub URL>``.
RENDER_DEPLOY_URL = "https://render.com/deploy"

# First line of every render.yaml BESSER writes. A repo whose render.yaml
# starts with it gets the file regenerated on each push; any other
# render.yaml is the user's and is never touched.
MANAGED_MARKER = "# Managed by BESSER: regenerated on every push from the editor."

# Render caps service hostnames at this many characters and truncates longer
# ones at dash boundaries, which drops the suffix that keeps them unique.
RENDER_HOSTNAME_MAX = 40

# Saves that only touch these paths change the model, not the app; Render
# skips the rebuild for them.
_MODEL_ONLY_PATHS = ("buml/**", ".besser/**")

_SKIPPED_DIRS = {"node_modules", ".git", "__pycache__", "venv", ".venv", "dist", "build", ".agent-trace"}
# How deep to look for the app; the agent nests it one or two levels down.
_MAX_DEPTH = 3
_FASTAPI_APP_RE = re.compile(r"^(\w+)\s*(?::[^=]+)?=\s*FastAPI\(", re.MULTILINE)
_VITE_OUT_DIR_RE = re.compile(r"""outDir\s*:\s*["']([^"']+)["']""")
_BACKEND_NAME_RE = re.compile(r"^\s*name:\s*\S+-backend-([a-f0-9]{4,16})\s*$", re.MULTILINE)


@dataclass(frozen=True)
class AppLayout:
    """Where the parts of an app live, relative to the repo root (``/`` separators)."""
    backend_dir: str
    backend_app: str  # the FastAPI instance in main_api.py
    frontend_dir: Optional[str] = None
    frontend_out: str = "build"
    warnings: tuple = field(default_factory=tuple)


class LayoutRefused(ValueError):
    """The repo holds no app BESSER can deploy; ``str()`` is the reason for the user."""


def fit_service_name(readable: str, stable_key: str) -> str:
    """A Render-safe service name: ``readable``, or ``besser-<hash of stable_key>`` when too long."""
    if len(readable) <= RENDER_HOSTNAME_MAX:
        return readable
    return f"besser-{hashlib.sha1(stable_key.encode('utf-8')).hexdigest()[:12]}"


def existing_suffix(render_yaml: Optional[str]) -> Optional[str]:
    """The service suffix of a render.yaml BESSER wrote, so a redeploy keeps the same services and URLs."""
    match = _BACKEND_NAME_RE.search(render_yaml or "")
    return match.group(1) if match else None


def is_managed(render_yaml: Optional[str]) -> bool:
    return bool(render_yaml) and render_yaml.startswith(MANAGED_MARKER)


def _rel(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _walk_dirs(root: str):
    """Yield every directory under ``root`` (itself included) down to ``_MAX_DEPTH``, skipping build output."""
    for current, dirs, files in os.walk(root):
        depth = 0 if current == root else _rel(root, current).count("/") + 1
        dirs[:] = [] if depth >= _MAX_DEPTH else [d for d in dirs if d not in _SKIPPED_DIRS]
        yield current, set(files), set(dirs)


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def detect_layout(root: str) -> AppLayout:
    """Find the deployable app in the checked-out repo at ``root``.

    Supported: a FastAPI backend (``main_api.py`` beside ``requirements.txt``)
    and, optionally, a Vite frontend (``package.json`` beside ``vite.config``).
    Raises :class:`LayoutRefused` for anything else.
    """
    backends, frontends, agent_dirs, unpinned = [], [], [], []
    for current, files, dirs in _walk_dirs(root):
        if {"main_api.py", "requirements.txt"} <= files:
            backends.append(current)
        elif "main_api.py" in files:
            unpinned.append(current)
        if "package.json" in files and any(f.startswith("vite.config.") for f in files):
            frontends.append(current)
        if "agents" in dirs and os.path.isdir(os.path.join(current, "agents")):
            agent_dirs.append(current)

    if not backends and unpinned:
        raise LayoutRefused(
            f"{_rel(root, os.path.join(unpinned[0], 'main_api.py'))} has no requirements.txt beside it, "
            "so Render cannot install the backend. Ask the agent to add one."
        )
    if not backends:
        markers = [f for _c, files, _d in _walk_dirs(root) for f in files if f in ("manage.py", "pubspec.yaml")]
        if markers:
            raise LayoutRefused(
                "Go live supports apps with a FastAPI backend for now; this repository holds a "
                f"{'Django' if 'manage.py' in markers else 'Flutter'} app."
            )
        raise LayoutRefused(
            "No FastAPI backend found (a main_api.py next to a requirements.txt). "
            "Build the app with the Spec-Driven Agent first, then save it to GitHub."
        )
    if len(backends) > 1:
        raise LayoutRefused(
            "This repository holds more than one backend ("
            + ", ".join(_rel(root, b) for b in backends) + "), so it is unclear which one to deploy."
        )
    if len(frontends) > 1:
        raise LayoutRefused(
            "This repository holds more than one frontend ("
            + ", ".join(_rel(root, f) for f in frontends) + "), so it is unclear which one to deploy."
        )
    if agent_dirs:
        raise LayoutRefused(
            "Apps with built-in agents cannot go live from their repository yet. "
            "Use the classic deploy for now, which rebuilds the app from the diagrams."
        )

    backend = backends[0]
    app_match = _FASTAPI_APP_RE.search(_read(os.path.join(backend, "main_api.py")))
    if not app_match:
        raise LayoutRefused(
            f"{_rel(root, os.path.join(backend, 'main_api.py'))} does not create a FastAPI app, "
            "so there is nothing for Render to start."
        )

    warnings = []
    frontend_dir, frontend_out = None, "build"
    if frontends:
        frontend = frontends[0]
        frontend_dir = _rel(root, frontend)
        config = next((_read(os.path.join(frontend, f)) for f in sorted(os.listdir(frontend))
                       if f.startswith("vite.config.")), "")
        out_match = _VITE_OUT_DIR_RE.search(config)
        frontend_out = out_match.group(1).strip("./") if out_match else "dist"  # Vite's default
        if not _frontend_reads_api_url(frontend):
            warnings.append(
                "The frontend never reads VITE_API_URL, so it may keep calling localhost once live. "
                "Ask the agent to read the API address from VITE_API_URL."
            )
    else:
        warnings.append("No frontend found: only the API goes live.")

    return AppLayout(
        backend_dir=_rel(root, backend),
        backend_app=app_match.group(1),
        frontend_dir=frontend_dir,
        frontend_out=frontend_out,
        warnings=tuple(warnings),
    )


def _frontend_reads_api_url(frontend: str) -> bool:
    for current, dirs, files in os.walk(os.path.join(frontend, "src")):
        dirs[:] = [d for d in dirs if d not in _SKIPPED_DIRS]
        for name in files:
            if name.endswith((".ts", ".tsx", ".js", ".jsx")) and "VITE_API_URL" in _read(os.path.join(current, name)):
                return True
    return False


@dataclass(frozen=True)
class ServiceNames:
    backend: str
    frontend: str

    @classmethod
    def for_app(cls, app_name: str, suffix: str) -> "ServiceNames":
        # Render rewrites underscores to dashes in hostnames; the URLs baked into
        # the frontend must match.
        host = app_name.replace("_", "-")
        return cls(
            backend=fit_service_name(f"{host}-backend-{suffix}", f"{app_name}:backend"),
            frontend=fit_service_name(f"{host}-frontend-{suffix}", f"{app_name}:frontend"),
        )


def web_services(
    names: ServiceNames,
    backend_dir: str = "backend",
    backend_app: str = "app",
    frontend_dir: Optional[str] = "frontend",
    frontend_out: str = "build",
    build_filter: bool = False,
) -> str:
    """The ``services:`` block for a FastAPI backend and an optional static frontend."""
    filter_block = ""
    if build_filter:
        ignored = "".join(f"\n        - {path}" for path in _MODEL_ONLY_PATHS)
        filter_block = f"\n    buildFilter:\n      ignoredPaths:{ignored}"
    config = f"""services:
  # Backend API (Free tier - 750 hours/month, spins down after 15 min idle)
  - type: web
    name: {names.backend}
    runtime: python
    plan: free
    buildCommand: pip install -r {backend_dir}/requirements.txt
    startCommand: cd {backend_dir} && uvicorn main_api:{backend_app} --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: {PYTHON_VERSION}{filter_block}
"""
    if frontend_dir:
        config += f"""
  # Frontend (Free static site)
  - type: web
    name: {names.frontend}
    runtime: static
    buildCommand: cd {frontend_dir} && npm install && npm run build
    staticPublishPath: {frontend_dir}/{frontend_out}
    envVars:
      - key: VITE_API_URL
        value: https://{names.backend}.onrender.com
    headers:
      - path: /assets/*.js
        name: Content-Type
        value: application/javascript
      - path: /assets/*.css
        name: Content-Type
        value: text/css
    routes:
      - type: rewrite
        source: /*
        destination: /index.html{filter_block}
"""
    return config


def deploy_files(layout: AppLayout, app_name: str, suffix: Optional[str] = None) -> tuple[dict[str, str], ServiceNames]:
    """The Render files for ``layout`` (repo path -> content) and the service names they use."""
    names = ServiceNames.for_app(app_name, suffix or uuid.uuid4().hex[:6])
    render_yaml = MANAGED_MARKER + "\n" + web_services(
        names, layout.backend_dir, layout.backend_app, layout.frontend_dir, layout.frontend_out, build_filter=True,
    )
    files = {"render.yaml": render_yaml}
    if layout.frontend_dir:
        files[f"{layout.frontend_dir}/.env.production"] = (
            "# Production environment variables, written by BESSER for Render\n"
            f"VITE_API_URL=https://{names.backend}.onrender.com\n"
        )
    return files, names
