"""Repairs to the deterministic scaffold that need no model and no LLM.

Phase 1 always writes a correct ``requirements.txt`` and a Dockerfile that
matches what it generated. A weak Phase-2 model sometimes deletes the
requirements file while "cleaning up", or leaves a ``COPY package-lock.json``
behind after removing the lockfile - either of which fails the image build
for a reason that has nothing to do with the application. These functions put
that back deterministically.

Split out of ``orchestrator.py``: they take a path and return a value, touch
no run state, and are repair rather than validation - which is why they sit
beside ``import_repair.py`` rather than under ``validation/``.
"""

from __future__ import annotations

import json
import os
import re as _re

from besser.spec_driven_agent.checkpoint import _SNAPSHOT_DIR


# The deterministic Phase-1 backend generator always writes a correct
# requirements.txt, but a weak Phase-2 model sometimes deletes it while
# customizing — leaving a Dockerfile that ``COPY requirements.txt``s a file that
# no longer exists (a hard build blocker the weak model then can't self-repair).
# Restore it deterministically instead of relying on the LLM fix loop.
_DEFAULT_BACKEND_REQUIREMENTS = (
    "fastapi>=0.103.0\n"
    "uvicorn>=0.15.0\n"
    "pydantic>=2.0.0\n"
    "typing-extensions>=4.6.0\n"
    "sqlalchemy>=2.0.0\n"
    "python-multipart>=0.0.6\n"
)

# import token -> pip requirement, for the few extras Phase 2 commonly adds
# (auth, http). Kept small and high-confidence; a baseline is better than a
# missing file that breaks the whole image build.
_IMPORT_TO_REQUIREMENT = {
    "jose": "python-jose[cryptography]>=3.3.0",
    "passlib": "passlib[bcrypt]>=1.7.4",
    "bcrypt": "bcrypt>=4.0.0",
    "jwt": "pyjwt>=2.8.0",
    "httpx": "httpx>=0.27.0",
    "requests": "requests>=2.31.0",
    "dotenv": "python-dotenv>=1.0.0",
    "aiofiles": "aiofiles>=23.0.0",
    "alembic": "alembic>=1.13.0",
}

def _is_dockerfile(fname: str) -> bool:
    """True for any Dockerfile naming convention, not just the bare name.

    Covers ``Dockerfile``, ``Dockerfile.frontend`` / ``.backend`` / ``.dev``
    (the common multi-service layout) and the ``frontend.Dockerfile`` spelling.
    """
    low = fname.lower()
    return (
        low == "dockerfile"
        or low.startswith("dockerfile.")
        or low.endswith(".dockerfile")
    )


def _strip_missing_lockfile_copy(content: str) -> str:
    """Remove package-lock.json from Dockerfile COPY lines.

    Handles the two shapes an LLM writes:

        COPY frontend/package.json frontend/package-lock.json ./
        COPY package-lock.json ./

    The first loses just the lockfile argument; the second has nothing left to
    copy, so the whole line goes.
    """
    out: list[str] = []
    for line in content.splitlines():
        if "package-lock.json" not in line or not line.lstrip().upper().startswith("COPY"):
            out.append(line)
            continue
        parts = line.split()
        kept = [p for p in parts if "package-lock.json" not in p]
        # COPY + at least one source + one destination is the minimum that still
        # copies something; below that the line only existed for the lockfile.
        if len(kept) >= 3:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(indent + " ".join(kept))
        # else: drop the line entirely
    return "\n".join(out) + ("\n" if content.endswith("\n") else "")


def _project_has_npm_lockfile(output_dir: str) -> bool:
    """True when the project contains an npm lockfile anywhere.

    ``npm ci`` refuses to run without one, whatever directory the Dockerfile
    builds from.
    """
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", _SNAPSHOT_DIR)]
        if "package-lock.json" in files or "npm-shrinkwrap.json" in files:
            return True
    return False


def _ensure_requirements_txt(docker_dir: str) -> bool:
    """Write a sensible requirements.txt next to a Dockerfile when it's missing.

    Base FastAPI stack plus a few extras inferred from the Python imports that
    actually appear beside the Dockerfile. Returns True when a file was written.
    """
    req_path = os.path.join(docker_dir, "requirements.txt")
    if os.path.isfile(req_path):
        return False
    extras: set = set()
    for root, _, files in os.walk(docker_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            try:
                with open(os.path.join(root, fn), "r", encoding="utf-8") as f:
                    src = f.read()
            except Exception:
                continue
            for token, pkg in _IMPORT_TO_REQUIREMENT.items():
                if _re.search(rf"^\s*(?:import|from)\s+{token}\b", src, _re.MULTILINE):
                    extras.add(pkg)
    content = _DEFAULT_BACKEND_REQUIREMENTS + "".join(sorted(e + "\n" for e in extras))
    try:
        with open(req_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception:
        return False


# ======================================================================
# Frontend scaffold repair (the flat backend/ + frontend/ app shape)
# ======================================================================
#
# The class-diagram path has no GUI model, so ``generate_web_app`` is never
# offered (tools.py requires {domain, gui}) and the frontend is written
# file-by-file by the LLM against the orchestrator's frontend checklist. That
# checklist names package.json / index.html / main.jsx / App.jsx / api.js /
# pages and nothing else, so the build configuration around them is left to
# chance. Across the 192 recorded runs under verification/spec-iterations:
#
#   192/192  no vite config of any kind
#   168/192  @vitejs/plugin-react not even a dependency
#    96/192  a JSX file with no `import React` -> with esbuild's default
#            classic transform the page dies on mount with
#            "ReferenceError: React is not defined" (reproduced in Chrome)
#   185/192  a hardcoded http://localhost:8000 with no env override
#    32/192  imports axios without declaring it
#    20/192  "build": "vite", which starts a dev server instead of building
#     0/192  a Dockerfile or a compose file
#
# All of it is decidable and fixable from the files on disk. These functions
# do that, deterministically and idempotently, instead of asking the model.

_VITE_CONFIG_PREFIX = "vite.config"

# Frameworks/plugins that already give JSX the automatic runtime. Mirrors
# validation/frontend_resolution.py::_automatic_jsx_runtime so a repaired
# project also passes that check.
_AUTOMATIC_JSX_PACKAGES = frozenset({
    "@vitejs/plugin-react", "@vitejs/plugin-react-swc", "react-scripts",
    "next", "@preact/preset-vite", "@babel/preset-react",
    "@rsbuild/plugin-react",
})

_PLUGIN_REACT_VERSION = "^4.3.4"
_VITE_VERSION = "^5.4.8"

# Bare imports a generated frontend commonly makes without declaring. Only
# packages actually observed, with a pinned major: a guessed package is a
# failing install, which is worse than the missing declaration.
_FRONTEND_IMPORT_TO_DEPENDENCY = {
    "axios": "^1.7.7",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.2",
}

_FRONTEND_SOURCE_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs")

_FRONTEND_SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__",
                       "dist", "build", ".next", "out", "coverage", _SNAPSHOT_DIR}

# A complete string literal holding a localhost API base. `$` and `{` are
# excluded from the path so a template literal with an interpolation
# (`http://localhost:8000/order/${id}`) never matches - rewriting one would
# drop the interpolation.
_LOCALHOST_BASE = (
    r"""(?P<q>['"`])"""
    r"""(?P<url>https?://(?:localhost|127\.0\.0\.1):8000(?:/[^'"`${}\s]*)?)"""
    r"""(?P=q)"""
)

# Two shapes, both of which accept a parenthesised expression in place of the
# literal. A JSX attribute (href="...") deliberately does NOT match: there
# `(expr)` without braces is a syntax error.
_API_BASE_DECL_RE = _re.compile(
    r"^(?P<lead>[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+[A-Za-z_$][\w$]*[ \t]*=[ \t]*)"
    + _LOCALHOST_BASE,
    _re.MULTILINE,
)
_API_BASE_OPTION_RE = _re.compile(
    r"(?P<lead>\bbaseURL[ \t]*:[ \t]*)" + _LOCALHOST_BASE
)

_IMPORT_SPEC_RE = _re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*|\bimport\s+)['"]([^'"\n]+)['"]"""
)

_COMPOSE_NAMES = ("docker-compose.yml", "docker-compose.yaml",
                  "compose.yml", "compose.yaml")

_VITE_CONFIG_TEMPLATE = """\
import {{ defineConfig }} from 'vite';
import react from '@vitejs/plugin-react';

// @vitejs/plugin-react enables the automatic JSX runtime, so a component that
// renders JSX without importing React still mounts. Without it Vite falls
// back to esbuild's classic transform and the page is blank with
// "ReferenceError: React is not defined".
export default defineConfig({{
  plugins: [react()],
  server: {{ host: true, port: {port} }},
  preview: {{ host: true, port: {port} }},
  build: {{ outDir: '{out_dir}' }},
}});
"""

_FRONTEND_DOCKERFILE_TEMPLATE = """\
FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Baked in at build time: Vite inlines import.meta.env.VITE_* into the bundle.
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

RUN npm install -g serve

EXPOSE 3000

# -s rewrites unknown paths to index.html so client-side routes work.
CMD ["serve", "-s", "{out_dir}", "-l", "3000"]
"""

_BACKEND_DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim

WORKDIR /app

RUN mkdir -p /app/data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD {cmd}
"""

_COMPOSE_TEMPLATE = """\
services:
  backend:
    build:
      context: ./{backend}
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - backend_data:/app/data

  frontend:
    build:
      context: ./{frontend}
      dockerfile: Dockerfile
      args:
        VITE_API_URL: http://localhost:8000
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  backend_data:
"""

_README_TEMPLATE = """\
# {title}

A FastAPI backend (`{backend}/`) and a React + Vite frontend (`{frontend}/`).

## Run it

```bash
docker compose up --build
```

| | |
| --- | --- |
| Web app | http://localhost:3000 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

## Run it without Docker

```bash
cd {backend} && pip install -r requirements.txt && python {entry}
cd {frontend} && npm install && npm run dev
```

## Pointing the frontend somewhere else

The frontend reads its API base from `VITE_API_URL` and falls back to
`http://localhost:8000`. If port 8000 is taken, start the backend on another
port and set the variable rather than editing the source:

```bash
cd {frontend} && VITE_API_URL=http://localhost:8080 npm run dev
```

Vite inlines `VITE_*` at build time, so a production image needs it as a
build argument - `docker-compose.yml` already passes it that way.
"""


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    except OSError:
        return ""


def _write_text(path: str, content: str) -> bool:
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        return True
    except OSError:
        return False


def _iter_project_dirs(root: str):
    """Yield (dir, filenames) for the tree, skipping build/vendor folders."""
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if d not in _FRONTEND_SKIP_DIRS and not d.startswith(".besser_")]
        yield folder, files


def _declared(manifest: dict) -> set:
    names: set = set()
    for key in ("dependencies", "devDependencies", "peerDependencies",
                "optionalDependencies"):
        section = manifest.get(key)
        if isinstance(section, dict):
            names.update(section)
    return names


def _find_vite_react_frontends(output_dir: str) -> list:
    """Directories holding a Vite+React frontend, as (dir, manifest) pairs."""
    found = []
    for folder, files in _iter_project_dirs(output_dir):
        if "package.json" not in files:
            continue
        try:
            manifest = json.loads(_read_text(os.path.join(folder, "package.json")) or "{}")
        except ValueError:
            continue
        if not isinstance(manifest, dict):
            continue
        declared = _declared(manifest)
        scripts = manifest.get("scripts") or {}
        uses_vite = "vite" in declared or any(
            "vite" in value for value in scripts.values() if isinstance(value, str))
        if "react" in declared and uses_vite:
            found.append((folder, manifest))
    return found


def _existing_vite_config(frontend_dir: str):
    try:
        names = sorted(os.listdir(frontend_dir))
    except OSError:
        return None
    for name in names:
        if name.startswith(_VITE_CONFIG_PREFIX):
            return os.path.join(frontend_dir, name)
    return None


def _vite_out_dir(frontend_dir: str) -> str:
    config = _existing_vite_config(frontend_dir)
    if config:
        match = _re.search(r"outDir\s*:\s*['\"]([^'\"]+)['\"]", _read_text(config))
        if match:
            return match.group(1).strip("./") or "dist"
    return "dist"


def _rewrite_api_base(frontend_dir: str) -> list:
    """Read the API base from VITE_API_URL, keeping the literal as fallback.

    A hardcoded ``http://localhost:8000`` means a busy port 8000 can only be
    worked around by editing delivered source, and a container build has no
    way to point the bundle anywhere else at all.
    """
    changed = []

    def _sub(match):
        return (f"{match.group('lead')}(import.meta.env.VITE_API_URL || "
                f"{match.group('q')}{match.group('url')}{match.group('q')})")

    for folder, files in _iter_project_dirs(frontend_dir):
        for fname in files:
            if not fname.endswith(_FRONTEND_SOURCE_EXT):
                continue
            path = os.path.join(folder, fname)
            original = _read_text(path)
            if "localhost:8000" not in original and "127.0.0.1:8000" not in original:
                continue
            updated = original
            # Idempotent: a line that already reads the env var has
            # `import.meta.env` between `=` and the literal, so `lead`
            # cannot match it.
            for pattern in (_API_BASE_DECL_RE, _API_BASE_OPTION_RE):
                updated = pattern.sub(_sub, updated)
            if updated != original and _write_text(path, updated):
                changed.append(os.path.relpath(path, frontend_dir).replace("\\", "/"))
    return changed


def _bare_imports(frontend_dir: str) -> set:
    names: set = set()
    for folder, files in _iter_project_dirs(frontend_dir):
        for fname in files:
            if not fname.endswith(_FRONTEND_SOURCE_EXT):
                continue
            for spec in _IMPORT_SPEC_RE.findall(_read_text(os.path.join(folder, fname))):
                if spec.startswith((".", "/")):
                    continue
                parts = spec.split("/")
                names.add("/".join(parts[:2]) if spec.startswith("@") else parts[0])
    return names


def _repair_frontend_manifest(frontend_dir: str, manifest: dict) -> list:
    """Fix the package.json defects that stop the app building or loading."""
    repairs = []
    scripts = manifest.setdefault("scripts", {})
    dev_deps = manifest.setdefault("devDependencies", {})
    deps = manifest.setdefault("dependencies", {})
    declared = _declared(manifest)

    # "build": "vite" starts a dev server and never emits a bundle, so an
    # image build hangs and the served directory stays empty.
    if str(scripts.get("build", "")).strip() in ("", "vite"):
        scripts["build"] = "vite build"
        repairs.append('scripts.build -> "vite build"')
    if not str(scripts.get("dev", "")).strip():
        scripts["dev"] = "vite"
        repairs.append('scripts.dev -> "vite"')

    if not declared & _AUTOMATIC_JSX_PACKAGES:
        dev_deps["@vitejs/plugin-react"] = _PLUGIN_REACT_VERSION
        repairs.append("devDependencies += @vitejs/plugin-react")
    if "vite" not in declared:
        dev_deps["vite"] = _VITE_VERSION
        repairs.append("devDependencies += vite")

    declared = _declared(manifest)
    for name in sorted(_bare_imports(frontend_dir) - declared):
        version = _FRONTEND_IMPORT_TO_DEPENDENCY.get(name)
        if version:
            deps[name] = version
            repairs.append(f"dependencies += {name}")

    if not dev_deps:
        manifest.pop("devDependencies", None)
    if not deps:
        manifest.pop("dependencies", None)
    return repairs


def _backend_entry_command(backend_dir: str):
    """Docker CMD (JSON-array text) for a generated FastAPI backend."""
    candidates = []
    try:
        names = sorted(os.listdir(backend_dir))
    except OSError:
        return None
    for fname in names:
        if not fname.endswith(".py"):
            continue
        source = _read_text(os.path.join(backend_dir, fname))
        if _re.search(r"^\s*app\s*=\s*FastAPI\(", source, _re.MULTILINE):
            candidates.append((fname, source))
    if not candidates:
        return None
    # The deterministic BackendGenerator writes main_api.py; prefer it when a
    # run has left several FastAPI modules around.
    candidates.sort(key=lambda item: (item[0] != "main_api.py", item[0]))
    fname, source = candidates[0]
    if "__main__" in source and "uvicorn.run" in source:
        return f'["python", "{fname}"]'
    return f'["uvicorn", "{fname[:-3]}:app", "--host", "0.0.0.0", "--port", "8000"]'


def _find_backend_dir(output_dir: str):
    """The single direct subdirectory holding a runnable Python backend."""
    matches = []
    try:
        names = sorted(os.listdir(output_dir))
    except OSError:
        return None
    for name in names:
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path) or name in _FRONTEND_SKIP_DIRS or name.startswith("."):
            continue
        if os.path.isfile(os.path.join(path, "requirements.txt")) and \
                _backend_entry_command(path) is not None:
            matches.append(path)
    return matches[0] if len(matches) == 1 else None


def _ensure_deployment_files(output_dir: str, frontend_dir: str) -> list:
    """Dockerfiles + compose + README, only for the plain two-service shape.

    Skipped entirely when the app already ships any Docker file: a
    hand-written deployment belongs to its author, and a second one would be
    a second source of truth.
    """
    repairs: list = []
    for _folder, files in _iter_project_dirs(output_dir):
        if any(_is_dockerfile(f) or f.lower() in _COMPOSE_NAMES for f in files):
            return repairs

    backend_dir = _find_backend_dir(output_dir)
    if backend_dir is None:
        return repairs
    if os.path.dirname(os.path.abspath(frontend_dir)) != os.path.abspath(output_dir):
        return repairs
    cmd = _backend_entry_command(backend_dir)
    if cmd is None:
        return repairs

    out_dir = _vite_out_dir(frontend_dir)
    frontend_name = os.path.basename(frontend_dir)
    backend_name = os.path.basename(backend_dir)

    if _write_text(os.path.join(frontend_dir, "Dockerfile"),
                   _FRONTEND_DOCKERFILE_TEMPLATE.format(out_dir=out_dir)):
        repairs.append(f"{frontend_name}/Dockerfile")
    if _write_text(os.path.join(backend_dir, "Dockerfile"),
                   _BACKEND_DOCKERFILE_TEMPLATE.format(cmd=cmd)):
        repairs.append(f"{backend_name}/Dockerfile")
    if _write_text(os.path.join(output_dir, "docker-compose.yml"),
                   _COMPOSE_TEMPLATE.format(backend=backend_name, frontend=frontend_name)):
        repairs.append("docker-compose.yml")

    readme = os.path.join(output_dir, "README.md")
    if not os.path.isfile(readme):
        entry = _re.search(r'"(\w+\.py)"', cmd)
        title = os.path.basename(os.path.abspath(output_dir)).replace("_", " ").strip()
        if _write_text(readme, _README_TEMPLATE.format(
                title=title or "Generated app",
                backend=backend_name, frontend=frontend_name,
                entry=entry.group(1) if entry else "main_api.py")):
            repairs.append("README.md")
    return repairs


def ensure_frontend_scaffold(output_dir: str) -> list:
    """Make an LLM-authored React frontend actually render, deterministically.

    Writes the build configuration the frontend checklist never asks for: a
    Vite config with ``@vitejs/plugin-react`` (which is also what makes a
    missing ``import React`` harmless), an env-driven API base, a working
    ``build`` script, the two Dockerfiles and a compose file, and a README.

    Idempotent, and never overwrites an existing Vite config, Dockerfile,
    compose file or README - a project that already has one keeps it.
    Returns one description per repair, for the run log.
    """
    repairs: list = []
    for frontend_dir, manifest in _find_vite_react_frontends(output_dir):
        label = os.path.relpath(frontend_dir, output_dir).replace("\\", "/")
        label = "" if label == "." else label + "/"

        if _existing_vite_config(frontend_dir) is None:
            # .mjs, not .js: always parsed as ESM whether or not package.json
            # declares "type": "module".
            if _write_text(os.path.join(frontend_dir, "vite.config.mjs"),
                           _VITE_CONFIG_TEMPLATE.format(port=3000, out_dir="dist")):
                repairs.append(f"{label}vite.config.mjs (React automatic JSX runtime)")

        manifest_repairs = _repair_frontend_manifest(frontend_dir, manifest)
        if manifest_repairs and _write_text(
                os.path.join(frontend_dir, "package.json"),
                json.dumps(manifest, indent=2) + "\n"):
            repairs.extend(f"{label}package.json: {item}" for item in manifest_repairs)

        for rel in _rewrite_api_base(frontend_dir):
            repairs.append(f"{label}{rel}: API base reads VITE_API_URL")

        repairs.extend(_ensure_deployment_files(output_dir, frontend_dir))
    return repairs
