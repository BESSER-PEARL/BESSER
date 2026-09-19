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

import os
import re as _re

from besser.generators.llm.checkpoint import _SNAPSHOT_DIR


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
