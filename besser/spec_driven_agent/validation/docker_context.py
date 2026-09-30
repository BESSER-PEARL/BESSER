"""Dockerfile COPY sources that do not exist in the build context.

Docker resolves a COPY source against the build context, not the Dockerfile's
own directory: a root ``Dockerfile.frontend`` with ``COPY frontend/package*.json``
is correct. The context is the compose ``build.context`` when a compose file
names this Dockerfile, else the Dockerfile's directory (``docker build .``
beside it). ``COPY --from=<stage>`` copies from a build stage, not the context.

Report-only: nothing here writes to the workspace.
"""

from __future__ import annotations

import json
import os
import shlex
from besser.spec_driven_agent.execution.workspace_fs import walk_plain

_COMPOSE_NAMES = ("docker-compose.yml", "docker-compose.yaml",
                  "compose.yml", "compose.yaml")
# Only the manifests a build cannot proceed without; other sources are the
# model's business and a wrong guess about them would be a false blocker.
_CHECKED_MANIFESTS = ("package.json", "requirements.txt")


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def compose_build_contexts(output_dir: str) -> dict[str, str]:
    """``{normalised Dockerfile path: build context dir}`` from compose files."""
    try:
        import yaml
    except ImportError:
        return {}
    contexts: dict[str, str] = {}
    for root, dirs, files in walk_plain(output_dir):
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git") and not d.startswith(".")]
        for name in files:
            if name not in _COMPOSE_NAMES:
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8") as f:
                    services = (yaml.safe_load(f) or {}).get("services") or {}
            except Exception:
                continue
            for service in services.values() if isinstance(services, dict) else ():
                build = service.get("build") if isinstance(service, dict) else None
                if isinstance(build, str):
                    build = {"context": build}
                if not isinstance(build, dict):
                    continue
                context = os.path.join(root, str(build.get("context") or "."))
                dockerfile = os.path.join(context, str(build.get("dockerfile") or "Dockerfile"))
                contexts[_norm(dockerfile)] = context
    return contexts


def _copy_sources(content: str) -> list[str]:
    """Context-relative sources of every COPY/ADD that reads the context."""
    sources: list[str] = []
    for line in content.replace("\\\n", " ").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or parts[0].upper() not in ("COPY", "ADD"):
            continue
        flags, rest = [], parts[1].strip()
        while rest.startswith("--"):
            flag, _, rest = rest.partition(" ")
            flags.append(flag)
            rest = rest.strip()
        if any(flag.startswith("--from") for flag in flags):
            continue
        if rest.startswith("["):
            try:
                args = [str(a) for a in json.loads(rest)]
            except ValueError:
                continue
        else:
            try:
                args = shlex.split(rest)
            except ValueError:
                continue
        sources.extend(args[:-1])
    return sources


def missing_copy_manifests(dockerfile: str, content: str,
                           contexts: dict[str, str]) -> list[str]:
    """Paths of package.json / requirements.txt COPY sources that do not exist."""
    context = contexts.get(_norm(dockerfile), os.path.dirname(dockerfile))
    missing: list[str] = []
    for src in _copy_sources(content):
        base = os.path.basename(src.rstrip("/"))
        # ``package*.json`` needs at least package.json; a bare directory or
        # ``.`` is not a manifest reference.
        manifest = "package.json" if base == "package*.json" else base
        if manifest not in _CHECKED_MANIFESTS:
            continue
        path = os.path.normpath(os.path.join(context, os.path.dirname(src), manifest))
        if not os.path.isfile(path):
            missing.append(path)
    return missing


def dockerfile_copy_issues(output_dir: str, dockerfile: str, content: str,
                           contexts: dict[str, str]) -> list[str]:
    """One issue per missing package.json / requirements.txt COPY source."""
    rel = os.path.relpath(dockerfile, output_dir).replace("\\", "/")
    return [
        f"{rel} references {os.path.relpath(path, output_dir).replace(chr(92), '/')} "
        "but it doesn't exist"
        for path in missing_copy_manifests(dockerfile, content, contexts)
    ]
