"""Required frontend-build evidence, without granting shell/install permission."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable

from besser.generators.llm.execution.process import _safe_subprocess_env
from besser.generators.llm.validation.issues import required_check_unverified, required_dependency_setup

_EXCLUDED = {"node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build",
             ".next", "target", ".gradle"}
_APP_FRAMEWORKS = {"next", "nuxt", "@angular/core", "@sveltejs/kit"}


def collect_frontend_build_issues(
    output_dir: str, *, enabled: bool, allow_shell: bool,
    source_revision: Callable[[], str], successful_builds: dict,
    can_run: Callable[[], bool], timeout: float = 120,
) -> list[str]:
    """Build discovered web apps only when explicitly authorized and installed.

    A generic successful shell command is not build evidence. Cache only this
    harness-owned check, for unchanged source and dependency-install markers.
    Cache entries are in-memory only; a resumed run checks again.
    """
    workspace = os.path.realpath(output_dir)
    issues = []

    def contained(path: str) -> bool:
        try:
            return os.path.commonpath([workspace, os.path.realpath(path)]) == workspace
        except ValueError:
            return False

    for folder, dirs, files in os.walk(workspace):
        dirs[:] = sorted(d for d in dirs if d not in _EXCLUDED
                         and not d.startswith(".besser_") and contained(os.path.join(folder, d)))
        if "package.json" not in files:
            continue
        rel = os.path.relpath(folder, workspace).replace("\\", "/")
        label = f"frontend build [{rel}]"
        html_entry = any(os.path.isfile(os.path.join(folder, path))
                         for path in ("index.html", "public/index.html"))
        manifest_path = os.path.realpath(os.path.join(folder, "package.json"))
        try:
            if not contained(manifest_path):
                if html_entry:
                    issues.append(required_check_unverified(label, "package manifest is outside the workspace"))
                continue
            with open(manifest_path, encoding="utf-8-sig") as handle:
                manifest = json.load(handle)
            if not isinstance(manifest, dict):
                raise ValueError("package.json must contain an object")
        except (OSError, ValueError) as exc:
            if html_entry:
                issues.append(f"{label}: cannot read a valid package.json: {exc}")
            continue
        dependencies = {
            name for key in ("dependencies", "devDependencies")
            for name in (manifest.get(key) if isinstance(manifest.get(key), dict) else {})
        }
        framework_app = bool(dependencies & _APP_FRAMEWORKS) and any(
            os.path.isdir(os.path.join(folder, path))
            for path in ("app", "pages", "src", "server"))
        if not html_entry and not framework_app:
            continue  # libraries and unrelated Node packages are not web apps
        scripts = manifest.get("scripts") or {}
        build = scripts.get("build") if isinstance(scripts, dict) else None
        reason = None
        if not enabled:
            reason = "toolchain validation is disabled"
        elif not allow_shell:
            reason = "configured build scripts require explicitly enabled shell tools"
        elif not isinstance(build, str) or not build.strip():
            reason = "package.json has no configured build script"
        elif not can_run():
            reason = "validation was stopped or its budget exhausted"
        npm = None if reason else shutil.which("npm") or shutil.which("npm.cmd")
        if not reason and not npm:
            reason = "npm is unavailable"
        if reason:
            issues.append(required_check_unverified(label, reason))
            continue
        if not os.path.isdir(os.path.join(folder, "node_modules")):
            issues.append(required_dependency_setup(label, rel))
            continue  # the agent may use its authorized install tool; never install here
        revision = source_revision()
        markers = []
        for path in ("node_modules", "node_modules/.package-lock.json"):
            try:
                stat = os.stat(os.path.join(folder, path))
                markers.append((stat.st_mtime_ns, stat.st_size))
            except OSError:
                markers.append(None)
        key = (rel, revision, tuple(markers), build)
        if successful_builds.get(key):
            continue
        env = _safe_subprocess_env()
        env["CI"] = "true"
        try:
            result = subprocess.run(
                [npm, "run", "build"], cwd=folder, env=env,
                capture_output=True, text=True, timeout=max(1, min(timeout, 120)),
            )
        except subprocess.TimeoutExpired:
            issues.append(required_check_unverified(label, "configured build timed out"))
            continue
        except OSError as exc:
            issues.append(required_check_unverified(label, f"could not launch build: {exc}"))
            continue
        if result.returncode != 0:
            output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
            issues.append(f"{label}: configured npm run build failed (exit {result.returncode}): "
                          + (output[-2500:] or "no diagnostic output"))
        elif source_revision() != revision:
            issues.append(required_check_unverified(label, "source changed during build; rerun against the final files"))
        else:
            successful_builds[key] = True
    return issues
