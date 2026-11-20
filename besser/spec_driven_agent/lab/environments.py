"""Each change's environments (venvs, ``node_modules``), cloned from stores shared by every change.

Stores, under ``<root>/.cache`` (``LAB_CACHE_DIRNAME``, on the trees' filesystem so files can be linked)::

    uv/                       uv's cache: every venv's packages are linked from it
    node/<lock sha256>/       a node_modules snapshot of the base editor: node_modules/, packages/*/node_modules/

How files are shared depends on the filesystem (:func:`probe_link_mode`): copy-on-write reflinks where it has
them, hard links otherwise. With hard links an in-place write into a linked file would reach every change, so the
snapshot's files are made read-only.

An environment is made when a check or the preview first needs it (:meth:`Environments.ensure`) and stamped with a
hash of the files it was made from, so a second call is a no-op until those files change. Installs run through a
runner the caller gives: for a change's tree a sandboxed one, since its requirements and lock file may be
model-written and installing runs their build scripts. Linking ``node_modules`` from the snapshot runs here,
unconfined, because the sandbox hides the stores; it copies only the snapshot and runs nothing from the tree.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Optional, Sequence

from besser.spec_driven_agent.execution.process import run_bounded
from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_FRONTEND, REPO_MODELING_AGENT, RepoSpec
from besser.utilities.secret_redaction import redact_text
from besser.utilities.subprocess_env import safe_subprocess_env

logger = logging.getLogger(__name__)

#: "reflink": copy-on-write clones; "hardlink": shared inodes, snapshot files read-only.
LinkMode = Literal["reflink", "hardlink"]

#: ``runner(argv, cwd) -> (exit code, combined output)``; the sandboxed one for a change's tree.
CommandRunner = Callable[[Sequence[str], str], tuple[int, str]]

#: Written at the root of each repository's tree; the ``.besser_`` prefix keeps it out of the change's files.
STAMP_FILENAME = ".besser_lab_env.json"

# The files each environment is made from, relative to its tree; a folder counts with every file under it.
_INPUTS = {
    REPO_BESSER: ("setup.cfg", "pyproject.toml", "requirements.txt",
                  "besser/utilities/web_modeling_editor/backend/requirements.txt", "docs/requirements.txt"),
    REPO_FRONTEND: ("package-lock.json",),
    REPO_MODELING_AGENT: ("requirements-dev.txt", "pyproject.toml", "patches"),
}

# The Python versions the repositories' own Docker images use.
_BESSER_PYTHON = "3.12"
_AGENT_PYTHON = "3.11"

# What BESSER's CI runs for the import layers (CLAUDE.md); without it the lab's import check skips.
_IMPORT_LINTER = "import-linter==2.15"

# uv's --link-mode for each link mode.
_UV_LINK_MODE = {"reflink": "clone", "hardlink": "hardlink"}

# npm ci of the editor and the agent's TensorFlow wheels take minutes on a cold cache.
_INSTALL_TIMEOUT_SECONDS = 1800.0

# A failed install's message keeps the end of its output, where npm and uv put the error.
_OUTPUT_TAIL_CHARS = 2000

# The modeling agent's fix over its installed BAF, applied as its Dockerfile does; find_spec imports nothing.
_AGENT_PATCH = "patches/websocket_platform.py"
_FIND_BAF_PLATFORM = ("import importlib.util, os\n"
                      "print(os.path.join(importlib.util.find_spec('baf').submodule_search_locations[0],\n"
                      "                   'platforms', 'websocket', 'websocket_platform.py'))")

_NPM_CI = ("npm", "ci", "--no-audit", "--no-fund", "--loglevel=error")

_HARDLINK_WARNING = ("No reflinks in %s (%s): node_modules and venvs are hard-linked from the shared stores and "
                     "the node_modules snapshot is made read-only. In-place writes into node_modules or a venv "
                     "can reach other changes.")


class LabEnvironmentError(Exception):
    """An environment could not be made; the message says which command or check failed."""


@dataclass(frozen=True)
class EnvironmentStamp:
    """What :meth:`Environments.ensure` made in one tree, stored as :data:`STAMP_FILENAME`."""

    repo: str
    #: hash of the files the environment was made from
    inputs_sha256: str
    created_at: float
    seconds: float
    #: "linked" from the node_modules snapshot, or "installed" in the tree (every venv: uv links from its cache)
    mode: Literal["linked", "installed"]


def probe_link_mode(cache_dir: str) -> LinkMode:
    """Reflink a probe file inside ``cache_dir``: "reflink" when that works, else "hardlink" (logged as a warning)."""
    os.makedirs(cache_dir, exist_ok=True)
    probe = os.path.join(cache_dir, f".link-probe-{uuid.uuid4().hex}")
    clone = probe + ".clone"
    with open(probe, "w", encoding="utf-8") as handle:
        handle.write("probe")
    try:
        _reflink_copy(probe, clone)
        return "reflink"
    except (OSError, LabEnvironmentError) as exc:
        logger.warning(_HARDLINK_WARNING, cache_dir, exc)
        return "hardlink"
    finally:
        for path in (probe, clone):
            if os.path.lexists(path):
                os.remove(path)


class Environments:
    """Makes each tree's environment from the stores under ``cache_dir`` (``<workspace root>/.cache``)."""

    def __init__(self, cache_dir: str, *, mode: LinkMode, runner: Optional[CommandRunner] = None):
        self.cache_dir: str = os.path.abspath(cache_dir)
        self.mode: LinkMode = mode
        # Interpreters must live outside the workspace root: every sandbox hides it (cache_dir is <root>/.cache).
        self._workspace_root = os.path.dirname(self.cache_dir)
        self._uv_cache = os.path.join(self.cache_dir, "uv")
        self._node_cache = os.path.join(self.cache_dir, "node")
        self._runner: CommandRunner = runner or _unconfined_runner
        os.makedirs(self._uv_cache, exist_ok=True)
        os.makedirs(self._node_cache, exist_ok=True)

    def ensure(self, repo: RepoSpec, *, runner: Optional[CommandRunner] = None) -> EnvironmentStamp:
        """Make ``repo``'s environment in its tree, unless its stamp matches the current inputs.

        Every install command goes through ``runner``, else the constructor's runner, else an
        unconfined subprocess. Raises :class:`LabEnvironmentError` when a command fails; no stamp
        is written then.
        """
        if repo.key not in _INPUTS:
            raise LabEnvironmentError(f"no environment is defined for repository {repo.key!r}")
        inputs = _inputs_sha256(repo)
        current = self.stamp(repo)
        if current is not None and current.inputs_sha256 == inputs and _present(repo):
            return current
        run = runner or self._runner
        started = time.monotonic()
        if repo.key == REPO_FRONTEND:
            mode = self._ensure_node(repo, run)
        elif repo.key == REPO_BESSER:
            self._ensure_besser(repo, run)
            mode = "installed"
        else:
            self._ensure_agent(repo, run)
            mode = "installed"
        made = EnvironmentStamp(repo=repo.key, inputs_sha256=inputs, created_at=time.time(),
                                seconds=round(time.monotonic() - started, 3), mode=mode)
        _write_json(os.path.join(repo.path, STAMP_FILENAME), asdict(made))
        logger.info("Environment of %s in %s made in %.1f s (%s)", repo.key, repo.path, made.seconds, mode)
        return made

    def stamp(self, repo: RepoSpec) -> Optional[EnvironmentStamp]:
        """The stamp in ``repo``'s tree, or None when there is none or it cannot be read (logged)."""
        path = os.path.join(repo.path, STAMP_FILENAME)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return EnvironmentStamp(**json.load(handle))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Ignoring the unreadable environment stamp %s: %s", path, exc)
            return None

    def snapshot_node(self, base_editor: RepoSpec) -> str:
        """Snapshot the base editor's ``node_modules`` (after its ``npm ci``) by its lock file; returns the folder.

        A snapshot is taken once per lock-file hash and never changes afterwards.
        """
        lock = os.path.join(base_editor.path, "package-lock.json")
        if not os.path.isfile(lock):
            raise LabEnvironmentError(f"{base_editor.path} has no package-lock.json")
        snapshot = os.path.join(self._node_cache, _sha256_file(lock))
        if os.path.isdir(snapshot):
            return snapshot
        folders = _node_module_folders(base_editor.path)
        if "node_modules" not in folders:
            raise LabEnvironmentError(f"{base_editor.path} has no node_modules: run npm ci there first")
        partial = f"{snapshot}.partial-{uuid.uuid4().hex}"
        for folder in folders:
            target = os.path.join(partial, *folder.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            self._clone(os.path.join(base_editor.path, *folder.split("/")), target)
        if self.mode == "hardlink":
            _make_files_read_only(partial)
        try:
            os.rename(partial, snapshot)
        except OSError:
            if not os.path.isdir(snapshot):  # not another thread's snapshot of the same lock
                raise
            shutil.rmtree(partial, ignore_errors=True)
        logger.info("node_modules snapshot %s taken from %s (%s)", snapshot, base_editor.path, self.mode)
        return snapshot

    # -- node_modules ------------------------------------------------------------------------------

    def _ensure_node(self, repo: RepoSpec, run: CommandRunner) -> Literal["linked", "installed"]:
        lock = os.path.join(repo.path, "package-lock.json")
        snapshot = os.path.join(self._node_cache, _sha256_file(lock)) if os.path.isfile(lock) else ""
        if snapshot and os.path.isdir(snapshot):
            self._link_node(snapshot, repo.path)
            return "linked"
        logger.warning("No node_modules snapshot matches the package-lock.json of %s: running npm ci in the tree",
                       repo.path)
        _run(run, _NPM_CI, repo.path)
        return "installed"

    def _link_node(self, snapshot: str, tree: str) -> None:
        """Clone every node_modules folder of the snapshot into ``tree``, replacing what is there."""
        for folder in _node_module_folders(snapshot):
            target = os.path.join(tree, *folder.split("/"))
            parent = os.path.dirname(target)
            # The tree is model-writable: a parent that is missing or leads out of it gets nothing.
            if not os.path.isdir(parent) or not _within(parent, tree):
                logger.warning("Not linking %s into %s: its folder is missing or leaves the tree", folder, tree)
                continue
            _remove(target)
            self._clone(os.path.join(snapshot, *folder.split("/")), target)

    def _clone(self, source: str, target: str) -> None:
        if self.mode == "reflink":
            _reflink_copy(source, target)
        else:  # cp -al: files hard-linked, symlinks (npm's relative workspace links) kept as links
            shutil.copytree(source, target, symlinks=True, copy_function=os.link)

    # -- Python ------------------------------------------------------------------------------------

    def _ensure_besser(self, repo: RepoSpec, run: CommandRunner) -> None:
        tree = repo.path
        python = self._venv(tree, _BESSER_PYTHON, run)
        _run(run, ["uv", "pip", "install", "--quiet", "--python", python, *self._uv_options(),
                   "-e", f"{tree}[dev]",
                   "-r", os.path.join(tree, "besser", "utilities", "web_modeling_editor", "backend",
                                      "requirements.txt"),
                   "-r", os.path.join(tree, "docs", "requirements.txt"), _IMPORT_LINTER], tree)

    def _ensure_agent(self, repo: RepoSpec, run: CommandRunner) -> None:
        tree = repo.path
        python = self._venv(tree, _AGENT_PYTHON, run)
        _run(run, ["uv", "pip", "install", "--quiet", "--python", python, *self._uv_options(),
                   "-r", os.path.join(tree, "requirements-dev.txt")], tree)
        _run(run, ["uv", "pip", "install", "--quiet", "--python", python, *self._uv_options(),
                   "--no-deps", "-e", tree], tree)
        patch = os.path.join(tree, *_AGENT_PATCH.split("/"))
        if not os.path.isfile(patch):
            logger.warning("%s has no %s: BAF's websocket platform is left unpatched", tree, _AGENT_PATCH)
            return
        lines = _run(run, [python, "-c", _FIND_BAF_PLATFORM], tree).strip().splitlines()
        target = lines[-1].strip() if lines else ""
        if not target or not _within(target, os.path.join(tree, "venv")):
            raise LabEnvironmentError(f"BAF's websocket platform resolves outside {tree}'s venv: {target!r}")
        # The installed file may be a hard link into uv's cache: unlink it so the copy cannot write through.
        _run(run, ["rm", "-f", target], tree)
        _run(run, ["cp", patch, target], tree)

    def _venv(self, tree: str, version: str, run: CommandRunner) -> str:
        """``<tree>/venv`` (made when missing) and its interpreter, checked to come from outside the workspace."""
        python = _venv_python(tree)
        if not os.path.isfile(python):
            _run(run, ["uv", "venv", "--quiet", "--seed", "--python", version, "--no-python-downloads",
                       *self._uv_options(), os.path.join(tree, "venv")], tree)
        home = _venv_home(tree)
        if _within(home, self._workspace_root):
            raise LabEnvironmentError(f"the venv of {tree} uses an interpreter under the workspace root ({home}), "
                                      "which every sandbox hides: install Python outside it")
        return python

    def _uv_options(self) -> list[str]:
        return ["--cache-dir", self._uv_cache, "--link-mode", _UV_LINK_MODE[self.mode]]


def _run(run: CommandRunner, argv: Sequence[str], cwd: str) -> str:
    code, output = run(list(argv), cwd)
    if code != 0:
        tail, _ = redact_text(output[-_OUTPUT_TAIL_CHARS:])
        raise LabEnvironmentError(f"{' '.join(argv[:3])} failed in {cwd} (exit {code}): {tail.strip()}")
    return output


def _unconfined_runner(argv: Sequence[str], cwd: str) -> tuple[int, str]:
    """Run ``argv`` with the scrubbed environment; for the base trees, whose code is reviewed upstream code."""
    env = safe_subprocess_env()
    executable = shutil.which(argv[0], path=env.get("PATH")) or argv[0]
    try:
        result = run_bounded([executable, *argv[1:]], timeout=_INSTALL_TIMEOUT_SECONDS, cwd=cwd, env=env)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {_INSTALL_TIMEOUT_SECONDS:g} s"
    except OSError as exc:
        return 127, f"could not start {argv[0]}: {exc}"
    return result.returncode, result.stdout + result.stderr


def _reflink_copy(source: str, target: str) -> None:
    """``cp -a --reflink=always``: a copy-on-write clone, or an error where the filesystem has none."""
    env = safe_subprocess_env()
    try:
        result = run_bounded(["cp", "-a", "--reflink=always", source, target],
                             timeout=_INSTALL_TIMEOUT_SECONDS, env=env)
    except subprocess.TimeoutExpired:
        raise LabEnvironmentError(f"cp --reflink timed out on {source}") from None
    if result.returncode != 0:
        raise LabEnvironmentError((result.stderr or result.stdout).strip() or f"cp exit {result.returncode}")


def _inputs_sha256(repo: RepoSpec) -> str:
    """Hash of the files ``repo``'s environment is made from; a missing file counts as missing."""
    digest = hashlib.sha256()
    for name in _INPUTS[repo.key]:
        path = os.path.join(repo.path, *name.split("/"))
        files = [(name, path)]
        if os.path.isdir(path) and not os.path.islink(path):
            files = [(f"{name}/{os.path.relpath(os.path.join(folder, file), path).replace(os.sep, '/')}",
                      os.path.join(folder, file))
                     for folder, _, names in os.walk(path) for file in names]
        for relative, file in sorted(files):
            if os.path.islink(file):  # a link is hashed as a link: nothing outside the tree is read
                content = "link:" + os.readlink(file)
            elif os.path.isfile(file):
                content = _sha256_file(file)
            else:
                content = "missing"
            digest.update(f"{relative}\0{content}\n".encode("utf-8"))
    return digest.hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _present(repo: RepoSpec) -> bool:
    """Whether the stamped environment is still on disk."""
    if repo.key == REPO_FRONTEND:
        return os.path.isdir(os.path.join(repo.path, "node_modules"))
    return os.path.isfile(_venv_python(repo.path))


def _node_module_folders(root: str) -> list[str]:
    """``node_modules`` and ``packages/<name>/node_modules`` under ``root`` that are real folders, '/'-separated."""
    folders = []
    if _real_dir(os.path.join(root, "node_modules")):
        folders.append("node_modules")
    packages = os.path.join(root, "packages")
    if _real_dir(packages):
        for name in sorted(os.listdir(packages)):
            if _real_dir(os.path.join(packages, name)) and _real_dir(os.path.join(packages, name, "node_modules")):
                folders.append(f"packages/{name}/node_modules")
    return folders


def _real_dir(path: str) -> bool:
    return os.path.isdir(path) and not os.path.islink(path)


def _make_files_read_only(folder: str) -> None:
    """Drop every write bit of the files under ``folder``; folders stay writable (Vite writes node_modules/.vite)."""
    for current, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(current, name)
            mode = os.lstat(path).st_mode
            if not stat.S_ISLNK(mode):
                os.chmod(path, stat.S_IMODE(mode) & ~0o222)


def _remove(path: str) -> None:
    """Remove a file, a link (never what it points at) or a folder."""
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _within(path: str, folder: str) -> bool:
    path, folder = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(folder))
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def _venv_python(tree: str) -> str:
    if os.name == "nt":
        return os.path.join(tree, "venv", "Scripts", "python.exe")
    return os.path.join(tree, "venv", "bin", "python")


def _venv_home(tree: str) -> str:
    """The ``home`` of ``<tree>/venv/pyvenv.cfg``: the folder of the interpreter the venv was made from."""
    config = os.path.join(tree, "venv", "pyvenv.cfg")
    try:
        with open(config, encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition("=")
                if key.strip() == "home" and value.strip():
                    return value.strip()
    except OSError as exc:
        raise LabEnvironmentError(f"the venv of {tree} has no readable pyvenv.cfg: {exc}") from None
    raise LabEnvironmentError(f"{config} names no interpreter home")


def _write_json(path: str, data: dict) -> None:
    partial = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    os.replace(partial, path)
