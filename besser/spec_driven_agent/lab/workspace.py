"""The lab's repositories: base clones, one git worktree per repository per change, and each change's state.

Layout (``root`` is ``BESSER_LAB_WORKSPACE_ROOT``)::

    <root>/base/besser/            base clone on its base branch, never edited; the editor's base clone
                                   at FRONTEND_SUBMODULE_PATH inside it
    <root>/base/modeling-agent/
    <root>/changes/<id>/besser/    the change's worktree on lab/<owner>/<slug>, the editor's worktree
                                   at FRONTEND_SUBMODULE_PATH inside it
    <root>/changes/<id>/modeling-agent/
    <git_root>/besser/  <git_root>/besser/modules/<FRONTEND_SUBMODULE_PATH>/  <git_root>/modeling-agent/
                                   the git directories, each worktree's own under its ``worktrees/``
    <state_dir>/changes/<id>/      state.json, notes.md (and the service's files)

Sandboxed code can write a change's trees but not the git root. So every git command runs
with explicit ``GIT_DIR`` and ``GIT_WORK_TREE``, and a worktree's git directory is found
from the git root (the ``worktrees/*/gitdir`` entry naming the tree), never from the
``.git`` file in the tree, which that code can rewrite. A change is a draft (state only;
it works in ``<root>/base``) until :meth:`LabWorkspace.open_change` gives it a branch and a
worktree in every repository. Writes to the shared refs of a git directory (worktrees,
branches, fetch, base update) take that repository's lock, in ``REPO_ORDER``.

Git always runs through an argv list with the scrubbed child environment. Paths are read
and written in git's C-quoted form (``core.quotepath``), so git's output is plain ASCII
whatever the locale's encoding is.
"""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
from dataclasses import asdict, replace
from typing import Collection, Iterator, Mapping, Optional, Sequence

from besser.spec_driven_agent.execution.process import run_bounded
from besser.spec_driven_agent.execution.sandbox import sandbox_home
from besser.spec_driven_agent.lab.contracts import (
    FRONTEND_SUBMODULE_PATH,
    LAB_BASE_DIRNAME,
    LAB_BRANCH_PREFIX,
    LAB_CHANGES_DIRNAME,
    LAB_STATE_VERSION,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    REPO_ORDER,
    ChangeKind,
    FileChange,
    FileDiff,
    LabChange,
    RepoBase,
    RepoChange,
    RepoSpec,
)
from besser.utilities.secret_redaction import redact_text
from besser.utilities.subprocess_env import safe_subprocess_env

logger = logging.getLogger(__name__)

# Applied to every git call: no line-ending conversion, and non-ASCII paths quoted. Git runs none of its hooks and
# no fsmonitor command: model-written code reaches what the trees hold, and anything git ran would run unconfined.
_GIT_CONFIG = ("core.autocrlf=false", "core.quotepath=true", "core.hooksPath=/dev/null", "core.fsmonitor=false")

# Base clones fetch public repositories without credentials: no helper may add the host's.
_ANONYMOUS = ("credential.helper=",)

# A fetch after a long pause can bring weeks of BESSER's history.
_FETCH_TIMEOUT_SECONDS = 600

# Run artefacts of the agent's tools and the shell sandbox's home: never part of a change.
_HARNESS_PREFIX = ".besser_"
_SANDBOX_HOME_SUFFIX = ".sandbox-home"

# Long enough for a readable title, short enough for a branch name in a URL.
_MAX_SLUG_LENGTH = 48

# Change ids reach the file system from HTTP requests, so only this shape is accepted.
_CHANGE_ID_RE = re.compile(r"^[0-9a-f]{12}$")

# BESSER_LAB_REPO_* values; the settings validate the same shape.
_UPSTREAM_RE = re.compile(r"^([A-Za-z0-9-]+)/([A-Za-z0-9._-]+)@([A-Za-z0-9._/-]+)$")

# Where refresh_bases puts each upstream base branch; base_status counts against it.
_UPSTREAM_REFS = "refs/remotes/upstream/"

# Where a pushed lab branch is recorded: open_change counts it as taken after the change is dropped.
_PUSHED_REFS = "refs/remotes/github/"

# What makes the layout this module expects; every readiness refusal names it.
_SETUP_SCRIPT = "scripts/lab/setup_workspace.sh"

# Each repository's tree under <root>/base (and under a change's root), and its git directory under the git root.
_SUBMODULE_PARTS = tuple(FRONTEND_SUBMODULE_PATH.split("/"))
_TREE_PARTS = {REPO_FRONTEND: ("besser", *_SUBMODULE_PARTS), REPO_BESSER: ("besser",),
               REPO_MODELING_AGENT: ("modeling-agent",)}
_GIT_DIR_PARTS = {REPO_FRONTEND: ("besser", "modules", *_SUBMODULE_PARTS), REPO_BESSER: ("besser",),
                  REPO_MODELING_AGENT: ("modeling-agent",)}

# Phase 1 statuses (one change checked out, the others parked) are both "open" now.
_PHASE_1_STATUSES = {"active": "open", "parked": "open"}

# First letter of a ``git diff --name-status`` code -> FileChange.status
_STATUS_BY_CODE = {"A": "added", "C": "added", "D": "deleted", "M": "modified", "R": "renamed", "T": "modified"}

# Shared by the numstat and the patch of file_diffs, so both list the same files in the same order.
_DIFF_ARGS = ("-M", "--no-color", "--no-ext-diff", "--ignore-submodules=all")

# The most diff output file_diffs reads in one call: a model-written file of gigabytes would otherwise fill
# the memory of the lab's one worker. Far more than anyone reviews line by line.
_MAX_DIFF_BYTES = 64 * 1024 * 1024

# Where each file's patch starts; content lines always start with " ", "+" or "-".
_PATCH_START_RE = re.compile(r"^diff --git ", re.MULTILINE)

# Header line of a file's patch -> FileDiff.status (no line: "modified")
_STATUS_BY_HEADER = (("new file mode", "added"), ("deleted file mode", "deleted"), ("rename from", "renamed"))

# What ``git diff --check`` prints for a conflict marker left in an added line (it also prints whitespace errors).
_CONFLICT_MARKER_RE = re.compile(r"^(.*):\d+: leftover conflict marker$")

_CLOSE_OUTCOMES = ("merged", "closed")

# The refusal of a closed change's diff, shown to the user as it is.
_CLOSED_DIFF = "its branches were deleted when it closed; the diff is on the pull request"


class LabWorkspaceError(Exception):
    """A git operation or a change's state does not allow what was asked."""


def parse_upstream(value: str) -> tuple[str, str, str]:
    """``owner/name@branch`` -> ``(owner, name, branch)``; raises ``ValueError`` on any other shape."""
    match = _UPSTREAM_RE.match(value)
    if match is None:
        raise ValueError(f"{value!r} is not owner/name@branch")
    owner, name, branch = match.groups()
    return owner, name, branch


def repo_specs_from_settings(root: str, git_root: str, upstreams: Mapping[str, str]) -> tuple[RepoSpec, ...]:
    """The base clones under ``<root>/base``, in pull-request order.

    ``upstreams`` maps each repository key to its ``owner/name@branch`` (the ``BESSER_LAB_REPO_*``
    settings): where the base is fetched from and pull requests go.
    """
    base = os.path.join(os.path.abspath(root), LAB_BASE_DIRNAME)
    git_root = os.path.abspath(git_root)
    specs = []
    for key in REPO_ORDER:
        owner, name, branch = parse_upstream(upstreams[key])
        specs.append(RepoSpec(key, os.path.join(base, *_TREE_PARTS[key]), branch, owner, name,
                              os.path.join(git_root, *_GIT_DIR_PARTS[key])))
    return tuple(specs)


def upstream_url(spec: RepoSpec) -> str:
    """Where ``refresh_bases`` fetches ``spec``'s base branch from (anonymously)."""
    return f"https://github.com/{spec.github_owner}/{spec.github_name}.git"


def _slug(text: str, fallback: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if len(slug) > _MAX_SLUG_LENGTH:
        # Cut after the last whole word that fits; a single word longer than the cap is cut at it.
        head = slug[:_MAX_SLUG_LENGTH + 1].rpartition("-")[0]
        slug = head or slug[:_MAX_SLUG_LENGTH]
    return slug.rstrip("-") or fallback


def _is_harness_path(path: str) -> bool:
    return any(part.startswith(_HARNESS_PREFIX) or part.endswith(_SANDBOX_HOME_SUFFIX)
               for part in path.split("/"))


def _unquote(path: str) -> str:
    """A path as git prints it: C-quoted (octal escapes of UTF-8 bytes) when not plain ASCII."""
    if path.startswith('"'):
        return ast.literal_eval("b" + path).decode("utf-8", errors="replace")
    return path


def _quote(path: str) -> str:
    """A path C-quoted for ``--pathspec-from-file``, ASCII whatever the path holds."""
    out = []
    for byte in path.encode("utf-8"):
        char = chr(byte)
        if char in '"\\':
            out.append("\\" + char)
        elif 0x20 <= byte < 0x7F:
            out.append(char)
        else:
            out.append(f"\\{byte:03o}")
    return '"' + "".join(out) + '"'


def _pathspec_input(paths: Sequence[str]) -> str:
    return "".join(_quote(path) + "\n" for path in paths)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _within(path: str, folder: str) -> bool:
    """Whether ``path`` is ``folder`` or inside it."""
    path, folder = _norm(path), _norm(folder)
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def _identity(login: str) -> tuple[str, str]:
    """Git config making ``login`` the author and committer."""
    return f"user.name={login}", f"user.email={login}@users.noreply.github.com"


class LabWorkspace:
    """The base clones of the lab's repositories and the changes made in worktrees of them."""

    def __init__(self, root: str, repos: Sequence[RepoSpec], *, state_dir: str, git_root: str):
        self.root: str = os.path.abspath(root)
        #: the changes' state: ``changes/<id>/`` and whatever else the lab keeps between messages
        self.state_dir: str = os.path.abspath(state_dir)
        self.git_root: str = os.path.abspath(git_root)
        #: repository key -> its base clone under ``<root>/base``
        self.base_repos: dict[str, RepoSpec] = {spec.key: spec for spec in repos}
        self._changes_dir = os.path.join(self.state_dir, "changes")
        self._locks = {key: threading.Lock() for key in self.base_repos}

    # -- git ---------------------------------------------------------------------------------------

    def run_git(self, repo: str, *args: str, change: Optional[LabChange] = None,
                env_extra: Optional[Mapping[str, str]] = None, timeout: float = 120) -> str:
        """Run ``git <args>`` on ``repo``'s base clone, or on the change's worktree, and return its output.

        ``env_extra`` is added to the scrubbed environment; when it sets ``GIT_DIR`` that
        repository is used instead, with no work tree. Its values are also redacted from an
        error message. Raises :class:`LabWorkspaceError` on failure or timeout.
        """
        return self._git(self._spec(repo, change), *args, env_extra=env_extra, timeout=timeout)

    def _run(self, spec: RepoSpec, args: Sequence[str], *, input: Optional[str] = None, config: Sequence[str] = (),
             env_extra: Optional[Mapping[str, str]] = None, timeout: float = 120) -> subprocess.CompletedProcess:
        env = safe_subprocess_env()
        env["GIT_TERMINAL_PROMPT"] = "0"
        if "GIT_DIR" not in (env_extra or {}):
            env["GIT_DIR"] = spec.git_dir
            env["GIT_WORK_TREE"] = spec.path
        env.update(env_extra or {})
        argv = ["git"]
        for item in (*_GIT_CONFIG, *config):
            argv += ["-c", item]
        argv += list(args)
        try:
            return run_bounded(argv, timeout=timeout, cwd=spec.path, env=env, input=input)
        except subprocess.TimeoutExpired:
            raise LabWorkspaceError(f"git {args[0]} timed out after {timeout:g} s in {spec.key}") from None
        except OSError as exc:
            raise LabWorkspaceError(f"cannot run git in {spec.key} ({spec.path}): {exc}") from None

    def _git(self, spec: RepoSpec, *args: str, input: Optional[str] = None, config: Sequence[str] = (),
             env_extra: Optional[Mapping[str, str]] = None, timeout: float = 120) -> str:
        result = self._run(spec, args, input=input, config=config, env_extra=env_extra, timeout=timeout)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "no output"
            detail, _ = redact_text(detail, secrets=list((env_extra or {}).values()))
            raise LabWorkspaceError(f"git {args[0]} failed in {spec.key} (exit {result.returncode}): {detail}")
        return result.stdout

    def _has_ref(self, spec: RepoSpec, ref: str) -> bool:
        return self._run(spec, ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]).returncode == 0

    def _rev(self, spec: RepoSpec, ref: str) -> str:
        return self._git(spec, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()

    def _merge_base(self, spec: RepoSpec) -> str:
        """Where the change's branch left the upstream base (moved by a base merge)."""
        return self._git(spec, "merge-base", "HEAD", _UPSTREAM_REFS + spec.base_branch).strip()

    def _is_ancestor(self, spec: RepoSpec, ancestor: str, descendant: str) -> bool:
        return self._run(spec, ["merge-base", "--is-ancestor", ancestor, descendant]).returncode == 0

    def _branch_names(self) -> set[str]:
        """Every branch name of every repository: local, upstream's and those the lab pushed."""
        names = set()
        for spec in self.base_repos.values():
            for ref in self._git(spec, "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes").split():
                if ref.startswith("refs/heads/"):
                    names.add(ref[len("refs/heads/"):])
                elif ref.count("/") >= 3:
                    names.add(ref.split("/", 3)[3])
        return names

    def _untracked(self, spec: RepoSpec) -> list[str]:
        paths = (_unquote(line) for line in
                 self._git(spec, "ls-files", "--others", "--exclude-standard").splitlines())
        return [path for path in paths if not _is_harness_path(path)]

    def _changed_files(self, spec: RepoSpec, since: str, *, renames: bool) -> list[FileChange]:
        """Working tree (staged or not, untracked included) against ``since``, run artefacts left out.

        ``--ignore-submodules=all`` keeps the editor's pointer out of BESSER's list:
        the editor reports its own files.
        """
        out = self._git(spec, "diff", "--name-status", "--ignore-submodules=all",
                        "-M" if renames else "--no-renames", since, "--")
        changes = []
        for line in out.splitlines():
            fields = line.split("\t")
            path = _unquote(fields[-1])
            if not _is_harness_path(path):
                changes.append(FileChange(path, _STATUS_BY_CODE.get(fields[0][:1], "modified")))
        changes += [FileChange(path, "added") for path in self._untracked(spec)]
        return changes

    def _unmerged(self, spec: RepoSpec) -> list[str]:
        """The paths a merge in progress left conflicted, run artefacts left out."""
        paths = (_unquote(line) for line in self._git(spec, "diff", "--name-only", "--diff-filter=U").splitlines())
        return [path for path in paths if not _is_harness_path(path)]

    def _conflict_markers(self, spec: RepoSpec) -> list[str]:
        """The files whose lines added against HEAD hold a conflict marker (``git diff --check``)."""
        out = self._run(spec, ["diff", "--check", "--ignore-submodules=all", "HEAD", "--"]).stdout
        found = (_CONFLICT_MARKER_RE.match(line) for line in out.splitlines())
        return sorted({_unquote(match.group(1)) for match in found if match})

    @contextlib.contextmanager
    def _diff_env(self, spec: RepoSpec) -> Iterator[tuple[dict[str, str], str]]:
        """Environment and a scratch folder for diffing the tree with its untracked files as new files.

        The untracked files enter through intent-to-add entries in a copy of the index, so the
        real index is left as it was.
        """
        with tempfile.TemporaryDirectory(prefix="besser_lab_diff_") as scratch:
            env = {"GIT_LITERAL_PATHSPECS": "1"}
            untracked = self._untracked(spec)
            if untracked:
                env["GIT_INDEX_FILE"] = os.path.join(scratch, "index")
                shutil.copyfile(os.path.join(spec.git_dir, "index"), env["GIT_INDEX_FILE"])
                self._git(spec, "add", "-N", "--pathspec-from-file=-", input=_pathspec_input(untracked),
                          env_extra=env)
            yield env, scratch

    def _diff_output(self, spec: RepoSpec, scratch: str, args: Sequence[str], env: Mapping[str, str]) -> str:
        """``git diff <args>`` read as UTF-8 bytes from a file: paths and patches whatever the locale."""
        out = os.path.join(scratch, "diff.out")
        self._git(spec, "diff", f"--output={out}", *args, env_extra=env)
        size = os.path.getsize(out)
        if size > _MAX_DIFF_BYTES:
            raise LabWorkspaceError(f"{spec.key}: the diff is {size // (1024 * 1024)} MB, more than the "
                                    f"{_MAX_DIFF_BYTES // (1024 * 1024)} MB the lab reads; a generated or vendored "
                                    "file this large does not belong in the change")
        with open(out, "rb") as handle:
            return handle.read().decode("utf-8", errors="replace")

    # -- locks -------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def repo_lock(self, repo: str) -> Iterator[None]:
        """Held while the shared refs of ``repo``'s git directory change; take several in ``REPO_ORDER``."""
        try:
            lock = self._locks[repo]
        except KeyError:
            raise LabWorkspaceError(f"unknown repository {repo!r}") from None
        with lock:
            yield

    @contextlib.contextmanager
    def _all_repo_locks(self) -> Iterator[None]:
        with contextlib.ExitStack() as stack:
            for key in self._keys():
                stack.enter_context(self.repo_lock(key))
            yield

    def _keys(self) -> list[str]:
        """The repositories in ``REPO_ORDER`` (any other key after them)."""
        return sorted(self.base_repos, key=lambda key: REPO_ORDER.index(key) if key in REPO_ORDER else len(REPO_ORDER))

    # -- layout ------------------------------------------------------------------------------------

    def _base_root(self) -> str:
        return os.path.join(self.root, LAB_BASE_DIRNAME)

    def _trees_root(self, change_id: str) -> str:
        _check_id(change_id)
        return os.path.join(self.root, LAB_CHANGES_DIRNAME, change_id)

    def _tree(self, trees_root: str, spec: RepoSpec) -> str:
        """Where ``spec``'s worktree goes under a change's root: the base tree's place under ``<root>/base``."""
        return os.path.join(trees_root, os.path.relpath(spec.path, self._base_root()))

    def _creation_order(self) -> list[RepoSpec]:
        """Base specs with a tree's parent before it (BESSER before the editor inside it), else ``REPO_ORDER``."""
        keys = self._keys()
        return sorted((self.base_repos[key] for key in keys),
                      key=lambda spec: (len(os.path.relpath(spec.path, self._base_root()).split(os.sep)),
                                        keys.index(spec.key)))

    def change_root(self, change: LabChange) -> str:
        """``<root>/changes/<id>``, the folder of the change's trees; ``<root>/base`` for a draft."""
        if change.status == "draft":
            return self._base_root()
        return self._trees_root(change.id)

    def change_repos(self, change: LabChange) -> dict[str, RepoSpec]:
        """The change's trees and their worktree git directories; the base clones for a draft."""
        if change.status == "draft":
            return dict(self.base_repos)
        return {key: self._spec(key, change) for key in self._keys() if key in change.branches}

    def _spec(self, repo: str, change: Optional[LabChange] = None) -> RepoSpec:
        try:
            base = self.base_repos[repo]
        except KeyError:
            raise LabWorkspaceError(f"unknown repository {repo!r}") from None
        if change is None or change.status == "draft":
            return base
        if change.status == "closed":
            raise LabWorkspaceError(f"change {change.id} is closed: its trees and branches were removed")
        if repo not in change.branches:
            raise LabWorkspaceError(f"change {change.id} has no branch in {repo}")
        tree = self._tree(self._trees_root(change.id), base)
        return replace(base, path=tree, git_dir=self._worktree_git_dir(base, tree))

    def _worktree_git_dir(self, base: RepoSpec, tree: str) -> str:
        """The git directory of the worktree at ``tree``, from the git root's ``worktrees/*/gitdir`` entries."""
        admin_root = os.path.join(base.git_dir, "worktrees")
        target = _norm(os.path.join(tree, ".git"))
        names = sorted(os.listdir(admin_root)) if os.path.isdir(admin_root) else []
        for name in names:
            try:
                with open(os.path.join(admin_root, name, "gitdir"), encoding="utf-8") as handle:
                    registered = handle.read().strip()
            except OSError:
                continue
            if registered and _norm(registered) == target:
                return os.path.join(admin_root, name)
        raise LabWorkspaceError(f"{base.key}: no worktree of {base.git_dir} is registered at {tree}")

    # -- readiness ---------------------------------------------------------------------------------

    def ensure_ready(self) -> None:
        """Check that every base clone exists, on its base branch, with its git directory outside the workspace."""
        for key in self._keys():
            spec = self.base_repos[key]
            if not _within(spec.path, self._base_root()):
                raise LabWorkspaceError(f"{key}: the base tree {spec.path} is not under {self._base_root()}")
            if os.path.isdir(os.path.join(spec.path, ".git")) or _within(spec.git_dir, self.root):
                raise LabWorkspaceError(
                    f"{key}: the git directory of {spec.path} is inside the workspace, where model-written code "
                    f"can rewrite it; it belongs under BESSER_LAB_GIT_ROOT ({self.git_root}). "
                    f"Make the workspace with {_SETUP_SCRIPT}")
            if not os.path.isdir(spec.path) or not os.path.isdir(spec.git_dir):
                raise LabWorkspaceError(f"{key}: the base clone {spec.path} (git directory {spec.git_dir}) is "
                                        f"missing; make the workspace with {_SETUP_SCRIPT}")
            head = self._run(spec, ["symbolic-ref", "-q", "HEAD"]).stdout.strip()
            if not self._has_ref(spec, _base_ref(spec)) or head != _base_ref(spec):
                raise LabWorkspaceError(f"{key}: base branch {spec.base_branch} not found or not checked out in "
                                        f"{spec.path}; make the workspace with {_SETUP_SCRIPT}")

    # -- change state ------------------------------------------------------------------------------

    def _state_dir(self, change_id: str) -> str:
        _check_id(change_id)
        return os.path.join(self._changes_dir, change_id)

    def get_change(self, change_id: str) -> LabChange:
        """The change ``change_id``; raises :class:`LabWorkspaceError` when there is none."""
        path = os.path.join(self._state_dir(change_id), "state.json")
        if not os.path.isfile(path):
            raise LabWorkspaceError(f"no change {change_id}")
        # A broken file is a refusal naming it, not a 500.
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            return _change_from_state(data)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise LabWorkspaceError(f"the state of change {change_id} ({path}) cannot be read: {exc}; "
                                    "fix or delete that file") from None

    def changes(self, *, statuses: Collection[str] = ("draft", "open")) -> list[LabChange]:
        """The changes whose status is in ``statuses``, oldest first."""
        if not os.path.isdir(self._changes_dir):
            return []
        ids = [name for name in os.listdir(self._changes_dir)
               if _CHANGE_ID_RE.match(name) and os.path.isfile(os.path.join(self._changes_dir, name, "state.json"))]
        found = (self.get_change(change_id) for change_id in ids)
        return sorted((change for change in found if change.status in statuses), key=lambda change: change.created_at)

    def save_change(self, change: LabChange) -> None:
        """Write the change's state (atomically), stamping ``updated_at``."""
        change.updated_at = time.time()
        directory = self._state_dir(change.id)
        os.makedirs(directory, exist_ok=True)
        data = {"version": LAB_STATE_VERSION, **asdict(change)}
        data["kind"] = change.kind.value
        path = os.path.join(directory, "state.json")
        with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2)
        os.replace(path + ".tmp", path)

    def notes(self, change: LabChange) -> str:
        """The change's notes (what was asked and done), empty before the first note."""
        path = os.path.join(self._state_dir(change.id), "notes.md")
        if not os.path.isfile(path):
            return ""
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def append_note(self, change: LabChange, text: str) -> None:
        """Append a paragraph to the change's notes."""
        directory = self._state_dir(change.id)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "notes.md"), "a", encoding="utf-8", newline="\n") as handle:
            handle.write(text.strip("\n") + "\n\n")

    # -- change lifecycle --------------------------------------------------------------------------

    def create_draft(self, *, owner: str) -> LabChange:
        """A new change with state only: no branch, its plan step reads the base clones."""
        now = time.time()
        change = LabChange(id=uuid.uuid4().hex[:12], slug="", title="", owner=owner, status="draft", created_at=now)
        self.save_change(change)
        return change

    def open_change(self, change: LabChange, *, title: str, kind: ChangeKind) -> LabChange:
        """Give a draft its branch ``lab/<owner>/<slug>`` and a worktree in every repository; returns it open.

        The bases are refreshed first, and each branch starts from its base tip, recorded in
        ``bases``. The slug is unique against every local branch, upstream's and those the lab
        pushed (``record_pushed``). A failure part way removes the worktrees and branches made
        and leaves the draft as it was.
        """
        if change.status != "draft":
            raise LabWorkspaceError(f"change {change.id} is not a draft ({change.status}): it has its branches")
        self.refresh_bases()
        trees_root = self._trees_root(change.id)
        made: list[tuple[RepoSpec, str]] = []
        with self._all_repo_locks():
            taken = self._branch_names()
            prefix = f"{LAB_BRANCH_PREFIX}{_slug(change.owner, 'user')}/"
            slug = base_slug = _slug(title, "change")
            suffix = 2
            while prefix + slug in taken:
                slug = f"{base_slug}-{suffix}"
                suffix += 1
            branch = prefix + slug
            try:
                bases = {}
                for spec in self._creation_order():
                    sha = self._rev(spec, _base_ref(spec))
                    made.append((spec, branch))  # before the add: a failing add can leave its branch behind
                    self._git(spec, "worktree", "add", "-q", "-b", branch, self._tree(trees_root, spec), sha)
                    bases[spec.key] = RepoBase(f"{spec.github_owner}/{spec.github_name}", spec.base_branch, sha)
                opened = replace(change, slug=slug, title=title, kind=kind, status="open",
                                 branches={key: branch for key in self._keys()},
                                 bases={key: bases[key] for key in self._keys()})
                self.save_change(opened)
            except (LabWorkspaceError, OSError):
                try:
                    self._remove_worktrees(trees_root, made)
                except (LabWorkspaceError, OSError):
                    logger.warning("BESSER Lab: change %s could not be opened and what it made could not all be "
                                   "removed", change.id, exc_info=True)
                raise
        return opened

    def repo_changes(self, change: LabChange) -> list[RepoChange]:
        """Committed and uncommitted changes of every repository against its merge-base with the base branch.

        Only repositories with changes are listed; a draft has none.
        """
        result = []
        for key in self._keys():
            if key not in change.branches:
                continue
            spec = self._spec(key, change)
            files = sorted(self._changed_files(spec, self._merge_base(spec), renames=True), key=lambda f: f.path)
            if files:
                result.append(RepoChange(repo=key, branch=change.branches[key], base=spec.base_branch,
                                         files=tuple(files)))
        return result

    def commit_all(self, change: LabChange, message: str, *, login: str) -> dict[str, str]:
        """Commit the uncommitted work of every repository of the change, authored as ``login``.

        Returns repository -> new commit for the repositories that had work. Only the changed
        paths are staged and committed: never run artefacts, never a submodule pointer.

        In a repository where :meth:`begin_base_merge` left a merge in progress, the changed and
        conflicted paths are staged and the merge is concluded with its own message (git allows
        no partial commit during a merge); ``bases[repo].sha`` then moves to the merged upstream
        commit and the change is saved. Refused while a file still holds a conflict marker.
        """
        literal = {"GIT_LITERAL_PATHSPECS": "1"}
        work = {}
        for key in self._keys():
            if key not in change.branches:
                continue
            spec = self._spec(key, change)
            files = self._changed_files(spec, "HEAD", renames=False)
            # git lists an untracked repository (git init, a scaffolder) as "dir/" and cannot add its files.
            nested = [f.path for f in files if f.path.endswith("/")]
            if nested:
                raise LabWorkspaceError(f"{key}: {', '.join(nested)} is a git repository of its own (made by git init "
                                        "or a scaffolder); delete its .git folder to keep its files in the change")
            merging = self._has_ref(spec, "MERGE_HEAD")
            markers = self._conflict_markers(spec) if merging else []
            if markers:
                raise LabWorkspaceError(f"{key}: the merge from {spec.base_branch} still has conflict markers in "
                                        f"{', '.join(markers)}; resolve them before committing")
            work[key] = (spec, files, merging)
        shas = {}
        for key, (spec, files, merging) in work.items():
            if merging:
                merged_upstream = self._rev(spec, "MERGE_HEAD")
                paths = sorted({f.path for f in files} | set(self._unmerged(spec)))
                if paths:
                    self._git(spec, "add", "-A", "--pathspec-from-file=-", input=_pathspec_input(paths),
                              env_extra=literal)
                self._git(spec, "commit", "-q", "--no-edit", config=_identity(login))
                change.bases[key] = replace(change.bases[key], sha=merged_upstream)
                self.save_change(change)
            elif files:
                pathspecs = _pathspec_input(sorted({f.path for f in files}))
                self._git(spec, "add", "-A", "--pathspec-from-file=-", input=pathspecs, env_extra=literal)
                self._git(spec, "commit", "-q", "-m", message, "--pathspec-from-file=-",
                          input=pathspecs, config=_identity(login), env_extra=literal)
            else:
                continue
            shas[key] = self._rev(spec, "HEAD")
        return shas

    def begin_base_merge(self, change: LabChange, repo: str) -> list[str]:
        """Merge the upstream base, as the caller's last :meth:`refresh_bases` fetched it, into the change's branch.

        ``git merge --no-ff --no-edit`` in the change's tree, as the change's owner. Returns ``[]``
        when it merged cleanly; ``bases[repo].sha`` then moves to the merged upstream commit and
        the change is saved. Otherwise returns the conflicted paths, with the merge left in
        progress: :meth:`commit_all` concludes it once the markers are gone. Refused for a change
        that is not open, with uncommitted work, or with a merge already in progress.
        """
        _require_open(change)
        spec = self._spec(repo, change)
        if self._has_ref(spec, "MERGE_HEAD"):
            raise LabWorkspaceError(f"{repo}: a merge from {spec.base_branch} is already in progress; resolve its "
                                    "conflicts and commit it first")
        if self._changed_files(spec, "HEAD", renames=False):
            raise LabWorkspaceError(f"{repo}: the change has uncommitted work; commit it before merging "
                                    f"{spec.base_branch} in")
        upstream = self._rev(spec, _UPSTREAM_REFS + spec.base_branch)
        merged = self._run(spec, ["merge", "-q", "--no-ff", "--no-edit", "-m",
                                  f"Merge {spec.base_branch} into {change.branches[repo]}", upstream],
                           config=_identity(change.owner))
        if merged.returncode == 0:
            change.bases[repo] = replace(change.bases[repo], sha=upstream)
            self.save_change(change)
            return []
        in_progress = self._has_ref(spec, "MERGE_HEAD")
        conflicts = self._unmerged(spec) if in_progress else []
        if conflicts:
            return conflicts
        if in_progress:
            self._run(spec, ["merge", "--abort"])
        detail = redact_text((merged.stderr or merged.stdout).strip() or "no output")[0]
        raise LabWorkspaceError(f"{repo}: merging {spec.base_branch} into {change.branches[repo]} failed: {detail}")

    def merge_in_progress(self, change: LabChange, repo: str) -> bool:
        """Whether a merge from the base waits in the change's ``repo`` tree for its conflicts to be resolved."""
        if change.status != "open":
            return False
        return self._has_ref(self._spec(repo, change), "MERGE_HEAD")

    def file_diffs(self, change: LabChange, repo: str, *, path: Optional[str] = None) -> list[FileDiff]:
        """One entry per file ``repo`` changes, the working tree and untracked files against the merge-base.

        The merge-base is the one of the change's branch with the upstream base. Entries come in
        git's order; a binary file has no counts and an empty patch; run artefacts are left out.
        ``path`` keeps the one file at that path (with its rename). Patches are not redacted
        here. A draft has no diff; a closed change's is refused (its trees are gone).
        """
        if change.status == "closed":
            raise LabWorkspaceError(_CLOSED_DIFF)
        if change.status == "draft":
            return []
        spec = self._spec(repo, change)
        since = self._merge_base(spec)
        with self._diff_env(spec) as (env, scratch):
            entries = _parse_numstat(self._diff_output(spec, scratch, ["--numstat", "-z", *_DIFF_ARGS, since, "--"],
                                                       env))
            pathspecs: list[str] = []
            if path is not None:
                entries = [entry for entry in entries if entry[2] == path]
                if not entries:
                    return []
                pathspecs = [p for p in (entries[0][3], entries[0][2]) if p is not None]
            text = self._diff_output(spec, scratch, [*_DIFF_ARGS, since, "--", *pathspecs], env)
        starts = [match.start() for match in _PATCH_START_RE.finditer(text)]
        patches = [text[start:end] for start, end in zip(starts, [*starts[1:], len(text)])]
        if len(patches) != len(entries):
            raise LabWorkspaceError(f"{repo}: git listed {len(entries)} changed files but printed {len(patches)} "
                                    "patches")
        diffs = []
        for (additions, deletions, new_path, old_path), patch in zip(entries, patches):
            if _is_harness_path(new_path) or (old_path is not None and _is_harness_path(old_path)):
                continue
            binary = additions is None
            diffs.append(FileDiff(path=new_path, old_path=old_path, status=_patch_status(patch), additions=additions,
                                  deletions=deletions, binary=binary, patch="" if binary else patch))
        return diffs

    def unpushed_repos(self, change: LabChange) -> list[str]:
        """The repositories of an open change holding work GitHub does not have.

        That is uncommitted work (a merge in progress included), or commits on the branch that
        neither the upstream base nor the last push (``record_pushed``) holds.
        """
        if change.status != "open":
            return []
        found = []
        for key in self._keys():
            if key not in change.branches:
                continue
            spec = self._spec(key, change)
            pushed = _PUSHED_REFS + change.branches[key]
            known = [_UPSTREAM_REFS + spec.base_branch, *([pushed] if self._has_ref(spec, pushed) else [])]
            ahead = int(self._git(spec, "rev-list", "--count", "HEAD", "--not", *known).strip())
            if ahead or self._has_ref(spec, "MERGE_HEAD") or self._changed_files(spec, "HEAD", renames=False):
                found.append(key)
        return found

    def close(self, change: LabChange, *, outcome: str) -> None:
        """Close an open change whose work is all on GitHub: its trees and branches go, its state stays.

        ``outcome`` is ``"merged"`` or ``"closed"`` (closed unmerged). Refused while
        :meth:`unpushed_repos` lists a repository. The state folder (notes, plan, runs, thread)
        is kept; ``status``, ``closed_at`` and ``outcome`` are set and saved.
        """
        if outcome not in _CLOSE_OUTCOMES:
            raise LabWorkspaceError(f"unknown outcome {outcome!r}: a change closes as merged or closed")
        _require_open(change)
        unpushed = self.unpushed_repos(change)
        if unpushed:
            raise LabWorkspaceError(f"change {change.id} has work that is not on GitHub in {', '.join(unpushed)}; "
                                    "push it to the pull requests or drop the change")
        self.remove_trees(change)
        change.status = "closed"
        change.closed_at = time.time()
        change.outcome = outcome
        self.save_change(change)

    def merge_github_work(self, change: LabChange, repo: str, source: str, ref: str) -> None:
        """Merge into the change's branch the commits added to it on GitHub (a suggestion, "Update branch").

        ``source`` is a git directory holding GitHub's branch at ``ref``. Nothing happens when the
        change's branch already contains it. Refused when that branch shares no commit with the
        change (an earlier change of the same name left it) and when the merge conflicts.
        Once taken, GitHub's head is recorded as pushed (``record_pushed``).
        """
        spec = self._spec(repo, change)
        branch = change.branches[repo]
        self._git(spec, "fetch", "-q", "--no-tags", source, ref)
        theirs = self._rev(spec, "FETCH_HEAD")
        common = self._git(spec, "merge-base", "HEAD", theirs).strip()
        if common == theirs:
            self.record_pushed(repo, branch, theirs)
            return
        if self._is_ancestor(spec, common, _base_ref(spec)):
            raise LabWorkspaceError(f"{repo}: the branch {branch} on GitHub holds commits that are not part of this "
                                    "change (an earlier change of the same name left it); delete it on GitHub")
        merged = self._run(spec, ["merge", "-q", "--no-edit", theirs], config=_identity(change.owner))
        if merged.returncode != 0:
            conflicts = ", ".join(self._git(spec, "diff", "--name-only", "--diff-filter=U").split()[:5])
            self._run(spec, ["merge", "--abort"])
            raise LabWorkspaceError(f"{repo}: the branch {branch} was changed on GitHub and merging those commits "
                                    f"failed ({conflicts or (merged.stderr or merged.stdout).strip()}); "
                                    "undo them on GitHub or resolve the conflict by hand")
        self.record_pushed(repo, branch, theirs)

    def record_pushed(self, repo: str, branch: str, sha: str) -> None:
        """Remember that ``branch`` exists on GitHub, so no later change reuses its name."""
        with self.repo_lock(repo):
            self._git(self._spec(repo), "update-ref", _PUSHED_REFS + branch, sha)

    def touched_paths(self, change: LabChange, repo: str) -> set[str]:
        """Every path ``repo`` changes against its merge-base, a renamed file under its old and new path."""
        spec = self._spec(repo, change)
        return {f.path for f in self._changed_files(spec, self._merge_base(spec), renames=False)}

    def drop(self, change: LabChange) -> None:
        """Delete the change: its worktrees (uncommitted work included), its branches, sandbox home and state."""
        self.remove_trees(change)
        state_dir = self._state_dir(change.id)
        if os.path.isdir(state_dir):
            shutil.rmtree(state_dir)

    def remove_trees(self, change: LabChange) -> None:
        """Remove the change's worktrees (uncommitted work included), branches and sandbox home; its state is kept.

        The sandbox home (``<root>/changes/<id>.sandbox-home``, the ``$HOME`` of its shell, checks and installs)
        that cannot be deleted is logged at warning level: the trees and branches are gone by then.
        """
        trees_root = self._trees_root(change.id)
        with self._all_repo_locks():
            made = [(spec, change.branches[spec.key]) for spec in self._creation_order() if spec.key in change.branches]
            self._remove_worktrees(trees_root, made)
        home = sandbox_home(trees_root)
        if os.path.lexists(home):
            try:
                shutil.rmtree(home)
            except OSError as exc:
                logger.warning("BESSER Lab: the sandbox home %s of change %s was not deleted: %s", home, change.id,
                               exc)

    def _remove_worktrees(self, trees_root: str, made: Sequence[tuple[RepoSpec, str]]) -> None:
        """Remove the worktrees under ``trees_root`` of ``made`` (in creation order) and their branches.

        Children go first. The caller holds the repository locks. A tree already gone only has
        its stale entry pruned.
        """
        for spec, branch in reversed(made):
            tree = self._tree(trees_root, spec)
            try:
                self._worktree_git_dir(spec, tree)
            except LabWorkspaceError:
                pass  # never made, or its folder was deleted: prune below drops the entry
            else:
                self._git(spec, "worktree", "remove", "--force", tree)
            self._git(spec, "worktree", "prune")
            if self._has_ref(spec, f"refs/heads/{branch}"):
                self._git(spec, "branch", "-q", "-D", branch)
        if os.path.isdir(trees_root):
            shutil.rmtree(trees_root)

    # -- bases -------------------------------------------------------------------------------------

    def refresh_bases(self) -> dict[str, RepoBase]:
        """Fetch each base branch from upstream, fast-forward the base clone to it and prune stale worktrees.

        Returns where each base now is. Open changes never move: their branches are their own.
        """
        bases = {}
        for key in self._keys():
            spec = self.base_repos[key]
            upstream_ref = _UPSTREAM_REFS + spec.base_branch
            with self.repo_lock(key):
                self._git(spec, "fetch", "-q", "--no-tags", upstream_url(spec),
                          f"+refs/heads/{spec.base_branch}:{upstream_ref}",
                          config=_ANONYMOUS, timeout=_FETCH_TIMEOUT_SECONDS)
                self._git(spec, "merge", "-q", "--ff-only", upstream_ref)
                self._git(spec, "worktree", "prune")
                bases[key] = RepoBase(f"{spec.github_owner}/{spec.github_name}", spec.base_branch,
                                      self._rev(spec, _base_ref(spec)))
        return bases

    def base_status(self, change: LabChange) -> dict[str, int]:
        """Repository -> commits its upstream base gained since the change started (as of the last refresh)."""
        status = {}
        for key, base in change.bases.items():
            spec = self._spec(key)
            count = self._git(spec, "rev-list", "--count", f"{base.sha}..{_UPSTREAM_REFS}{base.branch}")
            status[key] = int(count.strip())
        return status


def _base_ref(spec: RepoSpec) -> str:
    return f"refs/heads/{spec.base_branch}"


def _check_id(change_id: str) -> None:
    if not _CHANGE_ID_RE.match(change_id):
        raise LabWorkspaceError(f"invalid change id {change_id!r}")


def _require_open(change: LabChange) -> None:
    if change.status == "draft":
        raise LabWorkspaceError(f"change {change.id} is a draft: it has no branches yet")
    if change.status != "open":
        raise LabWorkspaceError(f"change {change.id} is already closed: its trees and branches were removed")


def _parse_numstat(out: str) -> list[tuple[Optional[int], Optional[int], str, Optional[str]]]:
    """``git diff --numstat -z`` -> (additions, deletions, path, old path) per file; counts None when binary.

    A record is ``added<TAB>deleted<TAB>path<NUL>``, or for a rename ``added<TAB>deleted<TAB><NUL>old<NUL>new<NUL>``.
    """
    tokens = out.split("\0")
    entries = []
    index = 0
    while index < len(tokens) and tokens[index]:
        added, deleted, path = tokens[index].split("\t", 2)
        old_path = None
        if path:
            index += 1
        else:
            old_path, path = tokens[index + 1], tokens[index + 2]
            index += 3
        binary = added == "-"
        entries.append((None if binary else int(added), None if binary else int(deleted), path, old_path))
    return entries


def _patch_status(patch: str) -> str:
    """A file's patch -> :attr:`FileDiff.status`, from the extended header lines before its first hunk."""
    for line in patch.splitlines():
        if line.startswith("@@"):
            break
        for prefix, status in _STATUS_BY_HEADER:
            if line.startswith(prefix):
                return status
    return "modified"


def _change_from_state(data: dict) -> LabChange:
    """A ``state.json`` object -> :class:`LabChange`; a phase 1 file loads with the new fields' defaults."""
    data = dict(data)
    data.pop("version", None)
    if "user" in data:
        data.setdefault("owner", data.pop("user"))
    if data.get("status") in _PHASE_1_STATUSES:
        data["status"] = _PHASE_1_STATUSES[data["status"]]
    data["kind"] = ChangeKind(data["kind"])
    data["bases"] = {key: RepoBase(**base) for key, base in data.get("bases", {}).items()}
    return LabChange(**data)
