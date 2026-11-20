"""The lab's working copy: one git checkout per repository, a branch per change, and the change's state.

Layout under ``root``: ``besser/`` (with the editor submodule checked out at
:data:`FRONTEND_SUBMODULE_PATH`) and ``modeling-agent/``. The lab's state folder
(``state_dir``, by default ``<root>/.lab``) holds ``changes/<id>/`` with
``state.json`` and ``notes.md`` of every change. A state folder outside ``root``
is out of reach of the lab agent's sandboxed shell, which can write all of ``root``.

A change creates its branch ``lab/<user>/<slug>`` in every repository when it
starts (eagerly), from the commit each checkout is on. A repository the change
never modifies then has a branch identical to its starting point, which
:meth:`LabWorkspace.repo_changes` and :meth:`LabWorkspace.commit_all` leave out.

Git always runs through an argv list with the scrubbed child environment. Paths
are read and written in git's C-quoted form (``core.quotepath``), so git's
output is plain ASCII whatever the locale's encoding is.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import uuid
from dataclasses import asdict
from typing import Mapping, Optional, Sequence

from besser.spec_driven_agent.execution.process import run_bounded
from besser.spec_driven_agent.lab.contracts import (
    FRONTEND_SUBMODULE_PATH,
    LAB_BRANCH_PREFIX,
    LAB_STATE_DIRNAME,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    ChangeKind,
    FileChange,
    LabChange,
    RepoChange,
    RepoSpec,
)
from besser.utilities.secret_redaction import redact_text
from besser.utilities.subprocess_env import safe_subprocess_env

logger = logging.getLogger(__name__)

# Applied to every git call: no line-ending conversion, and non-ASCII paths quoted. The lab agent's tools can
# write a checkout's .git/, so git runs none of its hooks and no fsmonitor command: they would run unconfined.
_GIT_CONFIG = ("core.autocrlf=false", "core.quotepath=true", "core.hooksPath=/dev/null", "core.fsmonitor=false")

# Run artefacts of the agent's tools and the shell sandbox's home: never part of a change.
_HARNESS_PREFIX = ".besser_"
_SANDBOX_HOME_SUFFIX = ".sandbox-home"

# Long enough for a readable title, short enough for a branch name in a URL.
_MAX_SLUG_LENGTH = 48

# Change ids reach the file system from HTTP requests (resume), so only this shape is accepted.
_CHANGE_ID_RE = re.compile(r"^[0-9a-f]{12}$")

# Where a pushed lab branch is recorded: start_change counts it as taken after the change is dropped.
_PUSHED_REFS = "refs/remotes/github/"

# First letter of a ``git diff --name-status`` code -> FileChange.status
_STATUS_BY_CODE = {"A": "added", "C": "added", "D": "deleted", "M": "modified", "R": "renamed", "T": "modified"}


class LabWorkspaceError(Exception):
    """A git operation or a change's state does not allow what was asked."""


def default_repos(root: str) -> tuple[RepoSpec, ...]:
    """The three repositories of a lab working copy under ``root``, in pull-request order."""
    besser = os.path.join(root, "besser")
    return (
        RepoSpec(REPO_FRONTEND, os.path.join(besser, *FRONTEND_SUBMODULE_PATH.split("/")), "develop",
                 "BESSER-PEARL", "BESSER-Web-Modeling-Editor"),
        RepoSpec(REPO_BESSER, besser, "development", "BESSER-PEARL", "BESSER"),
        RepoSpec(REPO_MODELING_AGENT, os.path.join(root, "modeling-agent"), "develop",
                 "BESSER-PEARL", "modeling-agent"),
    )


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


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _identity(change: LabChange) -> tuple[str, str]:
    """Git config making the change's user the author and committer."""
    return f"user.name={change.user}", f"user.email={change.user}@users.noreply.github.com"


class LabWorkspace:
    """The repositories of the lab and the changes made in them. At most one change is active."""

    def __init__(self, root: str, repos: Optional[Sequence[RepoSpec]] = None, *, state_dir: Optional[str] = None):
        self.root: str = os.path.abspath(root)
        specs = default_repos(self.root) if repos is None else repos
        self.repos: dict[str, RepoSpec] = {spec.key: spec for spec in specs}
        self._state_dir_path = os.path.abspath(state_dir or os.path.join(self.root, LAB_STATE_DIRNAME))
        self._changes_dir = os.path.join(self._state_dir_path, "changes")

    @property
    def state_dir(self) -> str:
        """The folder of the lab's state: ``changes/<id>/`` and whatever else the lab keeps between messages."""
        return self._state_dir_path

    # -- git ---------------------------------------------------------------------------------------

    def run_git(self, repo: str, *args: str, env_extra: Optional[Mapping[str, str]] = None,
                timeout: float = 120) -> str:
        """Run ``git <args>`` in ``repo`` and return its standard output.

        ``env_extra`` is added to the scrubbed environment; its values are also redacted
        from an error message. Raises :class:`LabWorkspaceError` on failure or timeout.
        """
        return self._git(repo, *args, env_extra=env_extra, timeout=timeout)

    def _run(self, repo: str, args: Sequence[str], *, input: Optional[str] = None, config: Sequence[str] = (),
             env_extra: Optional[Mapping[str, str]] = None, timeout: float = 120) -> subprocess.CompletedProcess:
        spec = self._spec(repo)
        env = safe_subprocess_env()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.update(env_extra or {})
        argv = ["git"]
        for item in (*_GIT_CONFIG, *config):
            argv += ["-c", item]
        argv += list(args)
        try:
            return run_bounded(argv, timeout=timeout, cwd=spec.path, env=env, input=input)
        except subprocess.TimeoutExpired:
            raise LabWorkspaceError(f"git {args[0]} timed out after {timeout:g} s in {repo}") from None
        except OSError as exc:
            raise LabWorkspaceError(f"cannot run git in {repo} ({spec.path}): {exc}") from None

    def _git(self, repo: str, *args: str, input: Optional[str] = None, config: Sequence[str] = (),
             env_extra: Optional[Mapping[str, str]] = None, timeout: float = 120) -> str:
        result = self._run(repo, args, input=input, config=config, env_extra=env_extra, timeout=timeout)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "no output"
            detail, _ = redact_text(detail, secrets=list((env_extra or {}).values()))
            raise LabWorkspaceError(f"git {args[0]} failed in {repo} (exit {result.returncode}): {detail}")
        return result.stdout

    def _spec(self, repo: str) -> RepoSpec:
        try:
            return self.repos[repo]
        except KeyError:
            raise LabWorkspaceError(f"unknown repository {repo!r}") from None

    def _has_ref(self, repo: str, ref: str) -> bool:
        return self._run(repo, ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]).returncode == 0

    def _base_ref(self, repo: str) -> str:
        """The local base branch, or origin's when the checkout has none (a fresh submodule)."""
        base = self._spec(repo).base_branch
        for ref in (f"refs/heads/{base}", f"refs/remotes/origin/{base}"):
            if self._has_ref(repo, ref):
                return ref
        raise LabWorkspaceError(f"{repo}: base branch {base} not found locally or on origin")

    def _merge_base(self, repo: str) -> str:
        return self._git(repo, "merge-base", "HEAD", self._base_ref(repo)).strip()

    def _is_ancestor(self, repo: str, ancestor: str, descendant: str) -> bool:
        return self._run(repo, ["merge-base", "--is-ancestor", ancestor, descendant]).returncode == 0

    def _current_branch(self, repo: str) -> str:
        return self._git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()

    def _branch_names(self, repo: str) -> set[str]:
        names = set()
        for ref in self._git(repo, "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes").split():
            if ref.startswith("refs/heads/"):
                names.add(ref[len("refs/heads/"):])
            elif ref.count("/") >= 3:
                names.add(ref.split("/", 3)[3])
        return names

    def _untracked(self, repo: str) -> list[str]:
        paths = (_unquote(line) for line in
                 self._git(repo, "ls-files", "--others", "--exclude-standard").splitlines())
        return [path for path in paths if not _is_harness_path(path)]

    def _changed_files(self, repo: str, since: str, *, renames: bool) -> list[FileChange]:
        """Working tree (staged or not, untracked included) against ``since``, run artefacts left out.

        ``--ignore-submodules=all`` keeps a submodule's pointer out of its parent's list:
        the submodule reports its own files.
        """
        out = self._git(repo, "diff", "--name-status", "--ignore-submodules=all",
                        "-M" if renames else "--no-renames", since, "--")
        changes = []
        for line in out.splitlines():
            fields = line.split("\t")
            path = _unquote(fields[-1])
            if not _is_harness_path(path):
                changes.append(FileChange(path, _STATUS_BY_CODE.get(fields[0][:1], "modified")))
        changes += [FileChange(path, "added") for path in self._untracked(repo)]
        return changes

    def _require_on_branch(self, repo: str, branch: str) -> None:
        current = self._current_branch(repo)
        if current != branch:
            raise LabWorkspaceError(f"{repo} is on {current}, not on the change's branch {branch}")

    def _change_branches(self, change: LabChange) -> list[tuple[str, str]]:
        return [(key, change.branches[key]) for key in self.repos if key in change.branches]

    def _switch_to_base(self, repo: str) -> None:
        """Check out the base branch and delete the files only the change's ignore rules hid.

        Such a file (a build output the change's ``.gitignore`` ignores) would otherwise show
        as uncommitted work on the base branch and block every later change.
        """
        hidden = [_unquote(line) for line in self._git(
            repo, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory").splitlines()]
        self._git(repo, "switch", "-q", self._spec(repo).base_branch)
        exposed = [path for path in self._untracked(repo)
                   if any(path == entry or (entry.endswith("/") and path.startswith(entry)) for entry in hidden)]
        for path in exposed:
            target = os.path.join(self._spec(repo).path, *path.rstrip("/").split("/"))
            if path.endswith("/"):
                shutil.rmtree(target)
            else:
                os.remove(target)
        if exposed:
            logger.warning("BESSER Lab: deleted %d file(s) in %s that only the change's ignore rules hid: %s",
                           len(exposed), repo, ", ".join(exposed[:5]))

    # -- readiness ---------------------------------------------------------------------------------

    def ensure_ready(self) -> None:
        """Check that every repository is its own git checkout with its base branch available."""
        for key, spec in self.repos.items():
            if not os.path.isdir(spec.path):
                raise LabWorkspaceError(f"{key}: {spec.path} does not exist")
            top = self._git(key, "rev-parse", "--show-toplevel").strip()
            if not _same_path(top, spec.path):
                raise LabWorkspaceError(f"{key}: {spec.path} is not a git checkout of its own (git sees {top}); "
                                        "is the submodule initialised?")
            self._base_ref(key)

    # -- change state ------------------------------------------------------------------------------

    def _state_dir(self, change_id: str) -> str:
        if not _CHANGE_ID_RE.match(change_id):
            raise LabWorkspaceError(f"invalid change id {change_id!r}")
        return os.path.join(self._changes_dir, change_id)

    def _load(self, change_id: str) -> LabChange:
        path = os.path.join(self._state_dir(change_id), "state.json")
        if not os.path.isfile(path):
            raise LabWorkspaceError(f"no change {change_id}")
        # The model's shell can write a state folder inside root: a broken file is a refusal naming it, not a 500.
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            data["kind"] = ChangeKind(data["kind"])
            return LabChange(**data)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise LabWorkspaceError(f"the state of change {change_id} ({path}) cannot be read: {exc}; "
                                    "fix or delete that file") from None

    def _changes(self) -> list[LabChange]:
        if not os.path.isdir(self._changes_dir):
            return []
        ids = [name for name in os.listdir(self._changes_dir)
               if _CHANGE_ID_RE.match(name) and os.path.isfile(os.path.join(self._changes_dir, name, "state.json"))]
        return sorted((self._load(change_id) for change_id in ids), key=lambda change: change.created_at)

    def active_change(self) -> Optional[LabChange]:
        """The change checked out in the working copy, if any."""
        return next((change for change in self._changes() if change.status == "active"), None)

    def parked_changes(self) -> list[LabChange]:
        """Parked changes, oldest first."""
        return [change for change in self._changes() if change.status == "parked"]

    def save_change(self, change: LabChange) -> None:
        """Write the change's state (atomically)."""
        directory = self._state_dir(change.id)
        os.makedirs(directory, exist_ok=True)
        data = asdict(change)
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

    def ensure_clean(self) -> None:
        """Refuse a working copy with uncommitted work (run artefacts aside): it would leak into a change."""
        for key in self.repos:
            files = self._changed_files(key, "HEAD", renames=False)
            if files:
                raise LabWorkspaceError(f"{key} has uncommitted changes outside any change "
                                        f"({', '.join(f.path for f in files[:5])}); reset the working copy first")

    def start_change(self, *, user: str, title: str, kind: ChangeKind) -> LabChange:
        """Create a change and its branch ``lab/<user>/<slug>`` in every repository, checked out.

        Raises when a change is active or a working copy has uncommitted changes. The state
        is saved before the branches are created, so a failure part way can be dropped. The
        slug is unique against local branches, origin's and those the lab pushed (``record_pushed``).
        """
        active = self.active_change()
        if active is not None:
            raise LabWorkspaceError(f"change {active.id} ({active.title}) is active; park or drop it first")
        self.ensure_ready()
        self.ensure_clean()
        taken = set().union(*(self._branch_names(key) for key in self.repos))
        prefix = f"{LAB_BRANCH_PREFIX}{_slug(user, 'user')}/"
        slug = base_slug = _slug(title, "change")
        suffix = 2
        while prefix + slug in taken:
            slug = f"{base_slug}-{suffix}"
            suffix += 1
        branch = prefix + slug
        change = LabChange(id=uuid.uuid4().hex[:12], slug=slug, title=title, user=user, kind=kind,
                           status="active", created_at=time.time(), branches={key: branch for key in self.repos})
        self.save_change(change)
        for key in self.repos:
            self._git(key, "switch", "-q", "-c", branch)
        return change

    def repo_changes(self, change: LabChange) -> list[RepoChange]:
        """Committed and uncommitted changes of every repository against its merge-base with the base branch.

        Only repositories with changes are listed. The change must be checked out.
        """
        result = []
        for key, branch in self._change_branches(change):
            self._require_on_branch(key, branch)
            files = sorted(self._changed_files(key, self._merge_base(key), renames=True), key=lambda f: f.path)
            if files:
                result.append(RepoChange(repo=key, branch=branch, base=self.repos[key].base_branch,
                                         files=tuple(files)))
        return result

    def commit_all(self, change: LabChange, message: str) -> dict[str, str]:
        """Commit the uncommitted work of every repository, authored as the change's user.

        Returns repository -> new commit for the repositories that had work. Only the changed
        paths are staged and committed: never run artefacts, never a submodule pointer.
        """
        literal = {"GIT_LITERAL_PATHSPECS": "1"}
        work = {}
        for key, branch in self._change_branches(change):
            self._require_on_branch(key, branch)
            work[key] = self._changed_files(key, "HEAD", renames=False)
            # git lists an untracked repository (git init, a scaffolder) as "dir/" and cannot add its files.
            nested = [f.path for f in work[key] if f.path.endswith("/")]
            if nested:
                raise LabWorkspaceError(f"{key}: {', '.join(nested)} is a git repository of its own (made by git init "
                                        "or a scaffolder); delete its .git folder to keep its files in the change")
        shas = {}
        for key, files in work.items():
            if not files:
                continue
            pathspecs = _pathspec_input(sorted({f.path for f in files}))
            self._git(key, "add", "-A", "--pathspec-from-file=-", input=pathspecs, env_extra=literal)
            self._git(key, "commit", "-q", "-m", message, "--pathspec-from-file=-",
                      input=pathspecs, config=_identity(change), env_extra=literal)
            shas[key] = self._git(key, "rev-parse", "HEAD").strip()
        return shas

    def merge_github_work(self, change: LabChange, repo: str, source: str, ref: str) -> None:
        """Merge into the change's branch the commits added to it on GitHub (a suggestion, "Update branch").

        ``source`` is a git directory holding GitHub's branch at ``ref``. Nothing happens when the
        change's branch already contains it. Refused when that branch shares no commit with the
        change (an earlier change of the same name left it) and when the merge conflicts.
        """
        branch = change.branches[repo]
        self._require_on_branch(repo, branch)
        self._git(repo, "fetch", "-q", "--no-tags", source, ref)
        theirs = self._git(repo, "rev-parse", "FETCH_HEAD").strip()
        common = self._git(repo, "merge-base", "HEAD", theirs).strip()
        if common == theirs:
            return
        if self._is_ancestor(repo, common, self._base_ref(repo)):
            raise LabWorkspaceError(f"{repo}: the branch {branch} on GitHub holds commits that are not part of this "
                                    "change (an earlier change of the same name left it); delete it on GitHub")
        merged = self._run(repo, ["merge", "-q", "--no-edit", theirs], config=_identity(change))
        if merged.returncode != 0:
            conflicts = ", ".join(self._git(repo, "diff", "--name-only", "--diff-filter=U").split()[:5])
            self._run(repo, ["merge", "--abort"])
            raise LabWorkspaceError(f"{repo}: the branch {branch} was changed on GitHub and merging those commits "
                                    f"failed ({conflicts or (merged.stderr or merged.stdout).strip()}); "
                                    "undo them on GitHub or resolve the conflict by hand")

    def record_pushed(self, repo: str, branch: str, sha: str) -> None:
        """Remember that ``branch`` exists on GitHub, so no later change reuses its name."""
        self._git(repo, "update-ref", _PUSHED_REFS + branch, sha)

    def park(self, change: LabChange) -> None:
        """Commit the work in progress and check out the base branches; the change becomes parked."""
        if change.status != "active":
            raise LabWorkspaceError(f"change {change.id} is not active")
        self.commit_all(change, f"wip: {change.title}")
        for key, _ in self._change_branches(change):
            self._switch_to_base(key)
        change.status = "parked"
        self.save_change(change)

    def resume(self, change_id: str) -> LabChange:
        """Check out a parked change's branches again; it becomes the active change."""
        active = self.active_change()
        if active is not None:
            raise LabWorkspaceError(f"change {active.id} ({active.title}) is active; park or drop it first")
        change = self._load(change_id)
        if change.status != "parked":
            raise LabWorkspaceError(f"change {change_id} is not parked")
        self.ensure_clean()
        for key, branch in self._change_branches(change):
            self._git(key, "switch", "-q", branch)
        change.status = "active"
        self.save_change(change)
        return change

    def drop(self, change: LabChange) -> None:
        """Discard the change: its uncommitted work (when active), its branches and its state.

        Refused, before anything is discarded, when a repository holding the change's branch has
        another branch checked out: its work is not the change's. A repository without the branch
        (a start that failed part way) is left as it is.
        """
        branches = self._change_branches(change)
        if change.status == "active":
            checked_out = []
            for key, branch in branches:
                current = self._current_branch(key)
                if current == branch:
                    checked_out.append(key)
                elif self._has_ref(key, f"refs/heads/{branch}"):
                    raise LabWorkspaceError(f"{key} is on {current}, not on the change's branch {branch}; "
                                            "nothing was dropped")
            for key in checked_out:
                self._git(key, "reset", "-q", "--hard")
                # -ff: also an untracked repository made inside the checkout (git init, a scaffolder).
                self._git(key, "clean", "-ffdq")
                self._switch_to_base(key)
        for key, branch in branches:
            if self._has_ref(key, f"refs/heads/{branch}"):
                self._git(key, "branch", "-q", "-D", branch)
        state_dir = self._state_dir(change.id)
        if os.path.isdir(state_dir):
            shutil.rmtree(state_dir)

    # -- review ------------------------------------------------------------------------------------

    def touched_paths(self, repo: str) -> set[str]:
        """Every path ``repo`` changes against its merge-base, a renamed file under its old and new path."""
        return {f.path for f in self._changed_files(repo, self._merge_base(repo), renames=False)}

    def diff(self, repo: str, *, max_bytes: int = 200_000) -> str:
        """Unified diff of ``repo``'s working tree (new files included) against its merge-base.

        Longer output is cut at ``max_bytes`` with a note saying so.
        """
        merge_base = self._merge_base(repo)
        untracked = self._untracked(repo)
        env: dict[str, str] = {}
        index_copy = None
        try:
            if untracked:
                # New files enter the diff through intent-to-add entries in a copy of the index,
                # so the real index is left as it was.
                handle, index_copy = tempfile.mkstemp(prefix="besser_lab_index_")
                os.close(handle)
                index = os.path.join(self._spec(repo).path, self._git(repo, "rev-parse", "--git-path", "index").strip())
                shutil.copyfile(index, index_copy)
                env = {"GIT_INDEX_FILE": index_copy, "GIT_LITERAL_PATHSPECS": "1"}
                self._git(repo, "add", "-N", "--pathspec-from-file=-", input=_pathspec_input(untracked), env_extra=env)
            text = self._git(repo, "diff", "--no-color", "--no-ext-diff", "-M", "--ignore-submodules=all",
                             merge_base, "--", env_extra=env)
        finally:
            if index_copy is not None:
                os.remove(index_copy)
        data = text.encode("utf-8")
        if len(data) <= max_bytes:
            return text
        return (data[:max_bytes].decode("utf-8", errors="ignore")
                + f"\n[diff truncated: {max_bytes} of {len(data)} bytes shown; read the files for the rest]\n")
