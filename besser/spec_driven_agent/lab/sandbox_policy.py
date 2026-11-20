"""The lab's one mount policy for the agent's shell, the checks, installs and the preview.

Read-write: the change's top-level trees (BESSER with the editor inside it, the modeling agent). Read-only: the
git root, so sandboxed code can read history but never rewrite a git directory the lab's own git reads.
Masked (an empty tmpfs): the workspace root, which hides the base clones, the shared caches and every other
change, plus the folders the caller names (state, sessions, run store, the lab's own checkout).
"""

from __future__ import annotations

import os
from typing import Mapping, Protocol, Sequence

from besser.spec_driven_agent.execution.sandbox import SandboxMounts
from besser.spec_driven_agent.lab.contracts import RepoSpec

_WORKSPACE_ROOT_SETTING = "BESSER_LAB_WORKSPACE_ROOT"
_GIT_ROOT_SETTING = "BESSER_LAB_GIT_ROOT"


class LabSandboxPolicyError(Exception):
    """The lab's folders overlap so that a sandbox would show or let code rewrite what it must not."""


class SandboxPolicy(Protocol):
    """The mounts for a change's trees; the service binds the folders with :func:`functools.partial`."""

    def __call__(self, repos: Mapping[str, RepoSpec], *, writable: bool = True) -> SandboxMounts: ...


def lab_sandbox_mounts(repos: Mapping[str, RepoSpec], *, workspace_root: str, git_root: str,
                       hidden: Sequence[str], writable: bool = True) -> SandboxMounts:
    """The mount rules of a sandbox working on ``repos`` (a change's trees, or the base trees of a draft).

    ``writable=False`` (a draft's plan step) binds the trees read-only next to the git root. A ``hidden``
    path that does not exist is dropped (nothing to hide). Raises :class:`LabSandboxPolicyError` naming the
    setting to change when the git root or a hidden path lies inside a tree, a tree inside the git root, or a
    masked path inside the git root (each would show what the policy hides or make a tree read-only).
    """
    trees = _top_level_trees(spec.path for spec in repos.values())
    git_root = os.path.realpath(git_root)
    masked = [os.path.realpath(workspace_root)]
    for path in hidden:
        real = os.path.realpath(path)
        if os.path.isdir(real):
            masked.append(real)
        elif os.path.exists(real):
            raise LabSandboxPolicyError(f"{path} is a file: the sandbox can hide only a folder; name its folder")
    for tree in trees:
        if _inside(git_root, tree) or _inside(tree, git_root):
            raise LabSandboxPolicyError(
                f"{_GIT_ROOT_SETTING} ({git_root}) and the working tree {tree} overlap: set {_GIT_ROOT_SETTING} "
                f"to a folder outside {_WORKSPACE_ROOT_SETTING}")
        for path in masked[1:]:
            if _inside(path, tree):
                raise LabSandboxPolicyError(
                    f"{path} is inside the working tree {tree}, so the agent's sandbox would show it: move it "
                    f"outside {_WORKSPACE_ROOT_SETTING}")
    for path in masked:
        if _inside(path, git_root):
            raise LabSandboxPolicyError(
                f"{path} is inside {_GIT_ROOT_SETTING} ({git_root}), which every sandbox reads: move it out")
    if writable:
        return SandboxMounts(writable=trees, read_only=(git_root,), masked=tuple(masked))
    return SandboxMounts(writable=(), read_only=(git_root, *trees), masked=tuple(masked))


def _top_level_trees(paths) -> tuple[str, ...]:
    """The trees to bind, in the given order; a tree inside another (the editor in BESSER) is bound with it."""
    real = [os.path.realpath(path) for path in paths]
    trees: list[str] = []
    for path in real:
        if path not in trees and not any(_inside(path, other) for other in real if other != path):
            trees.append(path)
    return tuple(trees)


def _inside(path: str, parent: str) -> bool:
    """Whether ``path`` is ``parent`` or below it (both real paths)."""
    path, parent = os.path.normcase(path), os.path.normcase(parent)
    try:
        return os.path.commonpath([path, parent]) == parent
    except ValueError:  # different drives (Windows)
        return False
