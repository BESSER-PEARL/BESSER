"""The diff of a BESSER Lab change as ``GET /lab/diff`` sends it: per file, capped, secrets redacted.

Git computes the diff (``LabWorkspace.file_diffs``); this module decides what leaves the
server. Every patch is redacted before it is sent, with the count of what was replaced. A
file over the per-file cap, or past the response's total cap, comes back without its patch
and says why (``omitted``); the editor then asks for that one file alone, under a higher cap.
"""

from __future__ import annotations

from typing import Optional

from besser.spec_driven_agent.lab.contracts import REPO_ORDER, FileDiff, LabChange
from besser.spec_driven_agent.lab.workspace import LabWorkspace
from besser.utilities.secret_redaction import redact_text
from besser.utilities.web_modeling_editor.backend.models.lab import LabDiffResponse, LabFileDiff, LabRepoDiff

# One screen of review; lock files and generated bundles exceed it.
MAX_FILE_PATCH_BYTES = 200_000
# The cap of a file the editor fetches alone, after the list said it was too large.
MAX_SINGLE_FILE_PATCH_BYTES = 2_000_000
# What one response carries in all; the files past it are fetched one by one.
MAX_DIFF_RESPONSE_BYTES = 2_000_000

# ``.env`` files with these endings hold placeholders, not secrets (the push boundary's rule,
# ``secret_redaction.scrub_secret_files``): only the value rules run on them.
_ENV_TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist")


def diff_response(workspace: LabWorkspace, change: LabChange, *, repo: Optional[str] = None,
                  path: Optional[str] = None) -> LabDiffResponse:
    """The change's working trees against their bases, as the diff route answers.

    Args:
        workspace: The lab's workspace.
        change: The change; a draft has no diff.
        repo: Only this repository; every repository of the change when None.
        path: With ``repo``: only this file, under :data:`MAX_SINGLE_FILE_PATCH_BYTES`.

    Raises:
        LabWorkspaceError: The change is closed (its trees are gone), or git failed.
        ValueError: ``path`` without ``repo``.
    """
    if path is not None and repo is None:
        raise ValueError("a file of the diff is named by its repository and its path")
    repos = [repo] if repo is not None else [key for key in REPO_ORDER if key in change.branches]
    file_cap = MAX_FILE_PATCH_BYTES if path is None else MAX_SINGLE_FILE_PATCH_BYTES
    budget = MAX_DIFF_RESPONSE_BYTES
    result = []
    for key in repos:
        files = []
        for diff in workspace.file_diffs(change, key, path=path):
            view, size = _file_view(diff, file_cap=file_cap, room=budget)
            budget -= size
            files.append(view)
        base = change.bases.get(key)
        result.append(LabRepoDiff(repo=key, branch=change.branches.get(key, ""),
                                  base=base.branch if base else workspace.base_repos[key].base_branch, files=files))
    return LabDiffResponse(changeId=change.id, repos=result)


def _file_view(diff: FileDiff, *, file_cap: int, room: int) -> tuple[LabFileDiff, int]:
    """One file as the route sends it, and the bytes its patch takes from the response."""
    fields = dict(path=diff.path, oldPath=diff.old_path, status=diff.status, additions=diff.additions,
                  deletions=diff.deletions, binary=diff.binary)
    if diff.binary:
        return LabFileDiff(**fields, patch=None, omitted="binary", redactions=0), 0
    # Measured before redacting: a megabyte lock file is not scanned to be dropped.
    if len(diff.patch.encode("utf-8")) > file_cap:
        return LabFileDiff(**fields, patch=None, omitted="too_large", redactions=0), 0
    patch, redactions = _redact_patch(diff.path, diff.patch)
    size = len(patch.encode("utf-8"))
    if size > room:
        return LabFileDiff(**fields, patch=None, omitted="response_full", redactions=0), 0
    return LabFileDiff(**fields, patch=patch, omitted=None, redactions=redactions), size


def _redact_patch(path: str, patch: str) -> tuple[str, int]:
    """``patch`` with its secrets replaced, and how many; a ``.env`` file's assignments too, as at the push."""
    name = path.rsplit("/", 1)[-1].lower()
    if not name.startswith(".env") or name.endswith(_ENV_TEMPLATE_SUFFIXES):
        return redact_text(patch, env_style=False)
    # The assignment rule reads ``KEY=value`` at a line start: it runs on each line without its diff marker.
    lines, total = [], 0
    for line in patch.split("\n"):
        if line[:1] in ("+", "-", " ") and not line.startswith(("+++ ", "--- ")):
            content, count = redact_text(line[1:], env_style=True)
            line, total = line[:1] + content, total + count
        lines.append(line)
    return "\n".join(lines), total
