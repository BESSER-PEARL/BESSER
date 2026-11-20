"""Open the draft pull requests of a BESSER Lab change.

Each repository the change touched gets its lab branch pushed with the user's
GitHub token and a draft pull request against its base branch, the editor
first: BESSER's branch points its editor submodule at the pushed editor commit.
The token reaches git only through the environment, never argv, a remote URL
or git config, and only in a scratch repository: never with the working copy's
own config, which the lab agent can write. Commits added to a lab branch on
GitHub are merged before the push; a pull request already open keeps its body
except for the related section.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional, Sequence

import httpx

from besser.spec_driven_agent.lab.contracts import (
    FRONTEND_SUBMODULE_PATH,
    LAB_BRANCH_PREFIX,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_ORDER,
    ChangeKind,
    LabChange,
    RepoChange,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError
from besser.utilities.secret_redaction import redact_text, scrub_secret_files
from besser.utilities.web_modeling_editor.backend.constants.constants import TEMP_DIR_PREFIX
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import GitHubService

# The sign-in token has no ``workflow`` scope, so GitHub refuses a push that changes these.
_WORKFLOWS_DIR = ".github/workflows/"
# A pull request body lists at most this many changed files; the diff on GitHub has the rest.
_MAX_LISTED_FILES = 50
_PUSH_TIMEOUT_SECONDS = 300
# The scratch repository's copy of the lab branch as GitHub has it.
_GITHUB_HEAD = "refs/lab/github"
_RELATED_HEADING = "## Related pull requests"


@dataclass(frozen=True)
class PullRequestInfo:
    """One pull request opened for a change."""

    repo: str
    number: int
    url: str
    branch: str
    base: str


class PullRequestError(Exception):
    """The pull requests could not be opened; the message is safe to show (no token)."""


def push_environment(access_token: str) -> dict[str, str]:
    """The extra git environment for a push to github.com authenticated with ``access_token``.

    ``GIT_CONFIG_COUNT`` entries act as command-line config: an ``Authorization``
    header for github.com and an empty credential helper, so no stored credential
    is used and nothing is written to any git config file.
    """
    basic = base64.b64encode(f"x-access-token:{access_token}".encode()).decode()
    return {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}",
        "GIT_CONFIG_KEY_1": "credential.helper",
        "GIT_CONFIG_VALUE_1": "",
        "GIT_TERMINAL_PROMPT": "0",
    }


def default_title(change: LabChange) -> str:
    """A Conventional Commit subject from the change's kind and title."""
    kind = "fix" if change.kind == ChangeKind.BUGFIX else "feat"
    title = change.title.strip().rstrip(".")
    # Keep an acronym ("BPMN ...") as written; lower-case an ordinary first word.
    if title[:2] != title[:2].upper():
        title = title[:1].lower() + title[1:]
    return f"{kind}: {title}"


async def open_pull_requests(
    workspace: LabWorkspace,
    change: LabChange,
    *,
    access_token: str,
    login: str,
    title: Optional[str] = None,
    body: Optional[str] = None,
    github: Optional[GitHubService] = None,
) -> list[PullRequestInfo]:
    """Push the change's lab branches and open a draft pull request in each repository it touched.

    Args:
        workspace: The lab's working copy, with ``change`` checked out.
        change: The active change.
        access_token: The signed-in user's GitHub token; it pushes and opens the pull requests.
        login: The user's GitHub login, the author of the submodule commit.
        title: The pull requests' title; defaults to ``default_title(change)``.
        body: Their description; defaults to the change's title and its changed files.
        github: The GitHub client; one is made from ``access_token`` when omitted.

    Returns:
        The pull requests in merge order (editor, BESSER, modeling agent).

    Raises:
        PullRequestError: Nothing to open, a change that may not be pushed (refused before any
            push), or a git or GitHub failure. The message never holds the token.
    """
    def safe(text: str) -> str:
        return redact_text(text, env_style=False, secrets=[access_token])[0]

    try:
        changes = await asyncio.to_thread(_changes_in_merge_order, workspace, change)
        if not changes:
            raise PullRequestError("This change has no modifications to open a pull request for.")
        await asyncio.to_thread(_refuse_unpushable, workspace, changes)
        title = title or default_title(change)
        await asyncio.to_thread(workspace.commit_all, change, title)
        repos_with_changes = {repo_change.repo for repo_change in changes}
        for repo_change in changes:
            if repo_change.repo == REPO_BESSER and REPO_FRONTEND in repos_with_changes:
                await asyncio.to_thread(_point_submodule_at_editor_branch, workspace, changes, login)
            await asyncio.to_thread(_push, workspace, change, repo_change, access_token)
    except LabWorkspaceError as exc:
        raise PullRequestError(safe(f"Pushing the change failed: {exc}")) from None

    github = github or GitHubService(access_token)
    opened: list[PullRequestInfo] = []
    # Body of each pull request without its related section; one already open keeps what people wrote in it.
    bodies: dict[str, str] = {}
    current: dict[str, str] = {}
    try:
        for repo_change in changes:
            spec = workspace.repos[repo_change.repo]
            pull = await github.get_pull_request_for_branch(spec.github_owner, spec.github_name, repo_change.branch)
            if pull and pull["state"] == "open":
                current[repo_change.repo] = pull["body"]
                bodies[repo_change.repo] = _without_related_section(pull["body"])
            else:
                bodies[repo_change.repo] = current[repo_change.repo] = body or _default_body(change, repo_change)
                pull = await github.create_pull_request(
                    spec.github_owner, spec.github_name, repo_change.branch, repo_change.base,
                    title, bodies[repo_change.repo], draft=True,
                )
            opened.append(PullRequestInfo(repo_change.repo, int(pull["number"]), pull["html_url"],
                                          repo_change.branch, repo_change.base))
        if len(opened) > 1:
            for info in opened:
                spec = workspace.repos[info.repo]
                new_body = bodies[info.repo] + _related_section(opened, info)
                if new_body != current[info.repo]:
                    await github.update_pull_request(spec.github_owner, spec.github_name, info.number, body=new_body)
    except httpx.HTTPError as exc:
        already = "".join(f" Already opened: {info.url}." for info in opened)
        raise PullRequestError(safe(f"GitHub refused the pull request: {_github_error(exc)}.{already}")) from None
    return opened


def _changes_in_merge_order(workspace: LabWorkspace, change: LabChange) -> list[RepoChange]:
    changes = [repo_change for repo_change in workspace.repo_changes(change) if repo_change.files]
    return sorted(changes, key=lambda repo_change: REPO_ORDER.index(repo_change.repo))


def _refuse_unpushable(workspace: LabWorkspace, changes: Sequence[RepoChange]) -> None:
    """Raise ``PullRequestError`` for a change that must not leave the server."""
    for repo_change in changes:
        if not repo_change.branch.startswith(LAB_BRANCH_PREFIX):
            raise PullRequestError(
                f"The {repo_change.repo} branch {repo_change.branch!r} is not a lab branch "
                f"({LAB_BRANCH_PREFIX}...); BESSER Lab pushes only its own branches.")
        # Renames split in two: moving a workflow out of the folder deletes it.
        workflows = sorted(p for p in workspace.touched_paths(repo_change.repo) if p.startswith(_WORKFLOWS_DIR))
        if workflows:
            raise PullRequestError(
                f"The change edits GitHub workflows in {repo_change.repo} ({', '.join(workflows)}). "
                "The sign-in token cannot push workflow files: undo those edits, or make them by hand.")
        secrets = _secret_files(workspace.repos[repo_change.repo].path,
                                [f.path for f in repo_change.files if f.status != "deleted"])
        if secrets:
            raise PullRequestError(
                f"The change adds secrets in {repo_change.repo} ({', '.join(secrets)}). "
                "Remove them before opening a pull request.")


def _secret_files(repo_path: str, paths: Sequence[str]) -> list[str]:
    """The ``paths`` the GitHub push boundary withholds as secret ``.env`` files.

    ``scrub_secret_files`` is that boundary's detection; it rewrites the files it
    scans, so it runs on a copy of the changed ``.env*`` files.
    """
    env_paths = [path for path in paths if os.path.basename(path).lower().startswith(".env")]
    if not env_paths:
        return []
    scratch = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    try:
        for path in env_paths:
            source = os.path.join(repo_path, path)
            if os.path.isfile(source) and not os.path.islink(source):
                target = os.path.join(scratch, path)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copyfile(source, target)
        return sorted(scrub_secret_files(scratch).removed_files)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _point_submodule_at_editor_branch(workspace: LabWorkspace, changes: Sequence[RepoChange], login: str) -> None:
    """Commit, on BESSER's lab branch, the editor submodule at the head of the editor's lab branch."""
    editor_branch = next(c.branch for c in changes if c.repo == REPO_FRONTEND)
    editor_sha = workspace.run_git(REPO_FRONTEND, "rev-parse", f"refs/heads/{editor_branch}").strip()
    recorded = workspace.run_git(REPO_BESSER, "rev-parse", f"HEAD:{FRONTEND_SUBMODULE_PATH}").strip()
    if recorded == editor_sha:
        return
    workspace.run_git(REPO_BESSER, "update-index", "--cacheinfo", f"160000,{editor_sha},{FRONTEND_SUBMODULE_PATH}")
    email = f"{login}@users.noreply.github.com"
    workspace.run_git(
        REPO_BESSER, "commit", "-m", f"chore: point the editor submodule at {editor_sha[:12]}",
        env_extra={"GIT_AUTHOR_NAME": login, "GIT_AUTHOR_EMAIL": email,
                   "GIT_COMMITTER_NAME": login, "GIT_COMMITTER_EMAIL": email},
    )


def _push(workspace: LabWorkspace, change: LabChange, repo_change: RepoChange, access_token: str) -> None:
    """Push the lab branch, first merging commits added to it on GitHub.

    The lab agent can write the working copy's ``.git/config`` (a proxy, TLS settings, URL
    rewrites), so every command carrying the token runs in a scratch bare repository with
    no config of its own, which reads the working copy's objects through ``alternates``.
    """
    repo = repo_change.repo
    spec = workspace.repos[repo]
    url = f"https://github.com/{spec.github_owner}/{spec.github_name}.git"
    ref = f"refs/heads/{repo_change.branch}"
    scratch = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    try:
        isolated = {"GIT_DIR": scratch}
        authenticated = {**isolated, **push_environment(access_token)}
        workspace.run_git(repo, "init", "-q", "--bare", env_extra=isolated)
        objects = workspace.run_git(repo, "rev-parse", "--path-format=absolute", "--git-path", "objects").strip()
        with open(os.path.join(scratch, "objects", "info", "alternates"), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(objects + "\n")
        if workspace.run_git(repo, "ls-remote", url, ref, env_extra=authenticated, timeout=_PUSH_TIMEOUT_SECONDS):
            workspace.run_git(repo, "fetch", "-q", "--no-tags", url, f"+{ref}:{_GITHUB_HEAD}",
                              env_extra=authenticated, timeout=_PUSH_TIMEOUT_SECONDS)
            workspace.merge_github_work(change, repo, scratch, _GITHUB_HEAD)
        head = workspace.run_git(repo, "rev-parse", ref).strip()
        workspace.run_git(repo, "update-ref", ref, head, env_extra=isolated)
        workspace.run_git(repo, "push", url, f"{ref}:{ref}", env_extra=authenticated, timeout=_PUSH_TIMEOUT_SECONDS)
        workspace.record_pushed(repo, repo_change.branch, head)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _default_body(change: LabChange, repo_change: RepoChange) -> str:
    lines = [f"{change.title}", "", f"Opened as a draft from BESSER Lab ({change.kind.value} change).", "",
             "Changed files:"]
    lines += [f"- `{f.path}` ({f.status})" for f in repo_change.files[:_MAX_LISTED_FILES]]
    if len(repo_change.files) > _MAX_LISTED_FILES:
        lines.append(f"- and {len(repo_change.files) - _MAX_LISTED_FILES} more")
    return "\n".join(lines)


def _related_section(opened: Sequence[PullRequestInfo], this: PullRequestInfo) -> str:
    """The pull requests of the same change, in the order to merge them."""
    lines = ["", "", _RELATED_HEADING, "", "Merge in this order (the editor first):", ""]
    for position, info in enumerate(opened, start=1):
        marker = " (this pull request)" if info == this else ""
        lines.append(f"{position}. {info.repo}: {info.url}{marker}")
    return "\n".join(lines)


def _without_related_section(body: str) -> str:
    """``body`` without the related section the lab appended to its end."""
    return re.split(rf"\s*^{re.escape(_RELATED_HEADING)}\s*$", body, maxsplit=1, flags=re.MULTILINE)[0]


def _github_error(exc: httpx.HTTPError) -> str:
    """GitHub's status and message, or the transport error."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            message = exc.response.json().get("message", "")
        except ValueError:
            message = ""
        return f"HTTP {exc.response.status_code} for {exc.request.url.path}" + (f" ({message})" if message else "")
    return f"{type(exc).__name__}: {exc}"
