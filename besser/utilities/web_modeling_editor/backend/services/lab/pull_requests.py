"""Open and update the draft pull requests of a BESSER Lab change.

Each repository the change touched gets its lab branch pushed with the clicking user's
GitHub token and a draft pull request against the upstream the change started from
(``change.bases[repo]``: a fork when the settings named one then), the editor first:
BESSER's branch points its editor submodule at the pushed editor commit. Commits are
authored by the login whose request made them.

The token reaches git only through the environment, never argv, a remote URL or git
config, and only in a scratch repository (:func:`_github_remote`): never with the working
copy's own config, which the lab agent can write. Commits added to a lab branch on GitHub
are merged before the push. The body of every pull request carries a lab-owned block
between :data:`LAB_BLOCK_BEGIN` and :data:`LAB_BLOCK_END`, rewritten at every push; what
reviewers wrote outside it is kept. A failure the user can act on (sign-in, access, branch
protection, network) is told in plain words; git's or GitHub's own message goes to the
server log only.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

import httpx

from besser.spec_driven_agent.lab.contracts import (
    DEFAULT_UPSTREAMS,
    FRONTEND_SUBMODULE_PATH,
    LAB_BRANCH_PREFIX,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_ORDER,
    ChangeKind,
    IssueRef,
    LabChange,
    RepoChange,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError, parse_upstream
from besser.utilities.secret_redaction import redact_text, scrub_secret_files
from besser.utilities.web_modeling_editor.backend.constants.constants import TEMP_DIR_PREFIX
from besser.utilities.web_modeling_editor.backend.models.lab import LabChangeView, LabPlan
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import GitHubService

logger = logging.getLogger(__name__)

#: The lab-owned part of a pull request body, rewritten at every push.
LAB_BLOCK_BEGIN = "<!-- besser-lab:begin -->"
LAB_BLOCK_END = "<!-- besser-lab:end -->"

# The sign-in token has no ``workflow`` scope, so GitHub refuses a push that changes these.
_WORKFLOWS_DIR = ".github/workflows/"
_PUSH_TIMEOUT_SECONDS = 300
# The scratch repository's copy of the lab branch as GitHub has it.
_GITHUB_HEAD = "refs/lab/github"
# Phase 1 appended this section to the end of a body; it moves inside the block.
_PHASE_1_RELATED_HEADING = "## Related pull requests"
_LAB_BLOCK_RE = re.compile(re.escape(LAB_BLOCK_BEGIN) + r".*?" + re.escape(LAB_BLOCK_END), re.DOTALL)
_FULL_SUITES = "The repositories' full test suites did not run in the lab."

# What to tell the user for each failure class; ``{repo}`` is owner/name, ``{branch}`` the lab branch.
_SIGN_IN = "GitHub did not accept your sign-in. Sign in to GitHub again, then try again."
_NO_ACCESS = ("Your GitHub account cannot push to {repo}. Ask a maintainer of {repo} for write access, then open "
              "the pull request again.")
_PROTECTED = ("GitHub's branch protection rules of {repo} refused the branch {branch}. Ask a maintainer of {repo} to "
              "allow pushes to lab branches.")
_NETWORK = ("The lab server could not reach GitHub. Try again in a few minutes; if it keeps failing, ask the lab's "
            "administrator to check the server's connection to github.com.")
# git's messages for those classes, checked in this order (a 403 also says "unable to access").
_GIT_FAILURES = (
    (re.compile(r"could not read Username|Authentication failed|Invalid username or password|"
                r"returned error: 401", re.IGNORECASE), _SIGN_IN),
    (re.compile(r"protected branch|GH006|GH013|repository rule violations", re.IGNORECASE), _PROTECTED),
    (re.compile(r"Permission to \S+ denied|returned error: 403|Repository not found|"
                r"write access to repository not granted", re.IGNORECASE), _NO_ACCESS),
    (re.compile(r"Could not resolve (host|proxy)|Failed to connect|Couldn't connect|"
                r"Connection (timed out|refused|reset)|Operation timed out|timed out after|SSL certificate problem|"
                r"SSL_ERROR|schannel|gnutls_handshake", re.IGNORECASE), _NETWORK),
)


@dataclass(frozen=True)
class PullRequestInfo:
    """One pull request opened for a change."""

    repo: str
    number: int
    url: str
    branch: str
    base: str


class PullRequestError(Exception):
    """A pull request action (open, push, status, update from base) failed; the message is safe to show."""


class PullRequestInputError(PullRequestError):
    """The request names something the change does not have (an issue it did not start from): the request is
    wrong, not the change's state."""


class PullRequestSignInError(PullRequestError):
    """GitHub refused the user's sign-in (the token): only signing in again helps."""


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


def upstream_of(workspace: LabWorkspace, change: LabChange, repo: str) -> tuple[str, str, str]:
    """``(owner, name, base branch)`` the change's pull request in ``repo`` goes to: where the change started.

    A change saved before it recorded its bases falls back to the settings' repository, with a warning.
    """
    base = change.bases.get(repo)
    if base is not None:
        owner, _, name = base.upstream.partition("/")
        return owner, name, base.branch
    spec = workspace.base_repos[repo]
    logger.warning("BESSER Lab: change %s has no recorded base in %s; its pull request goes to %s/%s",
                   change.id, repo, spec.github_owner, spec.github_name)
    return spec.github_owner, spec.github_name, spec.base_branch


def github_failure(exc: httpx.HTTPError, what: str, *, secrets: Sequence[str] = ()) -> PullRequestError:
    """The error to raise for a GitHub API failure while doing ``what`` (e.g. "the pull request").

    A refused sign-in or an unreachable GitHub becomes plain advice (GitHub's words are logged);
    any other failure names GitHub's status and message. ``secrets`` are redacted from both.
    """
    detail = redact_text(_github_error(exc), env_style=False, secrets=secrets)[0]
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 401:
        logger.warning("BESSER Lab: GitHub refused the sign-in for %s: %s", what, detail)
        return PullRequestSignInError(_SIGN_IN)
    if isinstance(exc, httpx.TransportError):
        logger.warning("BESSER Lab: GitHub could not be reached for %s: %s", what, detail)
        return PullRequestError(_NETWORK)
    return PullRequestError(f"GitHub refused {what}: {detail}.")


async def open_pull_requests(
    workspace: LabWorkspace,
    change: LabChange,
    *,
    login: str,
    access_token: str,
    github: Optional[GitHubService] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
    closes_issues: Optional[Sequence[IssueRef]] = None,
    plan: Optional[LabPlan] = None,
    last: Optional[LabChangeView] = None,
    model: str = "",
    requested_by: Sequence[str] = (),
) -> tuple[list[PullRequestInfo], str]:
    """Push the change's lab branches and open, or update, a draft pull request in each repository it touched.

    Args:
        workspace: The lab's workspace.
        change: An open change; its ``pull_requests`` and the ``closes`` flags of its ``issues``
            are updated and saved.
        login: The clicking user's GitHub login: the author of the commits made here.
        access_token: The clicking user's GitHub token; it pushes and opens the pull requests.
        github: The GitHub client; one is made from ``access_token`` when omitted.
        title: The pull requests' title; defaults to ``default_title(change)``.
        body: The text of a new pull request outside the lab block; a default when omitted.
        closes_issues: The change's issues the pull requests close; None keeps the stored ticks.
        plan: The change's plan, for the lab block.
        last: The change's last run (checks, report), for the lab block and the checks note.
        model: The model that built the change, named in the lab block.
        requested_by: Logins that asked for the change's steps, first request first.

    Returns:
        The pull requests in merge order (editor, BESSER, modeling agent), and the note on the
        checks the change passed in the lab.

    Raises:
        PullRequestError: Nothing to open, an issue that is not the change's, a change that may
            not be pushed (refused before any push), or a git or GitHub failure. The message never
            holds the token.
    """
    def safe(text: str) -> str:
        return redact_text(text, env_style=False, secrets=[access_token])[0]

    closes = _closing_issues(change, closes_issues)
    title = title or default_title(change)
    try:
        changes = await asyncio.to_thread(_changes_in_merge_order, workspace, change)
        if not changes:
            raise PullRequestError("This change has no modifications to open a pull request for.")
        await asyncio.to_thread(_refuse_unpushable, workspace, change, changes)
        await asyncio.to_thread(workspace.commit_all, change, title, login=login)
        repos_with_changes = {repo_change.repo for repo_change in changes}
        for repo_change in changes:
            if repo_change.repo == REPO_BESSER and REPO_FRONTEND in repos_with_changes:
                await asyncio.to_thread(_point_submodule_at_editor_branch, workspace, change, login)
            await asyncio.to_thread(_push, workspace, change, repo_change.repo, access_token)
    except LabWorkspaceError as exc:
        raise PullRequestError(safe(f"Pushing the change failed: {exc}")) from None

    for issue in change.issues:
        issue["closes"] = (issue.get("repo"), issue.get("number")) in closes
    github = github or GitHubService(access_token)
    opened: list[PullRequestInfo] = []
    # Each pull request's body as GitHub has it now, to skip an update that changes nothing.
    current: dict[str, str] = {}

    def block(this: PullRequestInfo) -> str:
        return compose_lab_block(change, plan, last, opened, model, requested_by, this=this)

    try:
        for repo_change in changes:
            repo, branch = repo_change.repo, repo_change.branch
            owner, name, base = upstream_of(workspace, change, repo)
            pull = await github.get_pull_request_for_branch(owner, name, branch)
            if pull and pull["state"] == "open":
                current[repo] = pull["body"]
            else:
                # Its block lists the pull requests known so far; it is rewritten once all are open.
                current[repo] = with_lab_block(body or _default_body(change),
                                               block(PullRequestInfo(repo, 0, "", branch, base)))
                pull = await github.create_pull_request(owner, name, branch, base, title, current[repo], draft=True)
            opened.append(PullRequestInfo(repo, int(pull["number"]), pull["html_url"], branch, base))
            change.pull_requests[repo] = int(pull["number"])
        for info in opened:
            new_body = with_lab_block(current[info.repo], block(info))
            if new_body != current[info.repo]:
                owner, name, _ = upstream_of(workspace, change, info.repo)
                await github.update_pull_request(owner, name, info.number, body=new_body)
    except httpx.HTTPError as exc:
        error = github_failure(exc, "the pull request", secrets=[access_token])
        already = "".join(f" Already opened: {info.url}." for info in opened)
        raise type(error)(str(error) + already) from None
    finally:
        await asyncio.to_thread(workspace.save_change, change)
    return opened, checks_note(last)


def sync_from_github(workspace: LabWorkspace, change: LabChange, repo: str, *, access_token: str) -> None:
    """Merge into the change's ``repo`` branch what GitHub's copy of it holds (blocking: run it in a thread).

    Used after GitHub merged the base into the pull request's branch. Nothing happens when the
    branch is not on GitHub or the working copy already has it.

    Raises:
        PullRequestError: GitHub refused or could not be reached (plain advice).
        LabWorkspaceError: The merge failed (a conflict, an unrelated branch).
    """
    ref = f"refs/heads/{change.branches[repo]}"
    with _github_remote(workspace, change, repo, access_token) as remote:
        if remote.run("ls-remote", remote.url, ref):
            remote.run("fetch", "-q", "--no-tags", remote.url, f"+{ref}:{_GITHUB_HEAD}")
            workspace.merge_github_work(change, repo, remote.scratch, _GITHUB_HEAD)


def compose_lab_block(change: LabChange, plan: Optional[LabPlan], last: Optional[LabChangeView],
                      opened: Sequence[PullRequestInfo], model: str, requested_by: Sequence[str], *,
                      this: Optional[PullRequestInfo] = None) -> str:
    """The lab-owned part of a pull request body, markers included.

    It holds the agent's account of the last build (its report, or its closing text when it
    gave none), the plan, the checks run in the lab, the issues (``Closes`` for the ticked
    ones), the related pull requests in merge order (``this`` one marked) and who asked and
    which model built it. Never the cost: the repositories are public.
    """
    lines = [LAB_BLOCK_BEGIN, "## BESSER Lab", ""]
    report = last.report if last is not None else None
    if report is not None:
        lines += [report.summary.strip() or "The lab agent's report has no summary.", ""]
        lines += _list_section("Deviations from the plan", report.deviations, "None reported.")
        lines += _list_section("Not done", report.notDone, "Nothing reported.")
    elif last is not None and last.summary.strip():
        lines += ["The lab agent gave no structured account. Its closing text:", "", last.summary.strip(), ""]
    else:
        lines += ["The lab agent gave no account of this change.", ""]
    if plan is not None:
        lines += ["### Plan", "", plan.summary.strip(), ""]
        lines += [f"{position}. {step}" for position, step in enumerate(plan.steps, start=1)]
        lines += [""] if plan.steps else []
    lines += ["### Checks in the lab", "", *_checks_lines(last), ""]
    lines += _issues_section(change)
    if len(opened) > 1:
        lines += ["### Related pull requests", "", "Merge in this order (the editor first):", ""]
        for position, info in enumerate(opened, start=1):
            marker = " (this pull request)" if this is not None and info.repo == this.repo else ""
            lines.append(f"{position}. {info.repo}: {info.url}{marker}")
        lines.append("")
    who = ", ".join(f"@{login}" for login in requested_by) or f"@{change.owner}"
    lines += [f"Requested in BESSER Lab by {who}" + (f", built by {model}." if model else "."), LAB_BLOCK_END]
    return "\n".join(lines)


def with_lab_block(body: str, block: str) -> str:
    """``body`` with its lab block replaced by ``block``; a body without one gets it appended.

    Text outside the block (what reviewers wrote) is kept. A phase 1 body loses the related
    section the lab appended to its end, which now lives in the block.
    """
    if _LAB_BLOCK_RE.search(body):
        return _LAB_BLOCK_RE.sub(lambda _match: block, body, count=1)
    if LAB_BLOCK_BEGIN in body:
        logger.warning("BESSER Lab: a pull request body has the lab block's start marker but not its end; the "
                       "lab block is rewritten from that marker to the end of the body")
        return body[:body.index(LAB_BLOCK_BEGIN)] + block
    kept = re.split(rf"\s*^{re.escape(_PHASE_1_RELATED_HEADING)}\s*$", body, maxsplit=1, flags=re.MULTILINE)[0]
    kept = kept.rstrip()
    return f"{kept}\n\n{block}" if kept else block


def checks_note(last: Optional[LabChangeView]) -> str:
    """What the user is told about the checks of a change pushed to its pull requests."""
    if last is None or not last.checks:
        return f"The lab ran no checks on this change. {_FULL_SUITES}"
    passed, failed, skipped = _check_groups(last)
    return (f"The lab's last build ran only the checks of the touched areas: {len(passed)} passed, {len(failed)} "
            f"failed, {len(skipped)} skipped. {_FULL_SUITES}")


def _check_groups(last: LabChangeView) -> tuple[list[str], list[str], list[str]]:
    """The titles of the last build's checks that passed, failed and were skipped."""
    passed = [check.title for check in last.checks if check.passed]
    failed = [check.title for check in last.checks if not check.passed and not check.skipped]
    skipped = [check.title for check in last.checks if check.skipped and not check.passed]
    return passed, failed, skipped


def _checks_lines(last: Optional[LabChangeView]) -> list[str]:
    if last is None or not last.checks:
        return [f"The lab ran no checks on this change. {_FULL_SUITES}"]
    lines = [f"- {label}: {', '.join(titles)}"
             for label, titles in zip(("Passed", "Failed", "Skipped"), _check_groups(last)) if titles]
    return [*lines, "", _FULL_SUITES]


def _list_section(heading: str, items: Sequence[str], empty: str) -> list[str]:
    lines = [f"### {heading}", ""]
    lines += [f"- {item}" for item in items] if items else [empty]
    return [*lines, ""]


def _issues_section(change: LabChange) -> list[str]:
    """``Closes`` for the issues the user ticked, ``Related to`` for the others, as ``owner/name#n``."""
    if not change.issues:
        return []
    lines = ["### Issues", ""]
    for issue in change.issues:
        # Issues are read from the canonical repositories, whatever repository the pull request goes to.
        owner, name, _ = parse_upstream(DEFAULT_UPSTREAMS[issue["repo"]])
        verb = "Closes" if issue.get("closes") else "Related to"
        lines.append(f"- {verb} {owner}/{name}#{issue['number']}")
    return [*lines, ""]


def _closing_issues(change: LabChange, closes_issues: Optional[Sequence[IssueRef]]) -> set[tuple[str, int]]:
    """The ``(repo, number)`` pairs to mark ``closes``; each must be one of the change's issues."""
    known = {(issue.get("repo"), issue.get("number")) for issue in change.issues}
    if closes_issues is None:
        return {(issue.get("repo"), issue.get("number")) for issue in change.issues if issue.get("closes")}
    wanted = {(ref.repo, ref.number) for ref in closes_issues}
    unknown = sorted(wanted - known)
    if unknown:
        names = ", ".join(f"{repo}#{number}" for repo, number in unknown)
        raise PullRequestInputError(f"{names} is not an issue of this change: its pull requests close only the issues "
                                    "the change started from.")
    return wanted


def _default_body(change: LabChange) -> str:
    return f"{change.title}\n\nOpened as a draft from BESSER Lab ({change.kind.value} change)."


def _changes_in_merge_order(workspace: LabWorkspace, change: LabChange) -> list[RepoChange]:
    changes = [repo_change for repo_change in workspace.repo_changes(change) if repo_change.files]
    return sorted(changes, key=lambda repo_change: REPO_ORDER.index(repo_change.repo))


def _refuse_unpushable(workspace: LabWorkspace, change: LabChange, changes: Sequence[RepoChange]) -> None:
    """Raise ``PullRequestError`` for a change that must not leave the server."""
    trees = workspace.change_repos(change)
    for repo_change in changes:
        if not repo_change.branch.startswith(LAB_BRANCH_PREFIX):
            raise PullRequestError(
                f"The {repo_change.repo} branch {repo_change.branch!r} is not a lab branch "
                f"({LAB_BRANCH_PREFIX}...); BESSER Lab pushes only its own branches.")
        # Renames split in two: moving a workflow out of the folder deletes it.
        workflows = sorted(path for path in workspace.touched_paths(change, repo_change.repo)
                           if path.startswith(_WORKFLOWS_DIR))
        if workflows:
            raise PullRequestError(
                f"The change edits GitHub workflows in {repo_change.repo} ({', '.join(workflows)}). "
                "The sign-in token cannot push workflow files: undo those edits, or make them by hand.")
        secrets = _secret_files(trees[repo_change.repo].path,
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


def _point_submodule_at_editor_branch(workspace: LabWorkspace, change: LabChange, login: str) -> None:
    """Commit, on BESSER's lab branch, the editor submodule at the head of the editor's lab branch."""
    editor_sha = workspace.run_git(REPO_FRONTEND, "rev-parse", f"refs/heads/{change.branches[REPO_FRONTEND]}").strip()
    recorded = workspace.run_git(REPO_BESSER, "rev-parse", f"HEAD:{FRONTEND_SUBMODULE_PATH}", change=change).strip()
    if recorded == editor_sha:
        return
    workspace.run_git(REPO_BESSER, "update-index", "--cacheinfo", f"160000,{editor_sha},{FRONTEND_SUBMODULE_PATH}",
                      change=change)
    email = f"{login}@users.noreply.github.com"
    workspace.run_git(
        REPO_BESSER, "commit", "-q", "-m", f"chore: point the editor submodule at {editor_sha[:12]}", change=change,
        env_extra={"GIT_AUTHOR_NAME": login, "GIT_AUTHOR_EMAIL": email,
                   "GIT_COMMITTER_NAME": login, "GIT_COMMITTER_EMAIL": email},
    )


@dataclass(frozen=True)
class _Remote:
    """A scratch bare repository that reads the change's objects and talks to GitHub with the token."""

    scratch: str
    url: str
    #: ``run(*args)``: ``git <args>`` authenticated, from the scratch repository
    run: Callable[..., str]


@contextlib.contextmanager
def _github_remote(workspace: LabWorkspace, change: LabChange, repo: str, access_token: str) -> Iterator[_Remote]:
    """A scratch bare repository for the change's upstream of ``repo``, removed afterwards.

    The lab agent can write the working copy's git config (a proxy, TLS settings, URL
    rewrites), so every command carrying the token runs in this repository, which has no
    config of its own and reads the working copy's objects through ``alternates``. A git
    failure the user can act on becomes a plain :class:`PullRequestError` (git's words are
    logged without the token).
    """
    owner, name, _ = upstream_of(workspace, change, repo)
    url = f"https://github.com/{owner}/{name}.git"
    scratch = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    try:
        isolated = {"GIT_DIR": scratch}
        authenticated = {**isolated, **push_environment(access_token)}

        def run(*args: str) -> str:
            try:
                return workspace.run_git(repo, *args, change=change, env_extra=authenticated,
                                         timeout=_PUSH_TIMEOUT_SECONDS)
            except LabWorkspaceError as exc:
                detail = redact_text(str(exc), env_style=False, secrets=[access_token])[0]
                advice = next((text for pattern, text in _GIT_FAILURES if pattern.search(detail)), None)
                if advice is None:
                    raise LabWorkspaceError(detail) from None
                logger.warning("BESSER Lab: git %s to %s failed: %s", args[0], url, detail)
                error = PullRequestSignInError if advice == _SIGN_IN else PullRequestError
                raise error(advice.format(repo=f"{owner}/{name}", branch=change.branches[repo])) from None

        workspace.run_git(repo, "init", "-q", "--bare", change=change, env_extra=isolated)
        objects = workspace.run_git(repo, "rev-parse", "--path-format=absolute", "--git-path", "objects",
                                    change=change).strip()
        with open(os.path.join(scratch, "objects", "info", "alternates"), "w", encoding="utf-8",
                  newline="\n") as handle:
            handle.write(objects + "\n")
        yield _Remote(scratch, url, run)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _push(workspace: LabWorkspace, change: LabChange, repo: str, access_token: str) -> None:
    """Push the change's ``repo`` branch, first merging commits added to it on GitHub."""
    branch = change.branches[repo]
    ref = f"refs/heads/{branch}"
    with _github_remote(workspace, change, repo, access_token) as remote:
        if remote.run("ls-remote", remote.url, ref):
            remote.run("fetch", "-q", "--no-tags", remote.url, f"+{ref}:{_GITHUB_HEAD}")
            workspace.merge_github_work(change, repo, remote.scratch, _GITHUB_HEAD)
        head = workspace.run_git(repo, "rev-parse", ref, change=change).strip()
        workspace.run_git(repo, "update-ref", ref, head, change=change, env_extra={"GIT_DIR": remote.scratch})
        remote.run("push", remote.url, f"{ref}:{ref}")
        workspace.record_pushed(repo, branch, head)


def _github_error(exc: httpx.HTTPError) -> str:
    """GitHub's status and message, or the transport error."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            message = exc.response.json().get("message", "")
        except ValueError:
            message = ""
        return f"HTTP {exc.response.status_code} for {exc.request.url.path}" + (f" ({message})" if message else "")
    return f"{type(exc).__name__}: {exc}"

