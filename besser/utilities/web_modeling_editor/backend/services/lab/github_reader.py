"""The lab agent's GitHub reads, with the signed-in user's token (implements ``lab.github_tools.GitHubReader``).

The agent's tools run in a worker thread; :class:`LabGitHubReader` runs each GitHub request on the
backend's event loop and waits for it. Issues and search read the canonical BESSER-PEARL repositories
(``contracts.DEFAULT_UPSTREAMS``), where the team's issues are; pull requests are read from the
repositories the lab pushes to (a fork when the settings name one). Every answer passes through
:func:`~besser.utilities.secret_redaction.redact_text` with the token, so the token never reaches the
agent, and neither does a credential someone pasted on GitHub.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Coroutine, Mapping, Optional, Sequence

import httpx

from besser.spec_driven_agent.lab.contracts import DEFAULT_UPSTREAMS, RepoSpec
from besser.spec_driven_agent.lab.github_tools import GitHubReadError, diff_sections, under_path
from besser.spec_driven_agent.lab.workspace import parse_upstream
from besser.utilities.secret_redaction import redact_text
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import (
    GitHubGraphQLError,
    GitHubService,
)

logger = logging.getLogger(__name__)

#: How much of a pull request's diff is fetched: a path filter may need files far past what the
#: agent is shown (``MAX_DIFF_CHARS``); 1 MB bounds the memory and time of one read.
DIFF_FETCH_MAX_BYTES = 1_000_000

#: GitHub's largest page of a GraphQL connection is 100; these keep one answer readable.
_MAX_FILES = 100
_MAX_REVIEWS = 20
_MAX_THREADS = 50
_MAX_THREAD_COMMENTS = 20
_MAX_CHECKS = 50
_MAX_LABELS = 20

_ISSUE_QUERY = """
query LabReadIssue($owner: String!, $name: String!, $number: Int!, $comments: Int!) {
  repository(owner: $owner, name: $name) {
    issueOrPullRequest(number: $number) {
      __typename
      ... on Issue {
        number title url state createdAt body authorAssociation
        author { login }
        labels(first: %(labels)d) { nodes { name } }
        comments(last: $comments) { totalCount nodes { body createdAt authorAssociation author { login } } }
      }
      ... on PullRequest { number title url state }
    }
  }
}
""" % {"labels": _MAX_LABELS}

_PULL_REQUEST_QUERY = """
query LabReadPullRequest($owner: String!, $name: String!, $number: Int!,
                         $files: Boolean!, $reviews: Boolean!, $checks: Boolean!) {
  repository(owner: $owner, name: $name) {
    issueOrPullRequest(number: $number) {
      __typename
      ... on Issue { number title url state }
      ... on PullRequest {
        number title url state isDraft body authorAssociation baseRefName headRefName
        author { login }
        files(first: %(files)d) @include(if: $files) { totalCount nodes { path additions deletions changeType } }
        reviews(last: %(reviews)d) @include(if: $reviews) {
          nodes { state body submittedAt authorAssociation author { login } }
        }
        reviewThreads(first: %(threads)d) @include(if: $reviews) {
          nodes {
            path line isResolved isOutdated
            comments(first: %(thread_comments)d) { nodes { body createdAt authorAssociation author { login } } }
          }
        }
        commits(last: 1) @include(if: $checks) {
          nodes { commit { statusCheckRollup { contexts(first: %(checks)d) { nodes {
            __typename
            ... on CheckRun { name status conclusion detailsUrl }
            ... on StatusContext { context state targetUrl }
          } } } } }
        }
      }
    }
  }
}
""" % {"files": _MAX_FILES, "reviews": _MAX_REVIEWS, "threads": _MAX_THREADS,
       "thread_comments": _MAX_THREAD_COMMENTS, "checks": _MAX_CHECKS}

_SEARCH_QUERY = """
query LabSearch($query: String!, $limit: Int!) {
  search(query: $query, type: ISSUE, first: $limit) {
    nodes {
      __typename
      ... on Issue { number title state url updatedAt author { login } repository { name owner { login } } }
      ... on PullRequest { number title state url updatedAt author { login } repository { name owner { login } } }
    }
  }
}
"""

_SEARCH_KIND = {"issue": "is:issue", "pr": "is:pr", "any": ""}
_SEARCH_STATE = {"open": "is:open", "closed": "is:closed", "all": ""}


def _login(node: Optional[dict]) -> str:
    """The author's login; GitHub gives none for a deleted account."""
    return ((node or {}).get("author") or {}).get("login") or "ghost"


def _date(value: Optional[str]) -> str:
    return (value or "")[:10]


def _comment(node: dict) -> dict:
    return {"author": _login(node), "association": node.get("authorAssociation") or "NONE",
            "date": _date(node.get("createdAt")), "body": node.get("body") or ""}


def _nodes(connection: Optional[dict]) -> list[dict]:
    return [node for node in (connection or {}).get("nodes") or [] if node]


class LabGitHubReader:
    """:class:`~besser.spec_driven_agent.lab.github_tools.GitHubReader` through ``github`` (the user's token).

    ``loop`` is the backend's running event loop; the methods are called from other threads (never from
    the loop's own, which would wait on itself) and give up after ``timeout_seconds``. ``login`` names the
    signed-in account in "not visible to" answers.
    """

    def __init__(self, github: GitHubService, repos: Mapping[str, RepoSpec], loop: asyncio.AbstractEventLoop,
                 *, timeout_seconds: float = 30.0, login: str = ""):
        self._github = github
        self._repos = dict(repos)
        self._loop = loop
        self._timeout = timeout_seconds
        self._login = login

    # --- GitHubReader ---------------------------------------------------

    def issue(self, repo: str, number: int, comments: int) -> dict:
        owner, name = self._canonical(repo)
        # GitHub's connections take 1..100; 0 comments still reads the count.
        data = self._graphql(_ISSUE_QUERY, {"owner": owner, "name": name, "number": number,
                                            "comments": max(comments, 1)})
        node = self._item(data)
        result = self._short(repo, node)
        if node["__typename"] == "PullRequest":
            return self._redact(result)
        thread = node.get("comments") or {}
        result.update({
            "author": _login(node), "association": node.get("authorAssociation") or "NONE",
            "created_at": _date(node.get("createdAt")), "body": node.get("body") or "",
            "labels": [label["name"] for label in _nodes(node.get("labels"))],
            "comments": [_comment(c) for c in _nodes(thread)][-comments:] if comments else [],
            "total_comments": int(thread.get("totalCount") or 0),
        })
        return self._redact(result)

    def pull_request(self, repo: str, number: int, parts: Sequence[str], path: str | None) -> dict:
        spec = self._spec(repo)
        owner, name = spec.github_owner, spec.github_name
        data = self._graphql(_PULL_REQUEST_QUERY, {
            "owner": owner, "name": name, "number": number,
            "files": "files" in parts, "reviews": "reviews" in parts, "checks": "checks" in parts,
        })
        node = self._item(data)
        result = self._short(repo, node)
        if node["__typename"] == "Issue":
            return self._redact(result)
        result.update({
            "is_draft": bool(node.get("isDraft")), "author": _login(node),
            "association": node.get("authorAssociation") or "NONE", "body": node.get("body") or "",
            "base": node.get("baseRefName") or "", "head": node.get("headRefName") or "",
        })
        if "files" in parts:
            files = node.get("files") or {}
            result["files"] = [{"path": f.get("path", ""), "additions": f.get("additions", 0),
                                "deletions": f.get("deletions", 0), "change_type": f.get("changeType", "")}
                               for f in _nodes(files)]
            result["total_files"] = int(files.get("totalCount") or len(result["files"]))
        if "reviews" in parts:
            result["reviews"] = [{"author": _login(r), "association": r.get("authorAssociation") or "NONE",
                                  "state": r.get("state") or "", "date": _date(r.get("submittedAt")),
                                  "body": r.get("body") or ""} for r in _nodes(node.get("reviews"))]
            result["threads"] = [{"path": t.get("path") or "", "line": t.get("line"),
                                  "resolved": bool(t.get("isResolved")), "outdated": bool(t.get("isOutdated")),
                                  "comments": [_comment(c) for c in _nodes(t.get("comments"))]}
                                 for t in _nodes(node.get("reviewThreads"))]
        if "checks" in parts:
            result["checks"] = _checks(node)
        if "diff" in parts:
            result["diff"], result["diff_cut"] = self._diff(owner, name, number, path)
        return self._redact(result)

    def search(self, query: str, repo: str | None, kind: str, state: str, limit: int) -> list[dict]:
        keys = [repo] if repo else list(self._repos)
        by_name = {}
        qualifiers = []
        for key in keys:
            owner, name = self._canonical(key)
            by_name[f"{owner}/{name}".lower()] = key
            qualifiers.append(f"repo:{owner}/{name}")
        qualifiers += [q for q in (_SEARCH_KIND[kind], _SEARCH_STATE[state]) if q]
        data = self._graphql(_SEARCH_QUERY, {"query": " ".join([query, *qualifiers]), "limit": limit})
        results = []
        for node in _nodes(data.get("search")):
            repository = node.get("repository") or {}
            key = by_name.get(f"{(repository.get('owner') or {}).get('login')}/{repository.get('name')}".lower())
            if key is None:  # the qualifiers keep GitHub to these repositories; never show another's
                continue
            results.append({"repo": key, "number": node.get("number"),
                            "kind": "pull_request" if node.get("__typename") == "PullRequest" else "issue",
                            "title": node.get("title") or "", "state": (node.get("state") or "").lower(),
                            "url": node.get("url") or "", "author": _login(node),
                            "updated_at": _date(node.get("updatedAt"))})
        return self._redact(results)

    # --- helpers --------------------------------------------------------

    def _spec(self, repo: str) -> RepoSpec:
        if repo not in self._repos:
            raise GitHubReadError("failed", detail=f"{repo} is not one of the lab's repositories")
        return self._repos[repo]

    def _canonical(self, repo: str) -> tuple[str, str]:
        """The BESSER-PEARL repository of ``repo``: where issues are filed and searched."""
        self._spec(repo)
        owner, name, _ = parse_upstream(DEFAULT_UPSTREAMS[repo])
        return owner, name

    @staticmethod
    def _short(repo: str, node: dict) -> dict:
        return {"repo": repo, "number": node.get("number"),
                "kind": "pull_request" if node["__typename"] == "PullRequest" else "issue",
                "title": node.get("title") or "", "state": (node.get("state") or "").lower(),
                "url": node.get("url") or ""}

    def _item(self, data: dict) -> dict:
        node = ((data.get("repository") or {}).get("issueOrPullRequest"))
        if not node:
            raise GitHubReadError("not_found", login=self._login)
        return node

    def _diff(self, owner: str, name: str, number: int, path: Optional[str]) -> tuple[str, bool]:
        """The diff, only the files under ``path`` when given, and whether GitHub's was longer than fetched."""
        diff = self._run(self._github.get_pull_request_diff(owner, name, number, max_bytes=DIFF_FETCH_MAX_BYTES + 1))
        cut = len(diff.encode("utf-8")) > DIFF_FETCH_MAX_BYTES
        if path:
            diff = "".join(text for file_path, text in diff_sections(diff) if under_path(file_path, path))
        return diff, cut

    def _graphql(self, query: str, variables: dict) -> dict:
        return self._run(self._github.graphql(query, variables))

    def _run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        """Run ``coroutine`` on the backend's loop and wait for it; every failure becomes a ``GitHubReadError``."""
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(self._timeout)
        except TimeoutError:
            future.cancel()
            error = GitHubReadError("failed", detail=f"GitHub did not answer within {self._timeout:g} s")
        except GitHubReadError as exc:
            error = exc
        except Exception as exc:  # every GitHub or network failure reaches the agent as a sentence
            error = self._read_error(exc)
        level = logging.INFO if error.reason == "not_found" else logging.WARNING
        logger.log(level, "Lab GitHub read failed (%s): %s", error.reason, error)
        raise error

    def _read_error(self, exc: Exception) -> GitHubReadError:
        if isinstance(exc, httpx.HTTPStatusError):
            response = exc.response
            status = response.status_code
            if status == 401:
                return GitHubReadError("signed_out")
            if status in (403, 429) and _rate_limited(response):
                return GitHubReadError("rate_limited", reset_at=_reset_at(response))
            if status == 404:
                return GitHubReadError("not_found", login=self._login)
            return GitHubReadError("failed", detail=self._clean(f"GitHub answered HTTP {status}: "
                                                                 f"{_message(response)}"))
        if isinstance(exc, GitHubGraphQLError):
            types = {str(error.get("type", "")) for error in exc.errors}
            if "RATE_LIMITED" in types:
                return GitHubReadError("rate_limited")
            if types & {"NOT_FOUND", "FORBIDDEN"}:
                return GitHubReadError("not_found", login=self._login)
            return GitHubReadError("failed", detail=self._clean(str(exc)))
        if isinstance(exc, httpx.HTTPError):
            return GitHubReadError("failed", detail=f"could not reach GitHub ({type(exc).__name__})")
        logger.warning("Lab GitHub read failed unexpectedly: %s", type(exc).__name__)
        return GitHubReadError("failed", detail=type(exc).__name__)

    def _clean(self, text: str) -> str:
        return redact_text(text, env_style=False, secrets=[self._github.access_token])[0]

    def _redact(self, value: Any) -> Any:
        """``value`` with the token and any credential-shaped text replaced, in every string."""
        if isinstance(value, str):
            return self._clean(value)
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        return value


def _checks(node: dict) -> list[dict]:
    """The head commit's check runs and commit statuses, as ``{"name", "status", "conclusion", "url"}``."""
    commits = _nodes(node.get("commits"))
    rollup = ((commits[-1].get("commit") or {}).get("statusCheckRollup") or {}) if commits else {}
    checks = []
    for context in _nodes(rollup.get("contexts")):
        if context.get("__typename") == "StatusContext":
            state = (context.get("state") or "").lower()
            checks.append({"name": context.get("context") or "", "status": "pending" if state == "pending"
                           else "completed", "conclusion": "" if state == "pending" else state,
                           "url": context.get("targetUrl") or ""})
        else:
            checks.append({"name": context.get("name") or "", "status": (context.get("status") or "").lower(),
                           "conclusion": (context.get("conclusion") or "").lower(),
                           "url": context.get("detailsUrl") or ""})
    return checks


def _message(response: httpx.Response) -> str:
    """GitHub's ``message`` from an error answer (a streamed answer may be unread: then none)."""
    try:
        return str(response.json().get("message", ""))
    except (ValueError, httpx.ResponseNotRead, AttributeError):
        return ""


def _rate_limited(response: httpx.Response) -> bool:
    return (response.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in response.headers
            or "rate limit" in _message(response).lower())


def _reset_at(response: httpx.Response) -> Optional[float]:
    """When the limit lifts, in seconds since the epoch, if GitHub said."""
    reset = response.headers.get("x-ratelimit-reset")
    if reset and reset.isdigit():
        return float(reset)
    retry_after = response.headers.get("retry-after")
    if retry_after and retry_after.isdigit():
        return time.time() + int(retry_after)
    return None
