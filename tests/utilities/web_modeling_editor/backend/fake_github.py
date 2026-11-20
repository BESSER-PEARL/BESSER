"""A GitHub REST API stand-in backed by real git repositories.

Blobs, trees, commits, refs, merges, compares and tarballs are real git objects
made with the ``git`` CLI, so ``GitHubService`` runs its real requests against
real merge results. Only the endpoints BESSER calls exist, with the status codes
GitHub documents for them (422 for a non-fast-forward ref update, 409 for a merge
conflict, 204 for a merge with nothing to merge, ``Link`` headers for paging,
409 from the git data API on a repository with no commit, which only the
contents API can give its first one).

``use_fake_github(monkeypatch, fake)`` routes the GitHub requests of every
``httpx.AsyncClient`` that does not bring its own transport to the fake, so the
backend's GitHub calls land here while a test client keeps its own transport.

The review surface of a pull request (checks with their Actions job logs, commit
statuses, review threads, reviews, comments, mergeability) is shaped through
``fake.pull_review(repo, number)`` and read back by ``POST /graphql``, which
answers only the operation ``LabPullRequest`` (variables ``owner``, ``name``,
``number``) and 400 for any other, so an unplanned query fails loudly. Job logs
redirect to ``LOG_HOST`` the way GitHub redirects to blob storage.
``fake.calls`` counts the requests per URL path.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from httpx._transports.asgi import ASGITransport

API = "https://api.github.com"
# Stands in for the blob storage GitHub redirects a job log download to.
LOG_HOST = "logs.fake-github.test"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
# GitHub's rule for an account or repository name; "." and ".." are refused too.
_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,100}")
_OPERATION_RE = re.compile(r"\b(?:query|mutation)\s+(\w+)")
# GitHub's 403 for an OAuth token without the ``workflow`` scope touching a workflow file.
WORKFLOW_REFUSAL = ("refusing to allow an OAuth App to create or update workflow `{path}` "
                    "without `workflow` scope")
MERGE_CONFLICT_MESSAGE = "merge conflict between base and head"
NOTHING_TO_UPDATE_MESSAGE = "There are no new commits on the base branch."
STALE_HEAD_MESSAGE = "expected_head_sha does not match the pull request's head"


@dataclass
class FakePullReview:
    """What GitHub's review surface shows for one pull request.

    ``checks``: ``{name, app="github-actions", status="completed", conclusion, job_id, log, summary}``;
    ``log`` is the job's log bytes, None when it expired (410).
    ``statuses``: ``{context, state, target_url, description}`` (commit statuses, ``state`` lower case).
    ``threads``: ``{path, line, start_line, resolved, outdated, comments: [{author, association, body}]}``.
    ``reviews``: ``{author, state, body}``; ``comments``: ``{author, association, body}``.
    ``mergeable``: ``MERGEABLE``, ``CONFLICTING`` or ``UNKNOWN``.
    """

    checks: list[dict] = field(default_factory=list)
    statuses: list[dict] = field(default_factory=list)
    threads: list[dict] = field(default_factory=list)
    reviews: list[dict] = field(default_factory=list)
    comments: list[dict] = field(default_factory=list)
    mergeable: str = "MERGEABLE"


@dataclass
class FakeRepo:
    owner: str
    name: str
    path: str
    private: bool = True
    default_branch: str = "main"
    topics: list[str] = field(default_factory=list)
    # Whether the signed-in user may push (a read-only collaborator may not).
    can_push: bool = True
    pulls: list[dict] = field(default_factory=list)
    # Kept apart from ``pulls``, which are served as REST JSON (a log is bytes).
    pull_reviews: dict[int, FakePullReview] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class FakeGitHub:
    """In-memory GitHub account ``login`` with git-backed repositories."""

    def __init__(self, login: str = "ana", token: str = "token", root: Optional[str] = None):
        self.login = login
        self.token = token
        self.root = root or tempfile.mkdtemp(prefix="besser_fake_github_")
        self.repos: dict[str, FakeRepo] = {}
        self.revoked: list[str] = []
        self.gists: dict[str, dict] = {}
        # Called as ``(repo, branch)`` before a ref update is checked: a test can
        # land someone else's commit between BESSER reading the head and moving it.
        self.before_ref_update: Optional[Callable[[FakeRepo, str], None]] = None
        # GitHub refuses an update-branch that brings in a workflow change when the token lacks ``workflow``.
        self.refuse_workflow_updates = False
        self.calls: Counter[str] = Counter()
        self.app = self._build_app()

    # ------------------------------------------------------------------
    # git helpers
    # ------------------------------------------------------------------
    def git(self, repo: FakeRepo, *args: str, input: Optional[bytes] = None, env: Optional[dict] = None,
            check: bool = True) -> subprocess.CompletedProcess:
        full_env = {
            **os.environ,
            "GIT_AUTHOR_NAME": self.login, "GIT_AUTHOR_EMAIL": f"{self.login}@example.com",
            "GIT_COMMITTER_NAME": self.login, "GIT_COMMITTER_EMAIL": f"{self.login}@example.com",
            **(env or {}),
        }
        result = subprocess.run(
            # No line-ending conversion: GitHub serves blobs as stored.
            ["git", "-c", "core.autocrlf=false", "-C", repo.path, *args], input=input, capture_output=True, env=full_env,
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.decode(errors='replace')}")
        return result

    def out(self, repo: FakeRepo, *args: str, **kwargs) -> str:
        return self.git(repo, *args, **kwargs).stdout.decode().strip()

    def ref(self, repo: FakeRepo, branch: str) -> Optional[str]:
        result = self.git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}^{{commit}}", check=False)
        return result.stdout.decode().strip() or None

    def blob(self, repo: FakeRepo, content: bytes) -> str:
        return self.out(repo, "hash-object", "-w", "--stdin", input=content)

    def make_tree(self, repo: FakeRepo, entries: list[dict], base_tree: Optional[str]) -> str:
        """A tree from ``base_tree`` plus ``entries``; raises KeyError deleting a missing path."""
        index = os.path.join(self.root, f"index-{os.getpid()}-{id(entries)}")
        env = {"GIT_INDEX_FILE": index}
        try:
            self.git(repo, "read-tree", base_tree or "--empty", env=env)
            present = set(self.out(repo, "ls-files", env=env).splitlines())
            lines = []
            for entry in entries:
                path = entry["path"]
                if entry.get("sha") is None and "content" not in entry:
                    if path not in present:
                        raise KeyError(path)
                    lines.append(f"0 {'0' * 40}\t{path}")
                    continue
                sha = entry.get("sha") or self.blob(repo, entry["content"].encode())
                lines.append(f"{entry['mode']} {sha}\t{path}")
            if lines:
                self.git(repo, "update-index", "--index-info", input="\n".join(lines).encode() + b"\n", env=env)
            return self.out(repo, "write-tree", env=env)
        finally:
            if os.path.exists(index):
                os.remove(index)

    def commit(self, repo: FakeRepo, tree: str, parents: list[str], message: str,
               author: Optional[str] = None) -> str:
        args = ["commit-tree", tree]
        for parent in parents:
            args += ["-p", parent]
        env = {"GIT_AUTHOR_NAME": author} if author else None
        return self.out(repo, *args, "-m", message, env=env)

    def set_ref(self, repo: FakeRepo, branch: str, sha: Optional[str]) -> None:
        if sha is None:
            self.git(repo, "update-ref", "-d", f"refs/heads/{branch}")
        else:
            self.git(repo, "update-ref", f"refs/heads/{branch}", sha)

    def tree_of(self, repo: FakeRepo, commit: str) -> str:
        return self.out(repo, "rev-parse", f"{commit}^{{tree}}")

    def is_empty(self, repo: FakeRepo) -> bool:
        """True for a repository with no commit (created without ``auto_init``)."""
        return not self.out(repo, "for-each-ref", "refs/heads")

    def is_ancestor(self, repo: FakeRepo, ancestor: str, descendant: str) -> bool:
        return self.git(repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False).returncode == 0

    # ------------------------------------------------------------------
    # test API: shape the account
    # ------------------------------------------------------------------
    def create_repo(self, name: str, owner: Optional[str] = None, files: Optional[dict] = None,
                    private: bool = True, topics: Optional[list[str]] = None, can_push: bool = True,
                    default_branch: str = "main") -> FakeRepo:
        owner = owner or self.login
        if not all(_NAME_RE.fullmatch(part) and part not in {".", ".."} for part in (owner, name)):
            raise ValueError(f"invalid repository name: {owner}/{name}")
        root = os.path.realpath(self.root)
        path = os.path.realpath(os.path.join(root, owner, f"{name}.git"))
        if not path.startswith(root + os.sep):
            raise ValueError(f"invalid repository name: {owner}/{name}")
        os.makedirs(path, exist_ok=True)
        subprocess.run(["git", "init", "--bare", "-q", "-b", default_branch, path], check=True)
        repo = FakeRepo(owner, name, path, private, default_branch, list(topics or []), can_push)
        self.repos[repo.full_name.lower()] = repo
        if files:
            self.commit_files(repo, files, "Initial commit")
        return repo

    def commit_files(self, repo: FakeRepo, files: dict, message: str, branch: Optional[str] = None,
                     author: Optional[str] = None) -> str:
        """Commit ``files`` (path -> str/bytes, or ``(content, mode)``, None deletes) as a person would."""
        branch = branch or repo.default_branch
        head = self.ref(repo, branch)
        entries = []
        for path, value in files.items():
            if value is None:
                entries.append({"path": path, "mode": "100644", "sha": None})
                continue
            content, mode = value if isinstance(value, tuple) else (value, "100644")
            data = content.encode() if isinstance(content, str) else content
            entries.append({"path": path, "mode": mode, "sha": self.blob(repo, data)})
        tree = self.make_tree(repo, entries, self.tree_of(repo, head) if head else None)
        sha = self.commit(repo, tree, [head] if head else [], message, author)
        self.set_ref(repo, branch, sha)
        return sha

    def files(self, repo: FakeRepo, ref: Optional[str] = None) -> dict[str, tuple[str, bytes]]:
        """``path -> (mode, content)`` of every blob at ``ref`` (default branch head)."""
        ref = ref or repo.default_branch
        listing = self.git(repo, "ls-tree", "-r", "-z", ref).stdout.decode().split("\0")
        result = {}
        for line in filter(None, listing):
            meta, path = line.split("\t", 1)
            mode, kind, sha = meta.split()
            if kind == "blob":
                result[path] = (mode, self.git(repo, "cat-file", "blob", sha).stdout)
        return result

    def merge_pull(self, repo: FakeRepo, number: int) -> str:
        """Merge pull request ``number`` the way the GitHub UI would (raises on a conflict)."""
        pull = next(p for p in repo.pulls if p["number"] == number)
        sha = self._merge(repo, pull["base"]["ref"], pull["head"]["ref"], f"Merge pull request #{number}")
        if sha is None:
            raise RuntimeError("conflict")
        pull.update(state="closed", merged_at="2026-10-29T12:00:00Z")
        return sha

    def close_pull(self, repo: FakeRepo, number: int) -> None:
        """Close pull request ``number`` without merging it."""
        self._pull(repo, number).update(state="closed")

    def pull_review(self, repo: FakeRepo, number: int) -> FakePullReview:
        """The review surface of pull request ``number``, for a test to fill."""
        self._pull(repo, number)
        return repo.pull_reviews.setdefault(number, FakePullReview())

    # ------------------------------------------------------------------
    # GitHub semantics
    # ------------------------------------------------------------------
    @staticmethod
    def _pull(repo: FakeRepo, number: int) -> dict:
        pull = next((p for p in repo.pulls if p["number"] == number), None)
        if pull is None:
            raise _NotFound()
        return pull

    def _job_log(self, repo: FakeRepo, job_id: int) -> tuple[bool, Optional[bytes]]:
        """``(found, log)`` of an Actions job of ``repo``; ``log`` is None when it expired."""
        for review in repo.pull_reviews.values():
            for check in review.checks:
                if check.get("job_id") == job_id:
                    return True, check.get("log")
        return False, None

    def _update_branch(self, repo: FakeRepo, pull: dict, expected_head_sha: Optional[str]) -> JSONResponse:
        """``PUT pulls/{n}/update-branch``: merge the base into the head branch, as GitHub does."""
        head_ref, base_ref = pull["head"]["ref"], pull["base"]["ref"]
        head_sha, base_sha = self.ref(repo, head_ref), self.ref(repo, base_ref)
        if pull["state"] != "open":
            return JSONResponse({"message": "Pull request is closed"}, status_code=422)
        if expected_head_sha and expected_head_sha != head_sha:
            return JSONResponse({"message": STALE_HEAD_MESSAGE}, status_code=422)
        if self.refuse_workflow_updates:
            fork_point = self.out(repo, "merge-base", head_sha, base_sha)
            workflows = self.out(repo, "diff", "--name-only", fork_point, base_sha, "--", ".github/workflows/")
            if workflows:
                path = workflows.splitlines()[0]
                return JSONResponse({"message": WORKFLOW_REFUSAL.format(path=path)}, status_code=403)
        sha = self._merge(repo, head_ref, base_ref, f"Merge branch '{base_ref}' into {head_ref}")
        if sha is False:
            return JSONResponse({"message": NOTHING_TO_UPDATE_MESSAGE}, status_code=422)
        if sha is None:
            return JSONResponse({"message": MERGE_CONFLICT_MESSAGE}, status_code=422)
        return JSONResponse({"message": "Updating pull request branch.",
                             "url": f"https://github.com/{repo.full_name}/pull/{pull['number']}"}, status_code=202)

    def pull_request_graphql(self, repo: FakeRepo, pull: dict) -> dict:
        """The ``pullRequest`` object GitHub's GraphQL API gives for the ``LabPullRequest`` fields."""
        review = repo.pull_reviews.get(pull["number"], FakePullReview())
        number = pull["number"]
        url = f"https://github.com/{repo.full_name}/pull/{number}"
        head_sha, base_sha = self.ref(repo, pull["head"]["ref"]), self.ref(repo, pull["base"]["ref"])
        merged = bool(pull.get("merged_at"))
        state = "MERGED" if merged else pull["state"].upper()

        contexts = []
        for check in review.checks:
            job_id = check.get("job_id")
            contexts.append({
                "__typename": "CheckRun", "databaseId": job_id, "name": check["name"],
                "status": check.get("status", "completed").upper(),
                "conclusion": (check.get("conclusion") or "").upper() or None,
                "detailsUrl": f"https://github.com/{repo.full_name}/actions/runs/1/job/{job_id}",
                "title": check.get("title"), "summary": check.get("summary"),
                "checkSuite": {"app": {"slug": check.get("app", "github-actions")}},
            })
        for status in review.statuses:
            contexts.append({"__typename": "StatusContext", "context": status["context"],
                             "state": status["state"].upper(), "targetUrl": status.get("target_url"),
                             "description": status.get("description")})

        def author(entry: dict) -> dict:
            return {"login": entry["author"]}

        threads = []
        for t, thread in enumerate(review.threads):
            threads.append({
                "isResolved": bool(thread.get("resolved")), "isOutdated": bool(thread.get("outdated")),
                "path": thread["path"], "line": thread.get("line"), "startLine": thread.get("start_line"),
                "comments": {"nodes": [{
                    "id": f"PRRC_{number}_{t}_{c}", "author": author(comment),
                    "authorAssociation": comment.get("association", "NONE"), "body": comment["body"],
                    "url": f"{url}#discussion_r{number}{t}{c}", "createdAt": _timestamp(t * 10 + c),
                } for c, comment in enumerate(thread["comments"])]},
            })
        return {
            "number": number, "url": url, "state": state, "isDraft": bool(pull.get("draft")), "merged": merged,
            "headRefOid": head_sha, "mergeable": review.mergeable,
            "mergeStateStatus": self._merge_state(repo, review, head_sha, base_sha, pull),
            "reviewDecision": _review_decision(review.reviews),
            "commits": {"nodes": [{"commit": {"oid": head_sha, "statusCheckRollup": {
                "state": _rollup_state(contexts), "contexts": {"nodes": contexts},
            } if contexts else None}}]},
            "reviews": {"nodes": [{
                "author": author(r), "state": r["state"], "body": r.get("body", ""),
                "url": f"{url}#pullrequestreview-{i}", "submittedAt": _timestamp(i),
            } for i, r in enumerate(review.reviews)]},
            "reviewThreads": {"nodes": threads},
            "comments": {"nodes": [{
                "id": f"IC_{number}_{i}", "author": author(c), "authorAssociation": c.get("association", "NONE"),
                "body": c["body"], "url": f"{url}#issuecomment-{i}", "createdAt": _timestamp(i),
            } for i, c in enumerate(review.comments)]},
        }

    def _merge_state(self, repo: FakeRepo, review: FakePullReview, head_sha: Optional[str],
                     base_sha: Optional[str], pull: dict) -> str:
        """GitHub's ``mergeStateStatus``: conflicts, a base the head lacks, then clean."""
        if pull.get("draft"):
            return "DRAFT"
        if review.mergeable == "CONFLICTING":
            return "DIRTY"
        if review.mergeable == "UNKNOWN" or not head_sha or not base_sha:
            return "UNKNOWN"
        return "CLEAN" if self.is_ancestor(repo, base_sha, head_sha) else "BEHIND"

    def _graphql(self, body: dict) -> JSONResponse:
        query = body.get("query", "")
        match = _OPERATION_RE.search(query)
        operation = body.get("operationName") or (match.group(1) if match else None)
        if operation != "LabPullRequest":
            return JSONResponse({"message": f"The fake answers only LabPullRequest, not {operation!r}"},
                                status_code=400)
        variables = body.get("variables") or {}
        repo = self.repos.get(f"{variables.get('owner')}/{variables.get('name')}".lower())
        if repo is None:
            return JSONResponse({"data": {"repository": None}, "errors": [{
                "type": "NOT_FOUND", "path": ["repository"],
                "message": f"Could not resolve to a Repository with the name "
                           f"'{variables.get('owner')}/{variables.get('name')}'.",
            }]})
        pull = next((p for p in repo.pulls if p["number"] == variables.get("number")), None)
        if pull is None:
            return JSONResponse({"data": {"repository": {"pullRequest": None}}, "errors": [{
                "type": "NOT_FOUND", "path": ["repository", "pullRequest"],
                "message": f"Could not resolve to a PullRequest with the number of {variables.get('number')}.",
            }]})
        return JSONResponse({"data": {"repository": {"pullRequest": self.pull_request_graphql(repo, pull)}}})
    def _merge(self, repo: FakeRepo, base: str, head: str, message: str) -> Optional[str] | bool:
        base_sha, head_sha = self.ref(repo, base), self.ref(repo, head) or head
        if self.is_ancestor(repo, head_sha, base_sha):
            return False
        result = self.git(repo, "merge-tree", "--write-tree", base_sha, head_sha, check=False)
        if result.returncode == 1:
            return None
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())
        tree = result.stdout.decode().splitlines()[0]
        sha = self.commit(repo, tree, [base_sha, head_sha], message)
        self.set_ref(repo, base, sha)
        return sha

    def commit_json(self, repo: FakeRepo, sha: str) -> dict:
        raw = self.out(repo, "show", "-s", "--format=%H%x00%P%x00%an%x00%aI%x00%B", sha)
        full, parents, author, date, message = raw.split("\0", 4)
        return {
            "sha": full,
            "commit": {"message": message.strip(), "author": {"name": author, "date": date},
                       "tree": {"sha": self.tree_of(repo, full)}},
            "author": {"login": author},
            "html_url": f"https://github.com/{repo.full_name}/commit/{full}",
            "parents": [{"sha": p} for p in parents.split()],
        }

    def repo_json(self, repo: FakeRepo) -> dict:
        return {
            "id": abs(hash(repo.full_name)) % 10**8, "name": repo.name, "full_name": repo.full_name,
            "owner": {"login": repo.owner}, "description": None,
            "html_url": f"https://github.com/{repo.full_name}", "private": repo.private,
            "updated_at": "2026-10-29T10:00:00Z", "default_branch": repo.default_branch,
            "topics": repo.topics, "permissions": {"admin": repo.owner == self.login, "push": repo.can_push,
                                                   "pull": True},
        }

    # ------------------------------------------------------------------
    # HTTP surface
    # ------------------------------------------------------------------
    def _build_app(self) -> FastAPI:  # noqa: C901 - one route table
        app = FastAPI()
        fake = self

        @app.middleware("http")
        async def auth(request: Request, call_next):
            fake.calls[request.url.path] += 1
            if request.url.hostname == LOG_HOST:
                # Blob storage takes the signed URL alone; a forwarded GitHub token is refused, not used.
                if "authorization" in request.headers:
                    return Response("Server failed to authenticate the request.", status_code=403)
                return await call_next(request)
            if request.headers.get("authorization") != f"Bearer {fake.token}" \
                    and not request.url.path.startswith(("/applications/", "/login/oauth/")):
                return JSONResponse({"message": "Bad credentials"}, status_code=401)
            return await call_next(request)

        def find(owner: str, name: str) -> FakeRepo:
            repo = fake.repos.get(f"{owner}/{name}".lower())
            if repo is None:
                raise _NotFound()
            return repo

        def find_with_commits(owner: str, name: str) -> FakeRepo:
            """A repo for the git data API, which refuses one with no commit."""
            repo = find(owner, name)
            if fake.is_empty(repo):
                raise _Empty()
            return repo

        @app.exception_handler(_NotFound)
        async def not_found(_request, _exc):
            return JSONResponse({"message": "Not Found"}, status_code=404)

        @app.exception_handler(_Empty)
        async def empty(_request, _exc):
            return JSONResponse({"message": "Git Repository is empty."}, status_code=409)

        def paged(request: Request, items: list) -> Response:
            per_page = int(request.query_params.get("per_page", 30))
            page = int(request.query_params.get("page", 1))
            chunk = items[(page - 1) * per_page: page * per_page]
            headers = {}
            if page * per_page < len(items):
                params = dict(request.query_params)
                params["page"] = str(page + 1)
                query = "&".join(f"{k}={v}" for k, v in params.items())
                headers["Link"] = f'<{API}{request.url.path}?{query}>; rel="next"'
            return JSONResponse(chunk, headers=headers)

        @app.get("/user")
        async def user():
            return {"login": fake.login}

        @app.get("/user/repos")
        async def user_repos(request: Request):
            affiliation = request.query_params.get("affiliation", "owner,collaborator,organization_member")
            repos = [r for r in fake.repos.values()
                     if (r.owner == fake.login and "owner" in affiliation)
                     or (r.owner != fake.login and ("collaborator" in affiliation
                                                   or "organization_member" in affiliation))]
            return paged(request, [fake.repo_json(r) for r in repos])

        @app.post("/user/repos")
        async def create(request: Request):
            body = await request.json()
            if f"{fake.login}/{body['name']}".lower() in fake.repos:
                return JSONResponse({"message": "Repository creation failed.",
                                     "errors": [{"message": "name already exists on this account"}]},
                                    status_code=422)
            try:
                repo = fake.create_repo(body["name"], private=body.get("private", False),
                                        files={"README.md": f"# {body['name']}\n"} if body.get("auto_init") else None)
            except ValueError:
                return JSONResponse({"message": "Repository creation failed.",
                                     "errors": [{"message": "name is invalid"}]}, status_code=422)
            return JSONResponse(fake.repo_json(repo), status_code=201)

        @app.get("/repos/{owner}/{name}")
        async def get_repo(owner: str, name: str):
            return fake.repo_json(find(owner, name))

        @app.get("/repos/{owner}/{name}/topics")
        async def get_topics(owner: str, name: str):
            return {"names": find(owner, name).topics}

        @app.put("/repos/{owner}/{name}/topics")
        async def put_topics(owner: str, name: str, request: Request):
            find(owner, name).topics = (await request.json())["names"]
            return {"names": find(owner, name).topics}

        @app.get("/repos/{owner}/{name}/branches")
        async def branches(owner: str, name: str, request: Request):
            repo = find(owner, name)
            names = fake.out(repo, "for-each-ref", "--format=%(refname:lstrip=2)", "refs/heads").splitlines()
            return paged(request, [{"name": n, "commit": {"sha": fake.ref(repo, n)}} for n in sorted(names)])

        @app.get("/repos/{owner}/{name}/git/ref/heads/{branch:path}")
        async def get_ref(owner: str, name: str, branch: str):
            sha = fake.ref(find_with_commits(owner, name), branch)
            if not sha:
                raise _NotFound()
            return {"ref": f"refs/heads/{branch}", "object": {"sha": sha, "type": "commit"}}

        @app.post("/repos/{owner}/{name}/git/refs")
        async def create_ref(owner: str, name: str, request: Request):
            repo, body = find_with_commits(owner, name), await request.json()
            branch = body["ref"].removeprefix("refs/heads/")
            if fake.ref(repo, branch):
                return JSONResponse({"message": "Reference already exists"}, status_code=422)
            fake.set_ref(repo, branch, body["sha"])
            return JSONResponse({"ref": body["ref"], "object": {"sha": body["sha"]}}, status_code=201)

        @app.patch("/repos/{owner}/{name}/git/refs/heads/{branch:path}")
        async def update_ref(owner: str, name: str, branch: str, request: Request):
            repo, body = find_with_commits(owner, name), await request.json()
            if not repo.can_push:
                return JSONResponse({"message": "Permission denied"}, status_code=403)
            if fake.before_ref_update is not None:
                fake.before_ref_update(repo, branch)
            current = fake.ref(repo, branch)
            if current is None:
                return JSONResponse({"message": "Reference does not exist"}, status_code=422)
            if not body.get("force") and not fake.is_ancestor(repo, current, body["sha"]):
                return JSONResponse({"message": "Update is not a fast forward"}, status_code=422)
            fake.set_ref(repo, branch, body["sha"])
            return {"ref": f"refs/heads/{branch}", "object": {"sha": body["sha"]}}

        @app.delete("/repos/{owner}/{name}/git/refs/heads/{branch:path}")
        async def delete_ref(owner: str, name: str, branch: str):
            repo = find(owner, name)
            if not fake.ref(repo, branch):
                return JSONResponse({"message": "Reference does not exist"}, status_code=422)
            fake.set_ref(repo, branch, None)
            return Response(status_code=204)

        @app.post("/repos/{owner}/{name}/git/blobs")
        async def create_blob(owner: str, name: str, request: Request):
            body = await request.json()
            data = base64.b64decode(body["content"]) if body.get("encoding") == "base64" else body["content"].encode()
            return JSONResponse({"sha": fake.blob(find_with_commits(owner, name), data)}, status_code=201)

        @app.post("/repos/{owner}/{name}/git/trees")
        async def create_tree(owner: str, name: str, request: Request):
            repo, body = find_with_commits(owner, name), await request.json()
            try:
                sha = fake.make_tree(repo, body["tree"], body.get("base_tree"))
            except KeyError:
                return JSONResponse({"message": "GitRPC::BadObjectState"}, status_code=422)
            return JSONResponse({"sha": sha}, status_code=201)

        @app.get("/repos/{owner}/{name}/git/trees/{sha}")
        async def get_tree(owner: str, name: str, sha: str, request: Request):
            repo = find_with_commits(owner, name)
            args = ["ls-tree", "-l", "-z"]
            if request.query_params.get("recursive"):
                args += ["-r", "-t"]
            entries = []
            for line in filter(None, fake.git(repo, *args, sha).stdout.decode().split("\0")):
                meta, path = line.split("\t", 1)
                mode, kind, obj, size = meta.split()
                entry = {"path": path, "mode": mode, "type": kind, "sha": obj}
                if kind == "blob":
                    entry["size"] = int(size)
                entries.append(entry)
            return {"sha": sha, "tree": entries, "truncated": False}

        @app.post("/repos/{owner}/{name}/git/commits")
        async def create_commit(owner: str, name: str, request: Request):
            repo, body = find_with_commits(owner, name), await request.json()
            sha = fake.commit(repo, body["tree"], body.get("parents") or [], body["message"])
            return JSONResponse({"sha": sha, "tree": {"sha": body["tree"]}}, status_code=201)

        @app.get("/repos/{owner}/{name}/git/commits/{sha}")
        async def get_commit(owner: str, name: str, sha: str):
            data = fake.commit_json(find_with_commits(owner, name), sha)
            return {"sha": data["sha"], "tree": data["commit"]["tree"], "parents": data["parents"],
                    "message": data["commit"]["message"]}

        @app.post("/repos/{owner}/{name}/merges")
        async def merge(owner: str, name: str, request: Request):
            repo, body = find(owner, name), await request.json()
            if not repo.can_push:
                return JSONResponse({"message": "Permission denied"}, status_code=403)
            sha = fake._merge(repo, body["base"], body["head"], body.get("commit_message") or "Merge")
            if sha is False:
                return Response(status_code=204)
            if sha is None:
                return JSONResponse({"message": "Merge conflict"}, status_code=409)
            return JSONResponse(fake.commit_json(repo, sha), status_code=201)

        @app.get("/repos/{owner}/{name}/compare/{spec:path}")
        async def compare(owner: str, name: str, spec: str):
            repo = find(owner, name)
            base, head = spec.split("...")
            # GitHub answers 404 for a commit the repo does not have (e.g. after a force-push).
            if any(fake.git(repo, "cat-file", "-e", f"{ref}^{{commit}}", check=False).returncode for ref in (base, head)):
                return JSONResponse({"message": "Not Found"}, status_code=404)
            ahead = fake.out(repo, "rev-list", "--reverse", f"{base}..{head}").split()
            behind = fake.out(repo, "rev-list", f"{head}..{base}").split()
            status = "identical" if not ahead and not behind else "ahead" if not behind else \
                "behind" if not ahead else "diverged"
            return {"status": status, "ahead_by": len(ahead), "behind_by": len(behind),
                    "commits": [fake.commit_json(repo, c) for c in ahead]}

        @app.get("/repos/{owner}/{name}/commits")
        async def commits(owner: str, name: str, request: Request):
            repo = find(owner, name)
            ref = request.query_params.get("sha") or repo.default_branch
            args = ["rev-list", f"--max-count={request.query_params.get('per_page', 30)}", ref]
            if request.query_params.get("path"):
                args += ["--", request.query_params["path"]]
            return [fake.commit_json(repo, c) for c in fake.out(repo, *args).split()]

        @app.get("/repos/{owner}/{name}/contents/{path:path}")
        async def contents(owner: str, name: str, path: str, request: Request):
            repo = find(owner, name)
            ref = request.query_params.get("ref") or repo.default_branch
            spec = f"{ref}:{path}" if path else f"{ref}^{{tree}}"
            kind = fake.git(repo, "cat-file", "-t", spec, check=False).stdout.decode().strip()
            if not kind:
                raise _NotFound()
            if kind == "blob":
                data = fake.git(repo, "cat-file", "blob", spec).stdout
                if "raw" in request.headers.get("accept", ""):
                    return Response(content=data)
                return {"type": "file", "name": path.rsplit("/", 1)[-1], "path": path,
                        "sha": fake.out(repo, "rev-parse", spec), "size": len(data),
                        "content": base64.b64encode(data).decode(), "encoding": "base64"}
            items = []
            for line in filter(None, fake.git(repo, "ls-tree", "-l", "-z", spec).stdout.decode().split("\0")):
                meta, child = line.split("\t", 1)
                _mode, child_kind, obj, size = meta.split()
                items.append({"name": child, "path": f"{path}/{child}".lstrip("/"), "sha": obj,
                              "type": "dir" if child_kind == "tree" else "file",
                              "size": 0 if size == "-" else int(size)})
            return items

        @app.put("/repos/{owner}/{name}/contents/{path:path}")
        async def put_contents(owner: str, name: str, path: str, request: Request):
            repo, body = find(owner, name), await request.json()
            branch = body.get("branch") or repo.default_branch
            head = fake.ref(repo, branch)
            if not head and not fake.is_empty(repo):
                return JSONResponse({"message": f"Branch {branch} not found"}, status_code=404)
            existing = fake.out(repo, "rev-parse", "--verify", "--quiet", f"{head}:{path}", check=False) if head else ""
            if existing and body.get("sha") != existing:
                return JSONResponse({"message": "\"sha\" wasn't supplied."}, status_code=422)
            sha = fake.commit_files(repo, {path: base64.b64decode(body["content"])}, body["message"], branch=branch)
            blob = fake.out(repo, "rev-parse", f"{sha}:{path}")
            return JSONResponse({"content": {"path": path, "sha": blob}, "commit": {"sha": sha}},
                                status_code=200 if existing else 201)

        @app.get("/repos/{owner}/{name}/tarball/{ref:path}")
        async def tarball(owner: str, name: str, ref: str):
            repo = find(owner, name)
            sha = fake.out(repo, "rev-parse", ref)
            data = fake.git(repo, "archive", "--format=tar.gz", f"--prefix={owner}-{name}-{sha[:7]}/", sha).stdout
            return Response(content=data, media_type="application/x-gzip")

        @app.post("/repos/{owner}/{name}/pulls")
        async def create_pull(owner: str, name: str, request: Request):
            repo, body = find(owner, name), await request.json()
            if not fake.ref(repo, body["head"]):
                return JSONResponse({"message": "Validation Failed", "errors": [{"field": "head"}]}, status_code=422)
            if any(p["head"]["ref"] == body["head"] and p["state"] == "open" for p in repo.pulls):
                return JSONResponse({"message": "Validation Failed",
                                     "errors": [{"message": "A pull request already exists"}]}, status_code=422)
            number = len(repo.pulls) + 1
            pull = {"number": number, "state": "open", "merged_at": None, "title": body["title"],
                    "body": body.get("body", ""), "html_url": f"https://github.com/{repo.full_name}/pull/{number}",
                    "head": {"ref": body["head"]}, "base": {"ref": body["base"]},
                    "draft": bool(body.get("draft", False))}
            repo.pulls.append(pull)
            return JSONResponse(pull, status_code=201)

        @app.patch("/repos/{owner}/{name}/pulls/{number}")
        async def update_pull(owner: str, name: str, number: int, request: Request):
            pull = next((p for p in find(owner, name).pulls if p["number"] == number), None)
            if pull is None:
                raise _NotFound()
            pull.update({k: v for k, v in (await request.json()).items() if k in ("title", "body", "state")})
            return pull

        @app.get("/repos/{owner}/{name}/pulls")
        async def list_pulls(owner: str, name: str, request: Request):
            repo = find(owner, name)
            head = request.query_params.get("head", "")
            state = request.query_params.get("state", "open")
            branch = head.split(":", 1)[-1]
            return [p for p in repo.pulls
                    if (not head or p["head"]["ref"] == branch) and (state == "all" or p["state"] == state)]

        @app.get("/repos/{owner}/{name}/pulls/{number}")
        async def get_pull(owner: str, name: str, number: int, request: Request):
            repo = find(owner, name)
            pull = fake._pull(repo, number)
            if request.headers.get("accept") == "application/vnd.github.diff":
                # A pull request's diff is against the merge base, like ``git diff base...head``.
                spec = f"{pull['base']['ref']}...{pull['head']['ref']}"
                diff = fake.git(repo, "diff", "--no-color", "--no-ext-diff", spec).stdout
                return Response(content=diff, media_type="application/vnd.github.diff")
            return pull

        @app.put("/repos/{owner}/{name}/pulls/{number}/update-branch")
        async def update_branch(owner: str, name: str, number: int, request: Request):
            repo = find(owner, name)
            body = await request.json() if await request.body() else {}
            return fake._update_branch(repo, fake._pull(repo, number), body.get("expected_head_sha"))

        @app.get("/repos/{owner}/{name}/actions/jobs/{job_id}/logs")
        async def job_logs(owner: str, name: str, job_id: int):
            found, log = fake._job_log(find(owner, name), job_id)
            if not found:
                raise _NotFound()
            if log is None:
                return JSONResponse({"message": "Gone"}, status_code=410)
            # The real Location is a short-lived signed blob URL.
            return Response(status_code=302, headers={"Location": f"https://{LOG_HOST}/{job_id}?sig=signed"})

        @app.get("/{job_id:int}")
        async def log_blob(job_id: int, request: Request):
            if request.url.hostname != LOG_HOST:
                raise _NotFound()
            for repo in fake.repos.values():
                found, log = fake._job_log(repo, job_id)
                if found and log is not None:
                    return _log_range(log, request.headers.get("range"))
            raise _NotFound()

        @app.post("/graphql")
        async def graphql(request: Request):
            return fake._graphql(await request.json())

        @app.post("/gists")
        async def create_gist(request: Request):
            body = await request.json()
            gist_id = uuid.uuid4().hex
            fake.gists[gist_id] = body
            return JSONResponse({"id": gist_id, "html_url": f"https://gist.github.com/{gist_id}", "files": {
                name: {"raw_url": f"https://gist.githubusercontent.com/{gist_id}/{name}"} for name in body["files"]}},
                status_code=201)

        @app.post("/login/oauth/access_token")
        async def exchange_code():
            # github.com's OAuth code exchange: any code yields the account's token.
            return {"access_token": fake.token, "token_type": "bearer", "scope": "repo,gist,user"}

        @app.delete("/applications/{client_id}/token")
        async def revoke(client_id: str, request: Request):
            fake.revoked.append((await request.json())["access_token"])
            return Response(status_code=204)

        return app


class _NotFound(Exception):
    pass


class _Empty(Exception):
    pass


def _timestamp(index: int) -> str:
    """An ISO time that orders entries as listed."""
    return f"2026-10-29T12:{index // 60 % 60:02d}:{index % 60:02d}Z"


def _review_decision(reviews: list[dict]) -> Optional[str]:
    """GitHub's ``reviewDecision`` from each reviewer's latest approving or blocking review."""
    latest = {r["author"]: r["state"] for r in reviews if r["state"] in ("APPROVED", "CHANGES_REQUESTED")}
    if "CHANGES_REQUESTED" in latest.values():
        return "CHANGES_REQUESTED"
    return "APPROVED" if latest else None


def _rollup_state(contexts: list[dict]) -> str:
    """``statusCheckRollup.state``: a failure wins, then anything unfinished, else success."""
    failed = {"FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE"}
    states = [c["conclusion"] if c["__typename"] == "CheckRun" and c["status"] == "COMPLETED"
              else c["state"] if c["__typename"] == "StatusContext" else "PENDING" for c in contexts]
    if any(s in failed for s in states):
        return "FAILURE"
    return "PENDING" if any(s in ("PENDING", "EXPECTED") for s in states) else "SUCCESS"


def _log_range(log: bytes, header: Optional[str]) -> Response:
    """Blob storage's answer to ``Range: bytes=-N`` (a suffix); without it, the whole log."""
    match = re.fullmatch(r"bytes=-(\d+)", header or "")
    if match is None:
        return Response(content=log, media_type="text/plain")
    if not log or int(match.group(1)) == 0:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{len(log)}"})
    tail = log[-int(match.group(1)):]
    start = len(log) - len(tail)
    return Response(content=tail, status_code=206, media_type="text/plain",
                    headers={"Content-Range": f"bytes {start}-{len(log) - 1}/{len(log)}"})


GITHUB_HOSTS = ("api.github.com", "github.com", LOG_HOST)


class _GitHubRouter(httpx.AsyncBaseTransport):
    """GitHub's hosts go to the fake; every other host to the network as usual."""

    def __init__(self, fake: FakeGitHub):
        self.fake = ASGITransport(app=fake.app)
        self.network = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        target = self.fake if request.url.host in GITHUB_HOSTS else self.network
        return await target.handle_async_request(request)


def github_client_class(fake: FakeGitHub) -> type:
    """An ``httpx.AsyncClient`` whose GitHub requests reach ``fake`` unless it is given a transport."""
    router = _GitHubRouter(fake)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", router)
            super().__init__(*args, **kwargs)

    return _Client


def use_fake_github(monkeypatch, fake: FakeGitHub) -> None:
    """Route the backend's GitHub requests to ``fake`` for one test."""
    monkeypatch.setattr(httpx, "AsyncClient", github_client_class(fake))
