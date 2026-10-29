"""A GitHub REST API stand-in backed by real git repositories.

Blobs, trees, commits, refs, merges, compares and tarballs are real git objects
made with the ``git`` CLI, so ``GitHubService`` runs its real requests against
real merge results. Only the endpoints BESSER calls exist, with the status codes
GitHub documents for them (422 for a non-fast-forward ref update, 409 for a merge
conflict, 204 for a merge with nothing to merge, ``Link`` headers for paging).

``use_fake_github(monkeypatch, fake)`` routes the GitHub requests of every
``httpx.AsyncClient`` that does not bring its own transport to the fake, so the
backend's GitHub calls land here while a test client keeps its own transport.
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from httpx._transports.asgi import ASGITransport

API = "https://api.github.com"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


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

    def is_ancestor(self, repo: FakeRepo, ancestor: str, descendant: str) -> bool:
        return self.git(repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False).returncode == 0

    # ------------------------------------------------------------------
    # test API: shape the account
    # ------------------------------------------------------------------
    def create_repo(self, name: str, owner: Optional[str] = None, files: Optional[dict] = None,
                    private: bool = True, topics: Optional[list[str]] = None, can_push: bool = True,
                    default_branch: str = "main") -> FakeRepo:
        owner = owner or self.login
        path = os.path.join(self.root, owner, f"{name}.git")
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

    # ------------------------------------------------------------------
    # GitHub semantics
    # ------------------------------------------------------------------
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
            if request.headers.get("authorization") != f"Bearer {fake.token}" \
                    and not request.url.path.startswith(("/applications/", "/login/oauth/")):
                return JSONResponse({"message": "Bad credentials"}, status_code=401)
            return await call_next(request)

        def find(owner: str, name: str) -> FakeRepo:
            repo = fake.repos.get(f"{owner}/{name}".lower())
            if repo is None:
                raise _NotFound()
            return repo

        @app.exception_handler(_NotFound)
        async def not_found(_request, _exc):
            return JSONResponse({"message": "Not Found"}, status_code=404)

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
            repo = fake.create_repo(body["name"], private=body.get("private", False),
                                    files={"README.md": f"# {body['name']}\n"} if body.get("auto_init") else None)
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
            sha = fake.ref(find(owner, name), branch)
            if not sha:
                raise _NotFound()
            return {"ref": f"refs/heads/{branch}", "object": {"sha": sha, "type": "commit"}}

        @app.post("/repos/{owner}/{name}/git/refs")
        async def create_ref(owner: str, name: str, request: Request):
            repo, body = find(owner, name), await request.json()
            branch = body["ref"].removeprefix("refs/heads/")
            if fake.ref(repo, branch):
                return JSONResponse({"message": "Reference already exists"}, status_code=422)
            fake.set_ref(repo, branch, body["sha"])
            return JSONResponse({"ref": body["ref"], "object": {"sha": body["sha"]}}, status_code=201)

        @app.patch("/repos/{owner}/{name}/git/refs/heads/{branch:path}")
        async def update_ref(owner: str, name: str, branch: str, request: Request):
            repo, body = find(owner, name), await request.json()
            if not repo.can_push:
                return JSONResponse({"message": "Permission denied"}, status_code=403)
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
            return JSONResponse({"sha": fake.blob(find(owner, name), data)}, status_code=201)

        @app.post("/repos/{owner}/{name}/git/trees")
        async def create_tree(owner: str, name: str, request: Request):
            repo, body = find(owner, name), await request.json()
            try:
                sha = fake.make_tree(repo, body["tree"], body.get("base_tree"))
            except KeyError as missing:
                return JSONResponse({"message": f"GitRPC::BadObjectState {missing}"}, status_code=422)
            return JSONResponse({"sha": sha}, status_code=201)

        @app.get("/repos/{owner}/{name}/git/trees/{sha}")
        async def get_tree(owner: str, name: str, sha: str, request: Request):
            repo = find(owner, name)
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
            repo, body = find(owner, name), await request.json()
            sha = fake.commit(repo, body["tree"], body.get("parents") or [], body["message"])
            return JSONResponse({"sha": sha, "tree": {"sha": body["tree"]}}, status_code=201)

        @app.get("/repos/{owner}/{name}/git/commits/{sha}")
        async def get_commit(owner: str, name: str, sha: str):
            data = fake.commit_json(find(owner, name), sha)
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
                    "head": {"ref": body["head"]}, "base": {"ref": body["base"]}}
            repo.pulls.append(pull)
            return JSONResponse(pull, status_code=201)

        @app.get("/repos/{owner}/{name}/pulls")
        async def list_pulls(owner: str, name: str, request: Request):
            repo = find(owner, name)
            head = request.query_params.get("head", "")
            state = request.query_params.get("state", "open")
            branch = head.split(":", 1)[-1]
            return [p for p in repo.pulls
                    if (not head or p["head"]["ref"] == branch) and (state == "all" or p["state"] == state)]

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


GITHUB_HOSTS = ("api.github.com", "github.com")


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
