"""Shared fixtures of the backend lab tests: a real lab workspace whose GitHub is a ``FakeGitHub``.

``lab`` (a fixture, :class:`Lab`) is the disk layout of ``tests/spec_driven_agent/lab/conftest.py``
(base clones under ``<root>/base``, git directories in a separate git root) with its upstreams
mirrored into a :class:`FakeGitHub` as ``BESSER-PEARL/<name>``, and one open change,
"Add a note element", owned by :data:`LOGIN`. Fetches of the bases read the fake's bare
repositories; a ``url.<fake>.insteadOf`` entry in the test's own global git config sends the
production ``https://github.com/...`` push URLs there too, so pushes run unchanged. Every
``httpx.AsyncClient`` request goes to the fake through :class:`RecordingTransport`, which
records it and lets a test answer it instead (``lab.transport.answer``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Optional

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.spec_driven_agent.lab import workspace as workspace_module
from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    REPO_ORDER,
    ChangeKind,
    LabChange,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import GitHubService
from besser.utilities.web_modeling_editor.backend.services.lab.pull_requests import open_pull_requests
from tests.spec_driven_agent.lab.conftest import (  # noqa: F401  (fixtures used through ``lab``)
    LabLayout,
    git,
    lab_layout,
    lab_layout_template,
    write,
)
from tests.utilities.web_modeling_editor.backend.fake_github import FakeGitHub, FakeRepo

LOGIN = "jordi"
TOKEN = "gho_" + "A1b2C3d4" * 4 + "Zz9Y"
OWNER = "BESSER-PEARL"
#: Repository key -> its name on GitHub (``DEFAULT_UPSTREAMS``).
NAMES = {REPO_FRONTEND: "BESSER-Web-Modeling-Editor", REPO_BESSER: "BESSER", REPO_MODELING_AGENT: "modeling-agent"}
TITLE = "Add a note element"
BRANCH = "lab/jordi/add-a-note-element"


class RecordingWorkspace(LabWorkspace):
    """The lab's workspace, recording what is passed to git and to ``commit_all``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[str, tuple[str, ...], Optional[dict]]] = []
        #: (message, login) of each commit_all
        self.commits: list[tuple[str, str]] = []

    def commit_all(self, change: LabChange, message: str, *, login: str) -> dict[str, str]:
        self.commits.append((message, login))
        return super().commit_all(change, message, login=login)

    def run_git(self, repo: str, *args: str, change: Optional[LabChange] = None, env_extra=None,
                timeout: float = 120) -> str:
        self.calls.append((repo, args, dict(env_extra) if env_extra else None))
        return super().run_git(repo, *args, change=change, env_extra=env_extra, timeout=timeout)


class RecordingTransport(httpx.AsyncBaseTransport):
    """Every request goes to the fake; method, path and JSON body (None when not JSON) are recorded."""

    def __init__(self, fake: FakeGitHub):
        self.inner = ASGITransport(app=fake.app)
        self.seen: list[tuple[str, str, object]] = []
        # A test sets it to answer a request itself (a GitHub refusal); None lets the fake answer.
        self.answer: Callable[[httpx.Request], Optional[httpx.Response]] = lambda request: None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            # A followed redirect arrives unread.
            body = json.loads(await request.aread() or b"null")
        except ValueError:
            body = None
        self.seen.append((request.method, request.url.path, body))
        return self.answer(request) or await self.inner.handle_async_request(request)


class Lab:
    """A lab workspace with one open change, and the fake GitHub behind it."""

    def __init__(self, fake: FakeGitHub, layout: LabLayout, workspace: RecordingWorkspace, change: LabChange,
                 transport: RecordingTransport, home: Path):
        self.fake = fake
        self.layout = layout
        self.workspace = workspace
        self.change = change
        self.transport = transport
        self.requests = transport.seen
        self.home = home

    @property
    def github(self) -> GitHubService:
        """A client with the signed-in user's token."""
        return GitHubService(TOKEN)

    def path(self, repo: str, change: Optional[LabChange] = None) -> Path:
        """The change's working tree of ``repo``."""
        return Path(self.workspace.change_repos(change or self.change)[repo].path)

    def write(self, repo: str, path: str, text: str, change: Optional[LabChange] = None) -> None:
        write(str(self.path(repo, change) / path), text)

    def remote(self, repo: str, owner: str = OWNER) -> FakeRepo:
        """``repo``'s repository on the fake GitHub."""
        return self.fake.repos[f"{owner}/{NAMES[repo]}".lower()]

    def open(self, change: Optional[LabChange] = None, **kwargs):
        """``open_pull_requests`` as the user :data:`LOGIN` (``login`` may be overridden); returns its answer."""
        kwargs.setdefault("login", LOGIN)
        return asyncio.run(open_pull_requests(self.workspace, change or self.change, access_token=TOKEN, **kwargs))

    def sent(self, method: str, suffix: str) -> list:
        return [body for m, path, body in self.requests if m == method and path.endswith(suffix)]

    def pushes(self) -> list[tuple]:
        return [call for call in self.workspace.calls if call[1] and call[1][0] == "push"]

    def mirror(self, repo: str, owner: str) -> FakeRepo:
        """A copy of ``repo``'s upstream on the fake as ``owner/<name>`` (a fork), reachable by git."""
        source = self.remote(repo)
        fork = self.fake.create_repo(NAMES[repo], owner=owner, default_branch=source.default_branch)
        git(source.path, "push", "-q", fork.path, "refs/heads/*:refs/heads/*")
        _route_git_to_fake(self.fake, self.home)
        return fork

    def new_change(self, workspace: Optional[LabWorkspace] = None, *, title: str = TITLE,
                   kind: ChangeKind = ChangeKind.EDITOR_ELEMENT, owner: str = LOGIN) -> LabChange:
        """Open another change (in ``workspace``, the lab's by default)."""
        workspace = workspace or self.workspace
        return workspace.open_change(workspace.create_draft(owner=owner), title=title, kind=kind)


def _route_git_to_fake(fake: FakeGitHub, home: Path) -> None:
    """``$HOME/.gitconfig``: every fake repository stands in for its github.com URL."""
    config = ["[user]", "\tname = setup", "\temail = setup@example.com", "[protocol \"file\"]", "\tallow = always"]
    for repo in fake.repos.values():
        config += [f"[url \"{Path(repo.path).as_uri()}\"]", f"\tinsteadOf = https://github.com/{repo.full_name}.git"]
    (home / ".gitconfig").write_text("\n".join(config) + "\n", encoding="utf-8")


@pytest.fixture
def lab(lab_layout, tmp_path, monkeypatch) -> Lab:  # noqa: F811 - lab_layout is the imported fixture
    """A lab workspace on a fake GitHub with the open change "Add a note element" of :data:`LOGIN`."""
    fake = FakeGitHub(login=LOGIN, token=TOKEN, root=str(tmp_path / "github"))
    specs = {spec.key: spec for spec in lab_layout.specs()}
    for key in REPO_ORDER:
        remote = fake.create_repo(NAMES[key], owner=OWNER, default_branch=specs[key].base_branch)
        git(lab_layout.upstream[key], "push", "-q", remote.path, "refs/heads/*:refs/heads/*")
    monkeypatch.setattr(workspace_module, "upstream_url",
                        lambda spec: fake.repos[f"{spec.github_owner}/{spec.github_name}".lower()].path)

    # git reads $HOME/.gitconfig: the fake's repositories stand in for github.com there.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _route_git_to_fake(fake, home)

    transport = RecordingTransport(fake)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", transport)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    workspace = RecordingWorkspace(lab_layout.root, lab_layout.specs(), state_dir=lab_layout.state_dir,
                                   git_root=lab_layout.git_root)
    workspace.ensure_ready()
    change = workspace.open_change(workspace.create_draft(owner=LOGIN), title=TITLE, kind=ChangeKind.EDITOR_ELEMENT)
    workspace.calls.clear()
    return Lab(fake, lab_layout, workspace, change, transport, home)
