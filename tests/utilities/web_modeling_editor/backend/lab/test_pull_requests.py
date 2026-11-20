"""Draft pull requests of a BESSER Lab change, pushed from real git working copies.

The working copies (BESSER with the editor as a submodule, and the modeling
agent) are real repositories under ``tmp_path``. Their GitHub remotes are the
bare repositories of ``FakeGitHub``: a ``url.<fake>.insteadOf`` entry in the
test's own global git config sends the production ``https://github.com/...``
push URL there, so the push and the pull request requests run unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.spec_driven_agent.lab.contracts import (
    FRONTEND_SUBMODULE_PATH,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    ChangeKind,
    LabChange,
    RepoSpec,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError
from besser.utilities.subprocess_env import safe_subprocess_env
from besser.utilities.web_modeling_editor.backend.services.lab.pull_requests import (
    PullRequestError,
    PullRequestInfo,
    default_title,
    open_pull_requests,
    push_environment,
)
from tests.utilities.web_modeling_editor.backend.fake_github import FakeGitHub

LOGIN = "jordi"
TOKEN = "gho_" + "A1b2C3d4" * 4 + "Zz9Y"
BRANCH = "lab/jordi/note-element"
OWNER = "BESSER-PEARL"
EDITOR_NAME = "BESSER-Web-Modeling-Editor"


def git(cwd, *args: str) -> str:
    result = subprocess.run(["git", "-c", "core.autocrlf=false", "-C", str(cwd), *args],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


class RecordingWorkspace(LabWorkspace):
    """The lab's workspace, recording what the pull requests pass to git."""

    def __init__(self, root: str, repos: list[RepoSpec]):
        super().__init__(root, repos)
        self.calls: list[tuple[str, tuple[str, ...], dict | None]] = []
        self.commit_messages: list[str] = []

    def commit_all(self, change: LabChange, message: str) -> dict[str, str]:
        self.commit_messages.append(message)
        return super().commit_all(change, message)

    def run_git(self, repo: str, *args: str, env_extra=None, timeout: float = 120) -> str:
        self.calls.append((repo, args, env_extra))
        return super().run_git(repo, *args, env_extra=env_extra, timeout=timeout)


class Lab:
    def __init__(self, fake: FakeGitHub, workspace: RecordingWorkspace, change: LabChange,
                 transport: "_RecordingTransport"):
        self.fake = fake
        self.workspace = workspace
        self.change = change
        self.transport = transport
        self.requests = transport.seen

    def path(self, repo: str) -> Path:
        return Path(self.workspace.repos[repo].path)

    def open(self, **kwargs) -> list[PullRequestInfo]:
        return asyncio.run(open_pull_requests(self.workspace, self.change, access_token=TOKEN, login=LOGIN,
                                              **kwargs))

    def sent(self, method: str, suffix: str) -> list[dict]:
        return [body for m, path, body in self.requests if m == method and path.endswith(suffix)]

    def pushes(self) -> list[tuple]:
        return [call for call in self.workspace.calls if call[1][0] == "push"]


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Every GitHub request goes to the fake; method, path and JSON body are recorded."""

    def __init__(self, fake: FakeGitHub, seen: list):
        self.inner = ASGITransport(app=fake.app)
        self.seen = seen
        # A test sets it to answer a request itself (a GitHub refusal); None lets the fake answer.
        self.answer: Callable[[httpx.Request], Optional[httpx.Response]] = lambda request: None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.method, request.url.path, json.loads(request.content or b"null")))
        return self.answer(request) or await self.inner.handle_async_request(request)


@pytest.fixture
def lab(tmp_path, monkeypatch) -> Lab:
    fake = FakeGitHub(login=LOGIN, token=TOKEN, root=str(tmp_path / "github"))
    editor = fake.create_repo(EDITOR_NAME, owner=OWNER, default_branch="develop",
                              files={"packages/webapp/src/app.ts": "export {};\n"})
    besser = fake.create_repo("BESSER", owner=OWNER, default_branch="development", files={"README.md": "# B\n"})
    agent = fake.create_repo("modeling-agent", owner=OWNER, default_branch="main", files={"README.md": "# A\n"})

    # git reads $HOME/.gitconfig: the fake remotes stand in for github.com there.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config = ["[user]", "\tname = setup", "\temail = setup@example.com", "[protocol \"file\"]", "\tallow = always"]
    for repo in (editor, besser, agent):
        config += [f"[url \"{Path(repo.path).as_uri()}\"]", f"\tinsteadOf = https://github.com/{repo.full_name}.git"]
    (home / ".gitconfig").write_text("\n".join(config) + "\n", encoding="utf-8")

    work = tmp_path / "work"
    work.mkdir()
    git(work, "clone", "-q", besser.path, "besser")
    git(work / "besser", "submodule", "add", "-q", "-b", "develop", editor.path, FRONTEND_SUBMODULE_PATH)
    git(work / "besser", "commit", "-q", "-m", "add the editor")
    git(work / "besser", "push", "-q", "origin", "development")
    git(work, "clone", "-q", agent.path, "modeling-agent")

    repos = [
        RepoSpec(REPO_FRONTEND, str(work / "besser" / FRONTEND_SUBMODULE_PATH), "develop", OWNER, EDITOR_NAME),
        RepoSpec(REPO_BESSER, str(work / "besser"), "development", OWNER, "BESSER"),
        RepoSpec(REPO_MODELING_AGENT, str(work / "modeling-agent"), "main", OWNER, "modeling-agent"),
    ]
    for spec in repos:
        git(spec.path, "checkout", "-q", "-b", BRANCH)
    change = LabChange(id="c1", slug="note-element", title="Add a note element", user=LOGIN,
                       kind=ChangeKind.EDITOR_ELEMENT, branches={spec.key: BRANCH for spec in repos})

    requests: list = []
    transport = _RecordingTransport(fake, requests)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", transport)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return Lab(fake, RecordingWorkspace(str(work), repos), change, transport)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _editor_and_besser_changes(lab: Lab) -> None:
    _write(lab.path(REPO_FRONTEND) / "packages/webapp/src/note.ts", "export const note = 1;\n")
    _write(lab.path(REPO_BESSER) / "besser/note.py", "NOTE = 1\n")


def test_editor_and_besser_get_linked_draft_pull_requests_editor_first(lab):
    _editor_and_besser_changes(lab)

    opened = lab.open()

    editor_url = f"https://github.com/{OWNER}/{EDITOR_NAME}/pull/1"
    besser_url = f"https://github.com/{OWNER}/BESSER/pull/1"
    assert opened == [PullRequestInfo(REPO_FRONTEND, 1, editor_url, BRANCH, "develop"),
                      PullRequestInfo(REPO_BESSER, 1, besser_url, BRANCH, "development")]
    # The untouched modeling agent gets no branch and no pull request.
    assert lab.fake.ref(lab.fake.repos["besser-pearl/modeling-agent"], BRANCH) is None

    created = lab.sent("POST", "/pulls")
    assert [(c["head"], c["base"], c["draft"], c["title"]) for c in created] == [
        (BRANCH, "develop", True, "feat: add a note element"),
        (BRANCH, "development", True, "feat: add a note element"),
    ]
    # Once both are open, each body lists both, in merge order, and marks itself.
    editor_body, besser_body = (p["body"] for p in lab.sent("PATCH", "/pulls/1"))
    for body, own in ((editor_body, editor_url), (besser_body, besser_url)):
        assert "Merge in this order (the editor first)" in body
        assert body.index(editor_url) < body.index(besser_url)
        assert f"{own} (this pull request)" in body
        assert "`besser/note.py` (added)" in body or "`packages/webapp/src/note.ts` (added)" in body
    assert lab.fake.repos[f"{OWNER}/BESSER".lower()].pulls[0]["body"] == besser_body
    assert all(pull["draft"] for name in (EDITOR_NAME, "BESSER")
               for pull in lab.fake.repos[f"{OWNER}/{name}".lower()].pulls)


def test_besser_branch_points_its_submodule_at_the_pushed_editor_commit(lab):
    _editor_and_besser_changes(lab)

    lab.open()

    fake = lab.fake
    editor_head = fake.ref(fake.repos[f"{OWNER}/{EDITOR_NAME}".lower()], BRANCH)
    besser = fake.repos[f"{OWNER}/BESSER".lower()]
    assert fake.out(besser, "ls-tree", BRANCH, FRONTEND_SUBMODULE_PATH).split()[:3] == \
        ["160000", "commit", editor_head]
    # One commit holds the work, the one on top moves only the pointer, authored as the user.
    assert fake.out(besser, "diff", "--name-only", f"{BRANCH}~1", BRANCH) == FRONTEND_SUBMODULE_PATH
    assert fake.out(besser, "log", "-1", "--format=%an <%ae>", BRANCH) == \
        f"{LOGIN} <{LOGIN}@users.noreply.github.com>"
    assert "besser/note.py" in fake.files(besser, f"{BRANCH}~1")
    # The work-in-progress commit carries the pull request title.
    assert lab.workspace.commit_messages == ["feat: add a note element"]


def test_the_token_reaches_git_only_through_the_push_environment(lab):
    _editor_and_besser_changes(lab)

    lab.open()

    basic = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    assert [call[0] for call in lab.pushes()] == [REPO_FRONTEND, REPO_BESSER]
    authenticated = []
    for repo, args, env in lab.workspace.calls:
        assert not any(TOKEN in arg or basic in arg for arg in args), args
        if env and any(TOKEN in value or basic in value for value in env.values()):
            authenticated.append(args[0])
            # Only in the scratch repository, never with the working copy's own git config.
            assert env == {"GIT_DIR": env["GIT_DIR"], **push_environment(TOKEN)}
            assert not Path(env["GIT_DIR"]).resolve().is_relative_to(lab.path(REPO_BESSER).parent.resolve())
            spec = lab.workspace.repos[repo]
            assert f"https://github.com/{spec.github_owner}/{spec.github_name}.git" in args
    assert authenticated == ["ls-remote", "push"] * 2
    for repo in (REPO_FRONTEND, REPO_BESSER):
        config = git(lab.path(repo), "config", "--list", "--show-origin")
        assert TOKEN not in config and basic not in config


def test_push_environment_is_command_line_config_git_reads(tmp_path):
    git(tmp_path, "init", "-q", "repo")
    env = {**safe_subprocess_env(), **push_environment(TOKEN)}

    def read(key: str) -> str:
        result = subprocess.run(["git", "-C", str(tmp_path / "repo"), "config", "--get", key],
                                capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    header = read("http.https://github.com/.extraheader")
    scheme, encoded = header.removeprefix("AUTHORIZATION: ").split(" ")
    assert scheme == "basic"
    assert base64.b64decode(encoded).decode() == f"x-access-token:{TOKEN}"
    # The last value wins, and an empty one clears any helper the system config sets.
    assert read("credential.helper") == ""
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_opening_again_returns_the_open_pull_requests_without_a_new_pointer_commit(lab):
    _editor_and_besser_changes(lab)
    first = lab.open()
    besser = lab.fake.repos[f"{OWNER}/BESSER".lower()]
    head = lab.fake.ref(besser, BRANCH)

    again = lab.open(title="feat: a custom title")

    assert again == first
    assert lab.fake.ref(besser, BRANCH) == head
    assert len(besser.pulls) == 1


def test_the_push_ignores_url_rewrites_and_proxies_planted_in_the_working_copy_config(lab):
    # The lab agent can write .git/config; the push must not follow it, or the token goes with it.
    _write(lab.path(REPO_BESSER) / "besser/note.py", "NOTE = 1\n")
    evil = lab.fake.create_repo("evil", owner=OWNER, default_branch="development", files={"x": "x\n"})
    besser_url = f"https://github.com/{OWNER}/BESSER.git"
    for key in ("pushInsteadOf", "insteadOf"):
        git(lab.path(REPO_BESSER), "config", f"url.{Path(evil.path).as_uri()}.{key}", besser_url)
    git(lab.path(REPO_BESSER), "config", "http.proxy", "http://127.0.0.1:9")

    opened = lab.open()

    assert [info.repo for info in opened] == [REPO_BESSER]
    assert lab.fake.ref(lab.fake.repos[f"{OWNER}/BESSER".lower()], BRANCH) is not None
    assert lab.fake.ref(evil, BRANCH) is None


def test_commits_added_on_github_are_merged_before_the_next_push(lab):
    _editor_and_besser_changes(lab)
    lab.open()
    besser = lab.fake.repos[f"{OWNER}/BESSER".lower()]
    suggestion = lab.fake.commit_files(besser, {"besser/note.py": "NOTE = 2\n"}, "Apply suggestion", branch=BRANCH,
                                       author="reviewer")
    _write(lab.path(REPO_BESSER) / "besser/more.py", "MORE = 1\n")

    lab.open()

    head = lab.fake.ref(besser, BRANCH)
    assert lab.fake.is_ancestor(besser, suggestion, head)
    files = lab.fake.files(besser, BRANCH)
    assert files["besser/note.py"][1] == b"NOTE = 2\n" and files["besser/more.py"][1].strip() == b"MORE = 1"
    assert (lab.path(REPO_BESSER) / "besser/note.py").read_text(encoding="utf-8") == "NOTE = 2\n"
    assert git(lab.path(REPO_BESSER), "rev-parse", "HEAD") == head


def test_github_commits_that_conflict_or_belong_to_another_change_are_refused_with_a_clear_message(lab):
    _write(lab.path(REPO_BESSER) / "besser/note.py", "NOTE = 1\n")
    lab.open()
    besser = lab.fake.repos[f"{OWNER}/BESSER".lower()]
    lab.fake.commit_files(besser, {"besser/note.py": "NOTE = 2\n"}, "Apply suggestion", branch=BRANCH)
    _write(lab.path(REPO_BESSER) / "besser/note.py", "NOTE = 3\n")

    with pytest.raises(PullRequestError, match="changed on GitHub.*besser/note.py"):
        lab.open()
    assert git(lab.path(REPO_BESSER), "status", "--porcelain", "--untracked-files=no") == ""

    # A branch an earlier change of the same name left on GitHub: never merged into this one.
    unrelated = lab.fake.commit_files(besser, {"old.py": "OLD = 1\n"}, "old change", branch="development")
    lab.fake.set_ref(besser, BRANCH, unrelated)
    with pytest.raises(PullRequestError, match="not part of this change"):
        lab.open()
    assert lab.fake.ref(besser, BRANCH) == unrelated


def test_a_dropped_change_does_not_hand_its_pushed_branch_name_to_the_next_change(lab):
    _write(lab.path(REPO_BESSER) / "besser/note.py", "NOTE = 1\n")
    lab.open()
    for key, spec in lab.workspace.repos.items():
        git(spec.path, "checkout", "-q", spec.base_branch)
        git(spec.path, "branch", "-q", "-D", BRANCH)  # what drop does to the local branches

    change = lab.workspace.start_change(user=LOGIN, title="Note element", kind=ChangeKind.EDITOR_ELEMENT)

    assert change.branches[REPO_BESSER] == BRANCH + "-2"


def test_moving_a_workflow_out_of_its_folder_is_refused_after_the_move_is_committed(lab):
    besser = lab.path(REPO_BESSER)
    git(besser, "checkout", "-q", "development")
    _write(besser / ".github/workflows/ci.yml", "on: push\n")
    git(besser, "add", "-A")
    git(besser, "commit", "-q", "-m", "ci")
    git(besser, "checkout", "-q", BRANCH)
    git(besser, "merge", "-q", "--ff-only", "development")
    (besser / "ci").mkdir()
    git(besser, "mv", ".github/workflows/ci.yml", "ci/ci.yml")
    git(besser, "commit", "-q", "-m", "move")  # repo_changes now lists only ci/ci.yml (renamed)

    with pytest.raises(PullRequestError, match=r"GitHub workflows in besser \(\.github/workflows/ci\.yml\)"):
        lab.open()
    assert lab.pushes() == []


def test_opening_again_keeps_the_descriptions_people_wrote_on_github(lab):
    _editor_and_besser_changes(lab)
    lab.open()
    besser = lab.fake.repos[f"{OWNER}/BESSER".lower()]
    original = besser.pulls[0]["body"]
    # GitHub's web editor sends CRLF line endings.
    related = original[original.index("## Related pull requests"):].replace("\n", "\r\n")
    besser.pulls[0]["body"] = "Hand-written description.\r\n\r\n" + related
    editor_body = lab.fake.repos[f"{OWNER}/{EDITOR_NAME}".lower()].pulls[0]["body"]
    _write(lab.path(REPO_BESSER) / "besser/more.py", "MORE = 1\n")

    lab.open()

    body = besser.pulls[0]["body"]
    assert body.startswith("Hand-written description.")
    assert body.count("## Related pull requests") == 1 and "(this pull request)" in body
    assert "Changed files" not in body
    assert lab.fake.repos[f"{OWNER}/{EDITOR_NAME}".lower()].pulls[0]["body"] == editor_body


def test_a_besser_only_change_opens_one_pull_request_without_related_links(lab):
    _write(lab.path(REPO_BESSER) / "besser/note.py", "NOTE = 1\n")

    opened = lab.open(title="fix: keep notes", body="Why it changed.")

    assert [info.repo for info in opened] == [REPO_BESSER]
    assert lab.sent("PATCH", "/pulls/1") == []
    assert lab.sent("POST", "/pulls")[0]["body"] == "Why it changed."
    besser = lab.fake.repos[f"{OWNER}/BESSER".lower()]
    # No editor change: the submodule pointer stays where development has it.
    assert lab.fake.out(besser, "diff", "--name-only", "development", BRANCH) == "besser/note.py"


@pytest.mark.parametrize("repo, path, content, refusal", [
    (REPO_BESSER, ".github/workflows/ci.yml", "on: push\n", "GitHub workflows"),
    (REPO_FRONTEND, ".github/workflows/ci.yml", "on: push\n", "GitHub workflows"),
    (REPO_MODELING_AGENT, ".env", "OPENAI_API_KEY=sk-proj-Q2xh8bQm4vT7nZr3Kp9Ls1Wd6Yf0Hc5J\n", "secrets"),
])
def test_refuses_before_anything_is_committed_or_pushed(lab, repo, path, content, refusal):
    _editor_and_besser_changes(lab)
    _write(lab.path(repo) / path, content)
    heads = {key: git(lab.path(key), "rev-parse", "HEAD") for key in lab.workspace.repos}

    with pytest.raises(PullRequestError, match=refusal):
        lab.open()

    assert lab.pushes() == []
    assert lab.workspace.commit_messages == []
    assert {key: git(lab.path(key), "rev-parse", "HEAD") for key in lab.workspace.repos} == heads
    assert lab.requests == []


def test_an_env_template_with_placeholders_is_not_a_secret(lab):
    _write(lab.path(REPO_MODELING_AGENT) / ".env.example", "OPENAI_API_KEY=your-key-here\n")

    opened = lab.open()

    assert [info.repo for info in opened] == [REPO_MODELING_AGENT]


def test_refuses_a_branch_that_is_not_a_lab_branch(lab):
    _write(lab.path(REPO_BESSER) / "besser/note.py", "NOTE = 1\n")
    git(lab.path(REPO_BESSER), "checkout", "-q", "-b", "feature/mine")
    lab.change.branches[REPO_BESSER] = "feature/mine"

    with pytest.raises(PullRequestError, match="not a lab branch"):
        lab.open()

    assert lab.pushes() == []


def test_a_change_without_modifications_is_refused(lab):
    with pytest.raises(PullRequestError, match="no modifications"):
        lab.open()

    assert lab.workspace.calls == []


def test_a_failed_push_is_reported_without_the_token(lab):
    _write(lab.path(REPO_BESSER) / "besser/note.py", "NOTE = 1\n")
    run_git = lab.workspace.run_git

    def failing_push(repo, *args, **kwargs):
        if args[0] == "push":
            raise LabWorkspaceError(f"fatal: unable to access 'https://x-access-token:{TOKEN}@github.com/': 403")
        return run_git(repo, *args, **kwargs)

    lab.workspace.run_git = failing_push

    with pytest.raises(PullRequestError) as raised:
        lab.open()

    assert TOKEN not in str(raised.value)
    assert "Pushing the change failed" in str(raised.value) and "403" in str(raised.value)
    assert lab.requests == []


def test_a_github_refusal_names_the_status_and_the_pull_requests_already_open(lab):
    _editor_and_besser_changes(lab)
    lab.transport.answer = lambda request: (
        httpx.Response(403, json={"message": "Resource not accessible by integration"})
        if request.method == "POST" and request.url.path == f"/repos/{OWNER}/BESSER/pulls" else None)

    with pytest.raises(PullRequestError) as raised:
        lab.open()

    message = str(raised.value)
    assert TOKEN not in message
    assert "HTTP 403" in message and "Resource not accessible by integration" in message
    assert f"Already opened: https://github.com/{OWNER}/{EDITOR_NAME}/pull/1." in message


@pytest.mark.parametrize("kind, title, expected", [
    (ChangeKind.GENERATOR, "Add a Rust generator.", "feat: add a Rust generator"),
    (ChangeKind.BUGFIX, "Keep OCL constraints on export", "fix: keep OCL constraints on export"),
    (ChangeKind.METAMODEL, "BPMN pools", "feat: BPMN pools"),
])
def test_default_title_is_a_conventional_commit_subject(kind, title, expected):
    change = LabChange(id="c", slug="s", title=title, user=LOGIN, kind=kind)
    assert default_title(change) == expected
