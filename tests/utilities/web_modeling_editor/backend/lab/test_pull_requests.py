"""Draft pull requests of a BESSER Lab change, pushed from its worktrees to a fake GitHub.

The ``lab`` fixture (``conftest.py``) gives a v2 lab workspace whose upstreams are the
``FakeGitHub``'s bare repositories and one open change. The pushes and the pull request
requests run unchanged against them.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import httpx
import pytest

from besser.spec_driven_agent.lab.contracts import (
    DEFAULT_UPSTREAMS,
    FRONTEND_SUBMODULE_PATH,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    ChangeKind,
    IssueRef,
    LabChange,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError, repo_specs_from_settings
from besser.utilities.subprocess_env import safe_subprocess_env
from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabBuildReport,
    LabChangeCost,
    LabChangeView,
    LabCheck,
    LabPlan,
)
from besser.utilities.web_modeling_editor.backend.services.lab.pull_requests import (
    LAB_BLOCK_BEGIN,
    LAB_BLOCK_END,
    PullRequestError,
    PullRequestInfo,
    PullRequestInputError,
    PullRequestSignInError,
    compose_lab_block,
    default_title,
    push_environment,
    with_lab_block,
)
from tests.utilities.web_modeling_editor.backend.lab.conftest import BRANCH, LOGIN, NAMES, OWNER, TOKEN, git

EDITOR_FILE = "packages/webapp/src/note.ts"


def _editor_and_besser_changes(lab) -> None:
    lab.write(REPO_FRONTEND, EDITOR_FILE, "export const note = 1;\n")
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")


def _body(lab, repo: str, number: int = 1) -> str:
    return next(p for p in lab.remote(repo).pulls if p["number"] == number)["body"]


def _last_run(*, report=None, checks=(), summary="") -> LabChangeView:
    return LabChangeView(step="build", changeId="c" * 12, title="t", kind=ChangeKind.EDITOR_ELEMENT, repoChanges=[],
                         checks=list(checks), checklist=[], summary=summary, stopReason="completed", report=report,
                         cost=LabChangeCost(usd=1.25, turns=10, elapsedSeconds=60.0, runs=1))


def test_editor_and_besser_get_linked_draft_pull_requests_editor_first(lab):
    _editor_and_besser_changes(lab)

    opened, note = lab.open()

    editor_url = f"https://github.com/{OWNER}/{NAMES[REPO_FRONTEND]}/pull/1"
    besser_url = f"https://github.com/{OWNER}/BESSER/pull/1"
    assert opened == [PullRequestInfo(REPO_FRONTEND, 1, editor_url, BRANCH, "develop"),
                      PullRequestInfo(REPO_BESSER, 1, besser_url, BRANCH, "development")]
    assert "full test suites did not run" in note
    # The untouched modeling agent gets no branch and no pull request.
    assert lab.fake.ref(lab.remote(REPO_MODELING_AGENT), BRANCH) is None

    created = lab.sent("POST", "/pulls")
    assert [(c["head"], c["base"], c["draft"], c["title"]) for c in created] == [
        (BRANCH, "develop", True, "feat: add a note element"),
        (BRANCH, "development", True, "feat: add a note element"),
    ]
    # Once both are open, each body's lab block lists both, in merge order, and marks itself.
    for repo, own in ((REPO_FRONTEND, editor_url), (REPO_BESSER, besser_url)):
        body = _body(lab, repo)
        assert "Merge in this order (the editor first)" in body
        assert body.index(editor_url) < body.index(besser_url)
        assert f"{own} (this pull request)" in body
        assert body.count(LAB_BLOCK_BEGIN) == 1 and body.endswith(LAB_BLOCK_END)
    assert all(pull["draft"] for repo in (REPO_FRONTEND, REPO_BESSER) for pull in lab.remote(repo).pulls)


def test_opened_pull_requests_are_recorded_on_the_change(lab):
    _editor_and_besser_changes(lab)

    lab.open()

    assert lab.change.pull_requests == {REPO_FRONTEND: 1, REPO_BESSER: 1}
    assert lab.workspace.get_change(lab.change.id).pull_requests == {REPO_FRONTEND: 1, REPO_BESSER: 1}


def test_besser_branch_points_its_submodule_at_the_pushed_editor_commit(lab):
    _editor_and_besser_changes(lab)

    lab.open()

    fake = lab.fake
    editor_head = fake.ref(lab.remote(REPO_FRONTEND), BRANCH)
    besser = lab.remote(REPO_BESSER)
    assert fake.out(besser, "ls-tree", BRANCH, FRONTEND_SUBMODULE_PATH).split()[:3] == \
        ["160000", "commit", editor_head]
    # One commit holds the work, the one on top moves only the pointer, both authored as the user.
    assert fake.out(besser, "diff", "--name-only", f"{BRANCH}~1", BRANCH) == FRONTEND_SUBMODULE_PATH
    for ref in (BRANCH, f"{BRANCH}~1"):
        assert fake.out(besser, "log", "-1", "--format=%an <%ae>", ref) == \
            f"{LOGIN} <{LOGIN}@users.noreply.github.com>"
    assert "besser/note.py" in fake.files(besser, f"{BRANCH}~1")
    # The work-in-progress commit carries the pull request title.
    assert lab.workspace.commits == [("feat: add a note element", LOGIN)]


def test_commits_are_authored_by_the_login_that_pushes(lab):
    # Anyone on the allowed list may push another's change: the commits are theirs (D4 decision 6).
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")

    lab.open(login="ana")

    assert lab.change.owner == LOGIN
    assert lab.fake.out(lab.remote(REPO_BESSER), "log", "-1", "--format=%an <%ae>", BRANCH) == \
        "ana <ana@users.noreply.github.com>"


def test_pull_requests_go_to_the_upstream_the_change_started_from(lab):
    # The change started from a fork; the settings name the canonical repository since: the fork still gets it.
    fork = lab.mirror(REPO_BESSER, "forker")
    fork_settings = {**DEFAULT_UPSTREAMS, REPO_BESSER: "forker/BESSER@development"}
    forked = LabWorkspace(lab.layout.root, repo_specs_from_settings(lab.layout.root, lab.layout.git_root,
                                                                    fork_settings),
                          state_dir=lab.layout.state_dir, git_root=lab.layout.git_root)
    change = lab.new_change(forked, title="Fork work")
    assert change.bases[REPO_BESSER].upstream == "forker/BESSER"
    lab.write(REPO_BESSER, "besser/fork.py", "FORK = 1\n", change=change)

    opened, _ = lab.open(change)

    assert opened == [PullRequestInfo(REPO_BESSER, 1, "https://github.com/forker/BESSER/pull/1", change.branches[
        REPO_BESSER], "development")]
    assert lab.fake.ref(fork, change.branches[REPO_BESSER]) is not None
    assert lab.fake.ref(lab.remote(REPO_BESSER), change.branches[REPO_BESSER]) is None
    assert lab.remote(REPO_BESSER).pulls == []


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
            assert not Path(env["GIT_DIR"]).resolve().is_relative_to(Path(lab.layout.home).resolve())
            assert f"https://github.com/{OWNER}/{NAMES[repo]}.git" in args
    assert authenticated == ["ls-remote", "push"] * 2
    for repo in (REPO_FRONTEND, REPO_BESSER):
        config = git(str(lab.path(repo)), "config", "--list", "--show-origin")
        assert TOKEN not in config and basic not in config


def test_push_environment_is_command_line_config_git_reads(tmp_path):
    git(str(tmp_path), "init", "-q", "repo")
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
    first, _ = lab.open()
    besser = lab.remote(REPO_BESSER)
    head = lab.fake.ref(besser, BRANCH)

    again, _ = lab.open(title="feat: a custom title")

    assert again == first
    assert lab.fake.ref(besser, BRANCH) == head
    assert len(besser.pulls) == 1


def test_the_push_ignores_url_rewrites_and_proxies_planted_in_the_working_copy_config(lab):
    # The lab agent can write the tree's git config; the push must not follow it, or the token goes with it.
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    evil = lab.fake.create_repo("evil", owner=OWNER, default_branch="development", files={"x": "x\n"})
    besser_url = f"https://github.com/{OWNER}/BESSER.git"
    tree = str(lab.path(REPO_BESSER))
    for key in ("pushInsteadOf", "insteadOf"):
        lab.workspace.run_git(REPO_BESSER, "config", f"url.{Path(evil.path).as_uri()}.{key}", besser_url,
                              change=lab.change)
    lab.workspace.run_git(REPO_BESSER, "config", "http.proxy", "http://127.0.0.1:9", change=lab.change)
    assert "insteadof" in git(tree, "config", "--list").lower()

    opened, _ = lab.open()

    assert [info.repo for info in opened] == [REPO_BESSER]
    assert lab.fake.ref(lab.remote(REPO_BESSER), BRANCH) is not None
    assert lab.fake.ref(evil, BRANCH) is None


def test_commits_added_on_github_are_merged_before_the_next_push(lab):
    _editor_and_besser_changes(lab)
    lab.open()
    besser = lab.remote(REPO_BESSER)
    suggestion = lab.fake.commit_files(besser, {"besser/note.py": "NOTE = 2\n"}, "Apply suggestion", branch=BRANCH,
                                       author="reviewer")
    lab.write(REPO_BESSER, "besser/more.py", "MORE = 1\n")

    lab.open()

    head = lab.fake.ref(besser, BRANCH)
    assert lab.fake.is_ancestor(besser, suggestion, head)
    files = lab.fake.files(besser, BRANCH)
    assert files["besser/note.py"][1] == b"NOTE = 2\n" and files["besser/more.py"][1].strip() == b"MORE = 1"
    assert (lab.path(REPO_BESSER) / "besser/note.py").read_text(encoding="utf-8") == "NOTE = 2\n"
    assert lab.workspace.run_git(REPO_BESSER, "rev-parse", "HEAD", change=lab.change).strip() == head


def test_github_commits_that_conflict_or_belong_to_another_change_are_refused_with_a_clear_message(lab):
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    lab.open()
    besser = lab.remote(REPO_BESSER)
    lab.fake.commit_files(besser, {"besser/note.py": "NOTE = 2\n"}, "Apply suggestion", branch=BRANCH)
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 3\n")

    with pytest.raises(PullRequestError, match="changed on GitHub.*besser/note.py"):
        lab.open()
    status = lab.workspace.run_git(REPO_BESSER, "status", "--porcelain", "--untracked-files=no", change=lab.change)
    assert status.strip() == ""

    # A branch an earlier change of the same name left on GitHub: never merged into this one.
    unrelated = lab.fake.commit_files(besser, {"old.py": "OLD = 1\n"}, "old change", branch="development")
    lab.fake.set_ref(besser, BRANCH, unrelated)
    with pytest.raises(PullRequestError, match="not part of this change"):
        lab.open()
    assert lab.fake.ref(besser, BRANCH) == unrelated


def test_a_dropped_change_does_not_hand_its_pushed_branch_name_to_the_next_change(lab):
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    lab.open()
    lab.workspace.drop(lab.change)

    change = lab.new_change()

    assert change.branches[REPO_BESSER] == BRANCH + "-2"


def test_moving_a_workflow_out_of_its_folder_is_refused_after_the_move_is_committed(lab):
    lab.fake.commit_files(lab.remote(REPO_BESSER), {".github/workflows/ci.yml": "on: push\n"}, "ci",
                          branch="development")
    lab.workspace.refresh_bases()
    assert lab.workspace.begin_base_merge(lab.change, REPO_BESSER) == []
    tree = str(lab.path(REPO_BESSER))
    (lab.path(REPO_BESSER) / "ci").mkdir()
    # Moved on disk, as the sandboxed agent does (it cannot write the git directory).
    (lab.path(REPO_BESSER) / ".github/workflows/ci.yml").rename(lab.path(REPO_BESSER) / "ci/ci.yml")
    lab.workspace.commit_all(lab.change, "move", login=LOGIN)  # repo_changes now lists only ci/ci.yml
    assert "ci/ci.yml" in git(tree, "ls-files")

    with pytest.raises(PullRequestError, match=r"GitHub workflows in besser \(\.github/workflows/ci\.yml\)"):
        lab.open()
    assert lab.pushes() == []


def test_second_push_rewrites_the_lab_block_and_keeps_reviewer_text(lab):
    _editor_and_besser_changes(lab)
    lab.open(last=_last_run(report=LabBuildReport(summary="First build.", deviations=[], notDone=[])))
    besser = lab.remote(REPO_BESSER)
    first = besser.pulls[0]["body"]
    # A reviewer writes above and below the block on GitHub (its web editor sends CRLF).
    besser.pulls[0]["body"] = ("Reviewer notes on top.\r\n\r\n" + first.replace("\n", "\r\n")
                               + "\r\n\r\nA reviewer's footer.")
    lab.write(REPO_BESSER, "besser/more.py", "MORE = 1\n")

    lab.open(last=_last_run(report=LabBuildReport(summary="Second build: more.py added.", deviations=[],
                                                  notDone=[])))

    body = besser.pulls[0]["body"]
    assert body.startswith("Reviewer notes on top.") and body.endswith("A reviewer's footer.")
    assert "Second build: more.py added." in body and "First build." not in body
    assert body.count(LAB_BLOCK_BEGIN) == 1 and body.count(LAB_BLOCK_END) == 1
    assert "(this pull request)" in body


def test_phase1_body_without_markers_gets_the_block_appended_once(lab):
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    lab.open()
    besser = lab.remote(REPO_BESSER)
    besser.pulls[0]["body"] = ("Hand-written description.\r\n\r\n## Related pull requests\r\n\r\n"
                               "Merge in this order (the editor first):\r\n\r\n1. besser: https://x (this pull request)")

    lab.open()
    lab.open()

    body = besser.pulls[0]["body"]
    assert body.startswith("Hand-written description.\n\n" + LAB_BLOCK_BEGIN)
    assert body.count(LAB_BLOCK_BEGIN) == 1
    assert "## Related pull requests" not in body


def test_body_has_report_plan_deviations_skipped_checks_and_no_cost(lab):
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    plan = LabPlan(kind=ChangeKind.EDITOR_ELEMENT, title="Note", summary="Add a note element to the editor.",
                   steps=["Add the element", "Register it"], repos=["besser"], questions=[])
    report = LabBuildReport(summary="Added the note element.", deviations=["Kept the old icon"],
                            notDone=["No German translation"])
    checks = [LabCheck(id="py", repo="besser", title="BESSER tests", passed=True, skipped=False, summary=""),
              LabCheck(id="docs", repo="besser", title="Docs build", passed=False, skipped=True, summary="no sphinx"),
              LabCheck(id="lint", repo="besser", title="Ruff", passed=False, skipped=False, summary="E1")]

    lab.open(plan=plan, last=_last_run(report=report, checks=checks), model="claude-sonnet-5",
             requested_by=[LOGIN, "ana"])

    body = _body(lab, REPO_BESSER)
    for text in ("Added the note element.", "Kept the old icon", "No German translation",
                 "Add a note element to the editor.", "1. Add the element", "2. Register it",
                 "Passed: BESSER tests", "Failed: Ruff", "Skipped: Docs build", "full test suites did not run",
                 "Requested in BESSER Lab by @jordi, @ana, built by claude-sonnet-5."):
        assert text in body, text
    assert "$" not in body and "1.25" not in body and "USD" not in body


def test_a_build_without_a_report_puts_its_closing_text_in_the_block(lab):
    block = compose_lab_block(lab.change, None, _last_run(summary="I added the note."), [], "m", [LOGIN])

    assert "The lab agent gave no structured account" in block and "I added the note." in block


def test_closes_only_confirmed_issues(lab):
    lab.change.issues = [{"repo": "besser", "number": 744, "title": "Notes", "closes": False},
                         {"repo": "frontend", "number": 12, "title": "Icon", "closes": False}]
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")

    lab.open(closes_issues=[IssueRef("besser", 744)])

    body = _body(lab, REPO_BESSER)
    assert "Closes BESSER-PEARL/BESSER#744" in body
    assert "Related to BESSER-PEARL/BESSER-Web-Modeling-Editor#12" in body
    assert "Closes BESSER-PEARL/BESSER-Web-Modeling-Editor#12" not in body
    stored = lab.workspace.get_change(lab.change.id).issues
    assert [issue["closes"] for issue in stored] == [True, False]

    # A later push without ticks keeps the stored ones.
    lab.write(REPO_BESSER, "besser/more.py", "MORE = 1\n")
    lab.open()
    assert "Closes BESSER-PEARL/BESSER#744" in _body(lab, REPO_BESSER)


def test_closing_an_issue_the_change_did_not_start_from_is_refused(lab):
    lab.change.issues = [{"repo": "besser", "number": 744, "title": "Notes", "closes": False}]
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")

    with pytest.raises(PullRequestInputError, match="besser#745 is not an issue of this change"):
        lab.open(closes_issues=[IssueRef("besser", 745)])
    assert lab.pushes() == [] and lab.requests == []


def test_with_lab_block_replaces_only_the_block():
    block = f"{LAB_BLOCK_BEGIN}\nnew\n{LAB_BLOCK_END}"

    assert with_lab_block("", block) == block
    assert with_lab_block(f"top\n{LAB_BLOCK_BEGIN}\nold\n{LAB_BLOCK_END}\nbottom", block) == f"top\n{block}\nbottom"


def test_a_besser_only_change_opens_one_pull_request_without_related_links(lab):
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")

    opened, _ = lab.open(title="fix: keep notes", body="Why it changed.")

    assert [info.repo for info in opened] == [REPO_BESSER]
    assert lab.sent("PATCH", "/pulls/1") == []
    sent = lab.sent("POST", "/pulls")[0]["body"]
    assert sent.startswith("Why it changed.\n\n" + LAB_BLOCK_BEGIN) and "Related pull requests" not in sent
    besser = lab.remote(REPO_BESSER)
    # No editor change: the submodule pointer stays where development has it.
    assert lab.fake.out(besser, "diff", "--name-only", "development", BRANCH) == "besser/note.py"


@pytest.mark.parametrize("repo, path, content, refusal", [
    (REPO_BESSER, ".github/workflows/ci.yml", "on: push\n", "GitHub workflows"),
    (REPO_FRONTEND, ".github/workflows/ci.yml", "on: push\n", "GitHub workflows"),
    (REPO_MODELING_AGENT, ".env", "OPENAI_API_KEY=sk-proj-Q2xh8bQm4vT7nZr3Kp9Ls1Wd6Yf0Hc5J\n", "secrets"),
])
def test_refuses_before_anything_is_committed_or_pushed(lab, repo, path, content, refusal):
    _editor_and_besser_changes(lab)
    lab.write(repo, path, content)
    keys = (REPO_FRONTEND, REPO_BESSER, REPO_MODELING_AGENT)
    heads = {key: lab.workspace.run_git(key, "rev-parse", "HEAD", change=lab.change) for key in keys}

    with pytest.raises(PullRequestError, match=refusal):
        lab.open()

    assert lab.pushes() == []
    assert lab.workspace.commits == []
    assert {key: lab.workspace.run_git(key, "rev-parse", "HEAD", change=lab.change) for key in keys} == heads
    assert lab.requests == []


def test_an_env_template_with_placeholders_is_not_a_secret(lab):
    lab.write(REPO_MODELING_AGENT, ".env.example", "OPENAI_API_KEY=your-key-here\n")

    opened, _ = lab.open()

    assert [info.repo for info in opened] == [REPO_MODELING_AGENT]


def test_refuses_a_branch_that_is_not_a_lab_branch(lab):
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    lab.workspace.run_git(REPO_BESSER, "checkout", "-q", "-b", "feature/mine", change=lab.change)
    lab.change.branches[REPO_BESSER] = "feature/mine"

    with pytest.raises(PullRequestError, match="not a lab branch"):
        lab.open()

    assert lab.pushes() == []


def test_a_change_without_modifications_is_refused(lab):
    with pytest.raises(PullRequestError, match="no modifications"):
        lab.open()

    assert lab.workspace.calls == []


def test_a_failed_push_is_reported_without_the_token(lab):
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
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


# git's own words for each failure a user can act on (E2E A-6: the first came from a live run with a bad token).
PUSH_FAILURES = [
    ("fatal: could not read Username for 'https://github.com': terminal prompts disabled", "Sign in to GitHub again"),
    ("remote: Invalid username or password.\nfatal: Authentication failed for "
     "'https://github.com/BESSER-PEARL/BESSER.git/'", "Sign in to GitHub again"),
    ("remote: Permission to BESSER-PEARL/BESSER.git denied to jordi.\nfatal: unable to access "
     "'https://github.com/BESSER-PEARL/BESSER.git/': The requested URL returned error: 403",
     "Ask a maintainer of BESSER-PEARL/BESSER for write access"),
    ("remote: error: GH006: Protected branch update failed for refs/heads/lab/jordi/add-a-note-element.\n"
     " ! [remote rejected] lab/jordi/add-a-note-element -> lab/jordi/add-a-note-element (protected branch hook "
     "declined)", "branch protection"),
    ("fatal: unable to access 'https://github.com/BESSER-PEARL/BESSER.git/': Could not resolve host: github.com",
     "could not reach GitHub"),
    ("git push timed out after 300 s in besser", "could not reach GitHub"),
]


@pytest.mark.parametrize("failing", ["ls-remote", "push"])
@pytest.mark.parametrize("stderr, advice", PUSH_FAILURES)
def test_a_failed_push_tells_the_user_what_to_do_and_logs_gits_words(lab, caplog, failing, stderr, advice):
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    run_git = lab.workspace.run_git

    def failing_git(repo, *args, **kwargs):
        if args[0] == failing:
            if "timed out" in stderr:
                raise LabWorkspaceError(stderr)
            raise LabWorkspaceError(f"git {failing} failed in {repo} (exit 128): {stderr} {TOKEN}")
        return run_git(repo, *args, **kwargs)

    lab.workspace.run_git = failing_git

    with pytest.raises(PullRequestError) as raised:
        lab.open()

    message = str(raised.value)
    assert advice in message
    # Only a refused sign-in is one: the route answers it with 401, so the editor offers "Sign in again".
    assert isinstance(raised.value, PullRequestSignInError) == (advice == "Sign in to GitHub again")
    # git's words stay in the server log, without the token.
    assert "fatal" not in message and "git " not in message and "exit 128" not in message
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert stderr.splitlines()[-1] in logged
    assert TOKEN not in message and TOKEN not in logged


def test_a_rejected_token_on_the_github_api_asks_to_sign_in_again(lab, caplog):
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    lab.transport.answer = lambda request: (
        httpx.Response(401, json={"message": "Bad credentials"}) if request.method == "POST" else None)

    with pytest.raises(PullRequestError) as raised:
        lab.open()

    assert "Sign in to GitHub again" in str(raised.value) and "HTTP 401" not in str(raised.value)
    assert isinstance(raised.value, PullRequestSignInError)
    assert "Bad credentials" in "\n".join(record.getMessage() for record in caplog.records)


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
    assert f"Already opened: https://github.com/{OWNER}/{NAMES[REPO_FRONTEND]}/pull/1." in message
    # What was opened is recorded, so the next push finds it.
    assert lab.workspace.get_change(lab.change.id).pull_requests == {REPO_FRONTEND: 1}


@pytest.mark.parametrize("kind, title, expected", [
    (ChangeKind.GENERATOR, "Add a Rust generator.", "feat: add a Rust generator"),
    (ChangeKind.BUGFIX, "Keep OCL constraints on export", "fix: keep OCL constraints on export"),
    (ChangeKind.METAMODEL, "BPMN pools", "feat: BPMN pools"),
])
def test_default_title_is_a_conventional_commit_subject(kind, title, expected):
    change = LabChange(id="c" * 12, slug="s", title=title, owner=LOGIN, kind=kind)
    assert default_title(change) == expected
