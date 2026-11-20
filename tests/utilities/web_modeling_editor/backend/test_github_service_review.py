"""GitHubService's review-loop calls against the git-backed GitHub stand-in.

GraphQL, Actions job log tails (302 to blob storage, ``Range``), GitHub's
update-branch and a pull request's diff: each per its docstring, and the token
never in an exception or a log record.
"""

import asyncio
import logging

import httpx
import pytest

from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import (
    GitHubGraphQLError,
    GitHubService,
)
from tests.utilities.web_modeling_editor.backend.fake_github import (
    LOG_HOST,
    MERGE_CONFLICT_MESSAGE,
    STALE_HEAD_MESSAGE,
    FakeGitHub,
    use_fake_github,
)

TOKEN = "ghp_" + "T0kenValue" * 3 + "abcdef"
WRONG_TOKEN = "ghp_" + "Wr0ngValue" * 3 + "abcdef"

PULL_QUERY = """
query LabPullRequest($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) { pullRequest(number: $number) { state headRefOid } }
}
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    fake = FakeGitHub(token=TOKEN, root=str(tmp_path / "github"))
    use_fake_github(monkeypatch, fake)
    return fake


@pytest.fixture
def github():
    return GitHubService(TOKEN)


def _pull(fake: FakeGitHub, github: GitHubService, base_files: dict, head_files: dict):
    """A repo ``ana/app`` with ``feature`` (``head_files`` on top of ``base_files``) open as pull request 1."""
    repo = fake.create_repo("app", files=base_files)
    fake.set_ref(repo, "feature", fake.ref(repo, "main"))
    fake.commit_files(repo, head_files, "Feature work", branch="feature")
    pull = asyncio.run(github.create_pull_request("ana", "app", "feature", "main", "Feature", ""))
    assert pull["number"] == 1
    return repo


# ---------------------------------------------------------------- graphql

def test_graphql_returns_the_data_of_the_lab_pull_request_operation(fake, github):
    repo = _pull(fake, github, {"a.txt": "a\n"}, {"b.txt": "b\n"})
    review = fake.pull_review(repo, 1)
    review.checks.append({"name": "tests", "conclusion": "failure", "job_id": 7, "log": b"boom\n"})
    review.threads.append({"path": "b.txt", "line": 1, "resolved": True,
                           "comments": [{"author": "bo", "association": "MEMBER", "body": "why?"}]})

    data = asyncio.run(github.graphql(PULL_QUERY, {"owner": "ana", "name": "app", "number": 1}))

    pull = data["repository"]["pullRequest"]
    assert pull["state"] == "OPEN"
    assert pull["headRefOid"] == fake.ref(repo, "feature")
    rollup = pull["commits"]["nodes"][0]["commit"]["statusCheckRollup"]
    assert rollup["state"] == "FAILURE"
    assert rollup["contexts"]["nodes"][0]["databaseId"] == 7
    assert pull["reviewThreads"]["nodes"][0]["isResolved"] is True
    assert fake.calls["/graphql"] == 1


def test_graphql_errors_in_the_answer_raise_github_graphql_error(fake, github):
    _pull(fake, github, {"a.txt": "a\n"}, {"b.txt": "b\n"})

    with pytest.raises(GitHubGraphQLError) as caught:
        asyncio.run(github.graphql(PULL_QUERY, {"owner": "ana", "name": "app", "number": 9}))

    assert "Could not resolve to a PullRequest with the number of 9." in str(caught.value)
    assert caught.value.errors[0]["type"] == "NOT_FOUND"
    assert caught.value.data == {"repository": {"pullRequest": None}}


def test_graphql_http_failure_raises_http_status_error(fake, github):
    with pytest.raises(httpx.HTTPStatusError) as caught:
        asyncio.run(github.graphql("query SomethingElse { viewer { login } }", {}))
    assert caught.value.response.status_code == 400


# ---------------------------------------------------------------- job logs

def _repo_with_job_log(fake, github, log):
    repo = _pull(fake, github, {"a.txt": "a\n"}, {"b.txt": "b\n"})
    fake.pull_review(repo, 1).checks.append({"name": "tests", "conclusion": "failure", "job_id": 42, "log": log})
    return repo


def test_job_log_tail_follows_the_redirect_and_reads_only_the_last_bytes(fake, github, caplog):
    log = b"".join(b"line %04d\n" % i for i in range(500))
    _repo_with_job_log(fake, github, log)

    with caplog.at_level(logging.INFO, logger="httpx"):
        tail = asyncio.run(github.get_job_log_tail("ana", "app", 42, max_bytes=100))

    assert tail == log[-100:]
    # The store sent only the range asked for, not the whole log.
    assert any(LOG_HOST in r.getMessage() and "206 Partial Content" in r.getMessage() for r in caplog.records)
    assert fake.calls["/repos/ana/app/actions/jobs/42/logs"] == 1
    # The blob host answered (it refuses a request carrying the GitHub token).
    assert fake.calls["/42"] == 1


def test_job_log_tail_of_a_short_log_is_the_whole_log(fake, github):
    _repo_with_job_log(fake, github, b"only line\n")
    assert asyncio.run(github.get_job_log_tail("ana", "app", 42, max_bytes=65_536)) == b"only line\n"


def test_job_log_tail_of_an_empty_log_is_empty(fake, github):
    _repo_with_job_log(fake, github, b"")
    assert asyncio.run(github.get_job_log_tail("ana", "app", 42, max_bytes=100)) == b""


def test_job_log_tail_of_an_expired_log_is_none(fake, github):
    _repo_with_job_log(fake, github, None)
    assert asyncio.run(github.get_job_log_tail("ana", "app", 42, max_bytes=100)) is None


def test_job_log_tail_of_an_unknown_job_is_none(fake, github):
    _repo_with_job_log(fake, github, b"log\n")
    assert asyncio.run(github.get_job_log_tail("ana", "app", 43, max_bytes=100)) is None


def test_the_fake_log_host_refuses_a_forwarded_token(fake):
    """The check above relies on this: a leaked Authorization header fails the read."""
    async def read():
        async with httpx.AsyncClient() as client:
            return await client.get(f"https://{LOG_HOST}/42", headers={"Authorization": f"Bearer {TOKEN}"})
    assert asyncio.run(read()).status_code == 403


# ---------------------------------------------------------------- update-branch

def test_update_branch_merges_the_base_into_the_head_branch(fake, github):
    repo = _pull(fake, github, {"a.txt": "a\n"}, {"b.txt": "b\n"})
    base_sha = fake.commit_files(repo, {"c.txt": "c\n"}, "Base moved on")
    head_sha = fake.ref(repo, "feature")

    outcome = asyncio.run(github.update_pull_request_branch("ana", "app", 1, head_sha))

    assert outcome == "updated"
    new_head = fake.ref(repo, "feature")
    assert new_head != head_sha
    assert fake.is_ancestor(repo, base_sha, new_head) and fake.is_ancestor(repo, head_sha, new_head)
    assert github.update_branch_message == ""


def test_update_branch_with_nothing_new_on_the_base_is_updated(fake, github):
    repo = _pull(fake, github, {"a.txt": "a\n"}, {"b.txt": "b\n"})
    head_sha = fake.ref(repo, "feature")

    assert asyncio.run(github.update_pull_request_branch("ana", "app", 1, head_sha)) == "updated"
    assert fake.ref(repo, "feature") == head_sha


def test_update_branch_conflict_is_reported_and_leaves_the_branch(fake, github):
    repo = _pull(fake, github, {"a.txt": "a\n"}, {"a.txt": "feature\n"})
    fake.commit_files(repo, {"a.txt": "main\n"}, "Same line on the base")
    head_sha = fake.ref(repo, "feature")

    assert asyncio.run(github.update_pull_request_branch("ana", "app", 1, head_sha)) == "conflict"
    assert fake.ref(repo, "feature") == head_sha
    assert github.update_branch_message == MERGE_CONFLICT_MESSAGE


def test_update_branch_with_a_moved_head_is_refused_with_githubs_message(fake, github):
    repo = _pull(fake, github, {"a.txt": "a\n"}, {"b.txt": "b\n"})
    stale = fake.ref(repo, "feature")
    fake.commit_files(repo, {"b.txt": "someone else\n"}, "Pushed meanwhile", branch="feature")
    fake.commit_files(repo, {"c.txt": "c\n"}, "Base moved on")

    assert asyncio.run(github.update_pull_request_branch("ana", "app", 1, stale)) == "refused"
    assert github.update_branch_message == STALE_HEAD_MESSAGE


def test_update_branch_bringing_a_workflow_change_is_refused(fake, github, caplog):
    repo = _pull(fake, github, {"a.txt": "a\n"}, {"b.txt": "b\n"})
    fake.commit_files(repo, {".github/workflows/ci.yml": "on: push\n"}, "CI")
    fake.refuse_workflow_updates = True
    head_sha = fake.ref(repo, "feature")

    with caplog.at_level(logging.WARNING):
        outcome = asyncio.run(github.update_pull_request_branch("ana", "app", 1, head_sha))

    assert outcome == "refused"
    assert "without `workflow` scope" in github.update_branch_message
    assert fake.ref(repo, "feature") == head_sha
    assert any("workflow" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------- diff

def test_pull_request_diff_is_the_diff_against_the_merge_base(fake, github):
    repo = _pull(fake, github, {"a.txt": "a\n"}, {"b.txt": "b\n"})
    fake.commit_files(repo, {"c.txt": "c\n"}, "Base moved on")

    diff = asyncio.run(github.get_pull_request_diff("ana", "app", 1, max_bytes=100_000))

    assert diff.startswith("diff --git a/b.txt b/b.txt")
    assert "c.txt" not in diff


def test_pull_request_diff_is_cut_at_max_bytes(fake, github, caplog):
    _pull(fake, github, {"a.txt": "a\n"}, {"big.txt": "x" * 5000 + "\n"})
    full = asyncio.run(github.get_pull_request_diff("ana", "app", 1, max_bytes=100_000))

    with caplog.at_level(logging.WARNING):
        cut = asyncio.run(github.get_pull_request_diff("ana", "app", 1, max_bytes=300))

    assert len(cut.encode()) == 300
    assert full.startswith(cut)
    assert any("cut" in record.getMessage() for record in caplog.records)


def test_pull_request_diff_cut_inside_a_character_drops_the_partial_character(fake, github):
    _pull(fake, github, {"a.txt": "a\n"}, {"accents.txt": "é" * 400 + "\n"})
    full = asyncio.run(github.get_pull_request_diff("ana", "app", 1, max_bytes=100_000))
    inside_a_character = full.encode().index("é".encode()) + 1

    cut = asyncio.run(github.get_pull_request_diff("ana", "app", 1, max_bytes=inside_a_character))

    assert "�" not in cut and full.startswith(cut)
    assert len(cut.encode()) == inside_a_character - 1


# ---------------------------------------------------------------- the token stays out

def test_the_token_is_never_in_an_exception_or_a_log_record(fake, caplog):
    github = GitHubService(TOKEN)
    repo = _pull(fake, github, {"a.txt": "a\n"}, {"b.txt": "b\n"})
    fake.pull_review(repo, 1).checks.append({"name": "tests", "conclusion": "failure", "job_id": 5, "log": b"x"})
    fake.commit_files(repo, {".github/workflows/ci.yml": "on: push\n"}, "CI")
    fake.refuse_workflow_updates = True
    wrong = GitHubService(WRONG_TOKEN)
    failures = []

    with caplog.at_level(logging.DEBUG):
        asyncio.run(github.graphql(PULL_QUERY, {"owner": "ana", "name": "app", "number": 1}))
        asyncio.run(github.get_job_log_tail("ana", "app", 5, max_bytes=10))
        asyncio.run(github.update_pull_request_branch("ana", "app", 1, fake.ref(repo, "feature")))
        asyncio.run(github.get_pull_request_diff("ana", "app", 1, max_bytes=10))
        for call in (
            lambda: wrong.graphql(PULL_QUERY, {"owner": "ana", "name": "app", "number": 1}),
            lambda: wrong.get_job_log_tail("ana", "app", 5, max_bytes=10),
            lambda: wrong.update_pull_request_branch("ana", "app", 1, "0" * 40),
            lambda: wrong.get_pull_request_diff("ana", "app", 1, max_bytes=10),
            lambda: github.graphql(PULL_QUERY, {"owner": "ana", "name": "app", "number": 9}),
        ):
            with pytest.raises((httpx.HTTPStatusError, GitHubGraphQLError)) as caught:
                asyncio.run(call())
            failures.append(caught.value)

    assert [type(f).__name__ for f in failures] == ["HTTPStatusError"] * 4 + ["GitHubGraphQLError"]
    for failure in failures:
        assert TOKEN not in f"{failure} {failure!r}" and WRONG_TOKEN not in f"{failure} {failure!r}"
    assert caplog.records
    for record in caplog.records:
        text = record.getMessage() + str(record.args)
        assert TOKEN not in text and WRONG_TOKEN not in text
    assert github.update_branch_message and TOKEN not in github.update_branch_message
