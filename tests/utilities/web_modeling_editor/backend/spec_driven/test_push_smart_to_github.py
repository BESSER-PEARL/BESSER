"""HTTP-level tests for POST /besser_api/spec-driven/push-to-github.

The endpoint pushes the *stored* artifact of a finished spec-driven
run (by ``run_id``) plus the re-importable model source to GitHub. These
tests mock ``GitHubService`` and seed ``SMART_RUN_REGISTRY`` with a fake
``SmartRunEntry`` pointing at a temp dir we build, so nothing touches the
real GitHub API.
"""

import asyncio
import os
import tempfile
import time

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import (
    spec_driven_router as router_module,
)
from besser.utilities.web_modeling_editor.backend.services.deployment import repo_model
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    CARRIED_RUN_FILES,
    SMART_RUN_REGISTRY,
    SmartRunEntry,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _clear_registry,
)


BASE_URL = "http://testserver"
PUSH_URL = "/besser_api/spec-driven/push-to-github"


@pytest.fixture(autouse=True)
def reset_registry():
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


# ---------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------


class _FakeGitHubService:
    """Records calls and snapshots the pushed directory tree.

    ``push_directory_to_repo`` walks the directory it is handed *at call
    time* — the real temp workdir is deleted when the endpoint's
    ``TemporaryDirectory`` context exits, so the snapshot must happen
    during the call, which is exactly what the endpoint does in practice.
    """

    def __init__(self, *, branches=None, get_branches_404=False, default_branch="main", repo_files=None):
        self._branches = ["main"] if branches is None else branches
        self._repo_files = repo_files or {}
        self._default_branch = default_branch
        self._get_branches_404 = get_branches_404
        self.created = False
        self.created_kwargs = None
        self.push = None

    async def get_authenticated_user(self):
        return {"login": "test-owner"}

    async def create_repository(
        self, repo_name, description="", is_private=False, auto_init=False
    ):
        self.created = True
        self.created_kwargs = dict(
            repo_name=repo_name,
            description=description,
            is_private=is_private,
            auto_init=auto_init,
        )
        return {"default_branch": "main"}

    async def get_branches(self, owner, repo_name, per_page=100):
        if self._get_branches_404:
            request = httpx.Request(
                "GET", "https://api.github.com/repos/test-owner/x/branches"
            )
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError(
                "Not Found", request=request, response=response
            )
        return self._branches

    async def get_default_branch(self, owner, repo_name):
        await self.get_branches(owner, repo_name)  # same 404 when the repo is missing
        return self._default_branch

    async def get_branch_head(self, owner, repo_name, branch):
        return "a" * 40

    async def get_file_content(self, owner, repo_name, file_path, branch="main", ref=None):
        content = self._repo_files.get(file_path)
        return None if content is None else {"content": content}

    async def add_topic(self, owner, repo_name, topic):
        self.topics = getattr(self, "topics", []) + [topic]

    async def create_pull_request(self, owner, repo_name, head, base, title, body):
        self.pull_request = dict(head=head, base=base, title=title)
        return {"number": 7, "html_url": f"https://github.com/{owner}/{repo_name}/pull/7"}

    async def push_directory_to_repo(
        self,
        owner,
        repo_name,
        directory_path,
        commit_message="",
        branch="main",
        preserve_existing_files=False,
        base_commit_sha=None,
        keep_path=None,
        model_merge=None,
        base_hashes=None,
    ):
        files = []
        for root, _dirs, names in os.walk(directory_path):
            for name in names:
                rel = os.path.relpath(
                    os.path.join(root, name), directory_path
                ).replace("\\", "/")
                files.append(rel)
        self.push = dict(
            owner=owner,
            repo_name=repo_name,
            directory_path=directory_path,
            files=sorted(files),
            commit_message=commit_message,
            branch=branch,
            preserve_existing_files=preserve_existing_files,
            base_commit_sha=base_commit_sha,
        )
        readme = os.path.join(directory_path, "README.md")
        if os.path.isfile(readme):
            with open(readme, encoding="utf-8", newline="") as fh:
                self.push["readme"] = fh.read()
        gitattributes = os.path.join(directory_path, ".gitattributes")
        if os.path.isfile(gitattributes):
            with open(gitattributes, encoding="utf-8") as fh:
                self.push["gitattributes"] = fh.read()
        return {
            "commit_sha": "d" * 40,
            "total_files": len(files),
            "uploaded_files": files,
            "status": "pushed",
            "conflict_branch": None,
            "own_commit_sha": "d" * 40,
        }


def _install(monkeypatch, fake):
    """Patch the GitHub factory + token lookup in the router namespace."""
    monkeypatch.setattr(router_module, "create_github_service", lambda token: fake)
    monkeypatch.setattr(router_module, "get_user_token", lambda sid: "fake-token")


def _project_export() -> dict:
    """A re-importable V2 project-export envelope (ProjectInput shape)."""
    return {
        "id": "test-project",
        "type": "BesserProject",
        "name": "TestProject",
        "createdAt": "2026-04-15T00:00:00Z",
        "diagrams": {
            "ClassDiagram": [
                {"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}
            ]
        },
        "currentDiagramIndices": {"ClassDiagram": 0},
    }


def _seed_run(run_id: str, *, with_secret_env: bool = True) -> str:
    """Create a fake generated file tree and register it under ``run_id``."""
    tmp = tempfile.mkdtemp(prefix="besser_smart_push_test_")

    with open(os.path.join(tmp, "main.py"), "w", encoding="utf-8") as fh:
        fh.write("# generated\nprint('hi')\n")

    # Internal artifact that must NOT be pushed.
    with open(os.path.join(tmp, ".besser_recipe.json"), "w", encoding="utf-8") as fh:
        fh.write('{"instructions": "x"}')
    # Line attribution state: rides along for continue-from-GitHub (#620).
    with open(os.path.join(tmp, ".besser_attribution.json"), "w", encoding="utf-8") as fh:
        fh.write('{"version": 1, "contributors": [], "files": {}}')
    # The generator's stored output: the base of the next model change's merge (#659).
    with open(os.path.join(tmp, ".besser_generator_output.json"), "w", encoding="utf-8") as fh:
        fh.write('{"format": 1, "out_dir": "backend", "files": {}}')
    os.makedirs(os.path.join(tmp, ".agent-trace"), exist_ok=True)
    with open(os.path.join(tmp, ".agent-trace", "run.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")

    # Build/dependency dir that must be skipped.
    nm = os.path.join(tmp, "node_modules", "left-pad")
    os.makedirs(nm, exist_ok=True)
    with open(os.path.join(nm, "index.js"), "w", encoding="utf-8") as fh:
        fh.write("module.exports = 1;\n")

    if with_secret_env:
        # Real-looking secret → must be scrubbed.
        with open(os.path.join(tmp, ".env"), "w", encoding="utf-8") as fh:
            fh.write("OPENAI_API_KEY=sk-ant-REALSECRET0123456789abcdefghij\n")
        # Template → must be preserved.
        with open(os.path.join(tmp, ".env.example"), "w", encoding="utf-8") as fh:
            fh.write("OPENAI_API_KEY=your-key-here\n")

    entry = SmartRunEntry(
        file_path=os.path.join(tmp, "main.py"),
        file_name="main.py",
        is_zip=False,
        temp_dir=tmp,
        created_at=time.time(),
    )
    asyncio.run(SMART_RUN_REGISTRY.put(run_id, entry))
    return tmp


def _post(body: dict, headers: dict | None = None) -> httpx.Response:
    transport = ASGITransport(app=app)

    async def _run() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
            return await ac.post(PUSH_URL, json=body, headers=headers or {})

    return asyncio.run(_run())


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestPushSmartToGitHub:
    def test_valid_run_pushes_and_returns_repo_url(self, monkeypatch):
        run_id = "a" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-generated-app", "is_private": True},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["repo_url"] == "https://github.com/test-owner/my-generated-app"
        assert data["owner"] == "test-owner"
        assert data["files_uploaded"] >= 1
        assert data["is_first_push"] is True
        # A fresh (default-private) repo was created.
        assert fake.created is True
        assert fake.created_kwargs["is_private"] is True

    def test_unknown_run_id_returns_404_run_expired(self, monkeypatch):
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {
            "run_id": "0" * 32,  # never seeded
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-generated-app"},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 404
        assert r.json()["detail"] == "run_expired"

    def test_missing_github_session_returns_401(self, monkeypatch):
        run_id = "b" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-generated-app"},
        }
        r = _post(body)  # no X-GitHub-Session header
        assert r.status_code == 401

    def test_model_and_diagrams_land_in_pushed_dir(self, monkeypatch):
        run_id = "c" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-generated-app"},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200

        files = fake.push["files"]
        # Model source injected under buml/.
        assert "buml/diagrams.json" in files
        assert "buml/domain_model.py" in files
        # The stored generated code is present.
        assert "main.py" in files
        # The recipe rides along so continue-from-GitHub re-hydrates the
        # seed's generator + history; other internals and build dirs stay out.
        assert ".besser_recipe.json" in files
        assert ".besser_attribution.json" in files
        assert ".besser_generator_output.json" in files
        assert ".agent-trace/run.json" in files
        assert not any(f.startswith(".besser_checkpoint") for f in files)
        assert not any(f.startswith(".besser_snapshot") for f in files)
        assert not any(f.startswith("node_modules/") for f in files)
        # Secret .env scrubbed; template preserved.
        assert ".env" not in files
        assert ".env.example" in files

    def test_use_existing_pushes_requested_branch_keeping_files_the_run_never_saw(self, monkeypatch):
        run_id = "d" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService(branches=["main", "dev"])
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {
                "repo_name": "my-generated-app",
                "use_existing": True,
                "branch": "dev",
            },
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200
        data = r.json()
        assert data["is_first_push"] is False
        # Reused, not created.
        assert fake.created is False
        assert fake.push["branch"] == "dev"
        # The run did not start from this repo, so it cannot have deleted any of
        # its files: it adds and updates, and e.g. the repo's README stays.
        assert fake.push["preserve_existing_files"] is True

    def test_without_a_requested_branch_the_push_lands_on_the_repos_real_default(self, monkeypatch):
        """The default was guessed from the branch list, so a leftover sync branch sorted first could win."""
        run_id = "f" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService(branches=["besser/sync-1", "develop"], default_branch="develop")
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-generated-app", "use_existing": True},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200, r.text
        assert fake.push["branch"] == "develop"

    def test_use_existing_missing_repo_returns_404_repo_missing(self, monkeypatch):
        run_id = "e" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService(get_branches_404=True)
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-generated-app", "use_existing": True},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 404
        assert r.json()["detail"] == "repo_missing"
        # Must NOT silently create a repo when the user asked to reuse one.
        assert fake.created is False

    def test_push_links_the_readme_back_to_besser_and_the_model(self, monkeypatch):
        run_id = "1" * 32
        tmp = _seed_run(run_id)
        with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# Library\n\nRun `python main.py`.\n")
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-generated-app"},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200, r.text
        readme = fake.push["readme"]
        assert readme.startswith("# Library\n\nRun `python main.py`.\n\n<!-- besser:begin -->")
        assert ("https://editor.besser-pearl.org/?buml=https://github.com/test-owner/my-generated-app/"
                "blob/main/buml/domain_model.py") in readme
        assert "https://github.com/BESSER-PEARL/Built-with-BESSER" in readme
        # The stored run keeps the agent's README; only the pushed copy carries the block.
        with open(os.path.join(tmp, "README.md"), encoding="utf-8") as fh:
            assert "besser:begin" not in fh.read()

    def test_push_without_a_readme_adds_one_named_after_the_repo(self, monkeypatch):
        run_id = "2" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-generated-app"},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200, r.text
        assert fake.push["readme"].startswith("# my-generated-app\n\n<!-- besser:begin -->")
        assert "README.md" in fake.push["files"]

    def test_the_run_records_are_marked_generated_so_github_folds_their_diffs(self, monkeypatch):
        run_id = "f" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {"run_id": run_id, "projectExport": _project_export(),
                "deploy_config": {"repo_name": "my-generated-app"}}
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200, r.text
        assert ".gitattributes" in fake.push["files"]
        assert fake.push["gitattributes"] == (
            "/.besser_recipe.json linguist-generated=true\n"
            "/.besser_attribution.json linguist-generated=true\n"
            "/.besser_generator_output.json linguist-generated=true\n"
        )

    def test_a_push_to_an_existing_repo_keeps_the_repos_own_gitattributes(self, monkeypatch):
        # The run did not start from this repo, so it holds no .gitattributes; pushing a
        # fresh one would replace the repo's.
        run_id = "e" * 32
        _seed_run(run_id)
        # The owner's own setting for one record is kept; a line that does not set it does not count.
        fake = _FakeGitHubService(repo_files={".gitattributes": (
            "*.png binary\n/.besser_recipe.json -linguist-generated\n/.besser_attribution.json -diff\n")})
        _install(monkeypatch, fake)

        body = {"run_id": run_id, "projectExport": _project_export(),
                "deploy_config": {"repo_name": "my-generated-app", "use_existing": True}}
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200, r.text
        assert fake.push["gitattributes"] == (
            "*.png binary\n"
            "/.besser_recipe.json -linguist-generated\n"
            "/.besser_attribution.json -diff\n"
            "/.besser_attribution.json linguist-generated=true\n"
            "/.besser_generator_output.json linguist-generated=true\n"
        )


class TestMarkGenerated:
    def _workdir(self, tmp_path, *names):
        for name in names:
            (tmp_path / name).write_text("{}", encoding="utf-8")
        return str(tmp_path)

    def _gitattributes(self, tmp_path):
        return (tmp_path / ".gitattributes").read_text(encoding="utf-8")

    def test_only_records_in_the_tree_are_marked(self, tmp_path):
        workdir = self._workdir(tmp_path, ".besser_recipe.json")
        repo_model.mark_generated(workdir, CARRIED_RUN_FILES)
        assert self._gitattributes(tmp_path) == "/.besser_recipe.json linguist-generated=true\n"

    def test_a_second_push_adds_nothing(self, tmp_path):
        workdir = self._workdir(tmp_path, *CARRIED_RUN_FILES)
        repo_model.mark_generated(workdir, CARRIED_RUN_FILES)
        first = self._gitattributes(tmp_path)
        repo_model.mark_generated(workdir, CARRIED_RUN_FILES)
        assert self._gitattributes(tmp_path) == first

    def test_the_runs_own_file_keeps_its_lines_and_wins_over_the_repos(self, tmp_path):
        workdir = self._workdir(tmp_path, *CARRIED_RUN_FILES)
        (tmp_path / ".gitattributes").write_text(
            "# mine\n* text=auto\n.besser_attribution.json linguist-generated=false", encoding="utf-8",
        )
        repo_model.mark_generated(workdir, CARRIED_RUN_FILES, repo_text="*.zip binary\n")
        assert self._gitattributes(tmp_path) == (
            "# mine\n* text=auto\n.besser_attribution.json linguist-generated=false\n"
            "/.besser_recipe.json linguist-generated=true\n"
            "/.besser_generator_output.json linguist-generated=true\n"
        )

    def test_without_records_or_a_repo_file_no_file_is_written(self, tmp_path):
        repo_model.mark_generated(str(tmp_path), CARRIED_RUN_FILES)
        assert not (tmp_path / ".gitattributes").exists()

    def test_the_repos_file_is_kept_even_when_nothing_is_added(self, tmp_path):
        # The push would otherwise delete it as a file the run dropped.
        repo_model.mark_generated(str(tmp_path), CARRIED_RUN_FILES, repo_text="*.zip binary\r\n")
        assert (tmp_path / ".gitattributes").read_bytes() == b"*.zip binary\r\n"

    def test_a_glob_line_setting_the_attribute_is_respected(self, tmp_path):
        workdir = self._workdir(tmp_path, *CARRIED_RUN_FILES)
        repo_model.mark_generated(workdir, CARRIED_RUN_FILES, repo_text="* text=auto\n.besser_* -linguist-generated\n")
        assert self._gitattributes(tmp_path) == "* text=auto\n.besser_* -linguist-generated\n"

    def test_lines_added_to_a_crlf_file_use_crlf(self, tmp_path):
        workdir = self._workdir(tmp_path, ".besser_recipe.json")
        repo_model.mark_generated(workdir, CARRIED_RUN_FILES, repo_text="*.png binary\r\n")
        assert (tmp_path / ".gitattributes").read_bytes() == (
            b"*.png binary\r\n/.besser_recipe.json linguist-generated=true\r\n")

    def test_a_linked_gitattributes_is_left_alone(self, tmp_path):
        workdir = self._workdir(tmp_path, *CARRIED_RUN_FILES)
        outside = tmp_path.parent / f"{tmp_path.name}_outside"
        outside.write_text("* text=auto\n", encoding="utf-8")
        try:
            os.symlink(outside, tmp_path / ".gitattributes")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available")
        repo_model.mark_generated(workdir, CARRIED_RUN_FILES)
        assert outside.read_text(encoding="utf-8") == "* text=auto\n"
