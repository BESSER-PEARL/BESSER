"""HTTP-level tests for POST /besser_api/spec-driven/import-generated-code.

A deterministic generator (``/generate-output``, e.g. the SQL generator)
hands its output to the browser and keeps nothing, so the editor had no run
to offer to save to GitHub (BESSER-PEARL/BESSER#748). The endpoint stores
the output the editor received as a run that ``/spec-driven/push-to-github``
pushes like any other.
"""

import asyncio
import io
import os
import zipfile

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import (
    spec_driven_router as router_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven import github_sync
from besser.utilities.web_modeling_editor.backend.settings import get_settings
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime
from tests.utilities.web_modeling_editor.backend.spec_driven.test_push_model_sync import (
    _CapturingGitHubService,
    _export,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _clear_registry,
)

BASE_URL = "http://testserver"
IMPORT_URL = "/besser_api/spec-driven/import-generated-code"
PUSH_URL = "/besser_api/spec-driven/push-to-github"
SESSION = {"X-GitHub-Session": "sid"}
SQL = b"CREATE TABLE book (id INTEGER PRIMARY KEY, title VARCHAR(100));\n"


@pytest.fixture(autouse=True)
def reset_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_run_workspace_root", str(tmp_path))
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


@pytest.fixture
def signed_in(monkeypatch):
    monkeypatch.setattr(router_module, "get_user_token", lambda _sessions, sid: "fake-token")


def _upload(name: str, content: bytes, headers: dict | None = None) -> httpx.Response:
    async def _run() -> httpx.Response:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as ac:
            return await ac.post(IMPORT_URL, files={"file": (name, content)}, headers=headers or {})

    return asyncio.run(_run())


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _entry(run_id: str):
    return asyncio.run(current_runtime().registry.get(run_id))


def _files(root: str) -> set[str]:
    return {
        os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/")
        for d, _dirs, names in os.walk(root) for n in names
    }


def test_needs_a_github_session():
    assert _upload("library.sql", SQL).status_code == 401


def test_an_expired_github_session_is_refused(monkeypatch):
    monkeypatch.setattr(router_module, "get_user_token", lambda _sessions, sid: None)
    assert _upload("library.sql", SQL, SESSION).status_code == 401


def test_a_single_file_becomes_the_runs_only_file(signed_in):
    response = _upload("library.sql", SQL, SESSION)

    assert response.status_code == 200, response.text
    entry = _entry(response.json()["run_id"])
    assert entry is not None
    assert entry.file_name == "library.sql" and not entry.is_zip
    with open(os.path.join(entry.temp_dir, "library.sql"), "rb") as fh:
        assert fh.read() == SQL
    assert {f for f in _files(entry.temp_dir) if not f.startswith(".besser_")} == {"library.sql"}


def test_a_directory_in_the_file_name_is_dropped(signed_in):
    response = _upload("../../etc/library.sql", SQL, SESSION)

    assert response.status_code == 200, response.text
    entry = _entry(response.json()["run_id"])
    assert os.path.isfile(os.path.join(entry.temp_dir, "library.sql"))


def test_a_zip_is_unpacked_without_besser_records(signed_in):
    archive = _zip({
        "app/models.py": b"class Book: ...\n",
        "README.md": b"# app\n",
        ".besser_github_source.json": b'{"commitSha": "' + b"a" * 40 + b'"}',
        "app/.besser_recipe.json": b"{}",
    })
    response = _upload("app.zip", archive, SESSION)

    assert response.status_code == 200, response.text
    entry = _entry(response.json()["run_id"])
    assert entry.is_zip and entry.file_name == "app.zip"
    with open(entry.file_path, "rb") as fh:
        assert fh.read() == archive  # the download hands back what was uploaded
    files = _files(entry.temp_dir)
    assert {"app/models.py", "README.md"} <= files
    assert ".besser_github_source.json" not in files
    assert "app/.besser_recipe.json" not in files


@pytest.mark.parametrize("unsafe", ["../outside.txt", "/abs/outside.txt", "a/../../outside.txt"])
def test_a_zip_reaching_outside_the_run_is_refused(signed_in, tmp_path, unsafe):
    response = _upload("app.zip", _zip({"ok.txt": b"x", unsafe: b"x"}), SESSION)

    assert response.status_code == 400
    assert current_runtime().registry._entries == {}
    assert not (tmp_path / "outside.txt").exists()
    assert os.listdir(tmp_path) == []  # the run's directory went with it


def test_a_zip_past_the_member_cap_is_refused(signed_in, monkeypatch):
    monkeypatch.setattr(github_sync, "MAX_ARCHIVE_MEMBERS", 2)
    response = _upload("app.zip", _zip({"a": b"1", "b": b"2", "c": b"3"}), SESSION)

    assert response.status_code == 400
    assert current_runtime().registry._entries == {}


def test_an_upload_past_the_size_cap_is_refused(signed_in, monkeypatch):
    monkeypatch.setattr(router_module, "MAX_GENERATED_CODE_BYTES", 8)
    assert _upload("library.sql", SQL, SESSION).status_code == 413


def test_a_zip_holding_a_file_and_a_folder_of_one_name_is_refused(signed_in):
    response = _upload("app.zip", _zip({"a": b"1", "a/b": b"2"}), SESSION)

    assert response.status_code == 400
    assert current_runtime().registry._entries == {}


def test_a_zip_file_that_is_not_named_zip_is_one_file(signed_in):
    """A ``.jar`` (or ``.xlsx``) is a ZIP container, but a generator's single output file."""
    archive = _zip({"META-INF/MANIFEST.MF": b"Manifest-Version: 1.0\n"})
    response = _upload("app.jar", archive, SESSION)

    assert response.status_code == 200
    entry = _entry(response.json()["run_id"])
    assert not entry.is_zip
    assert {f for f in _files(entry.temp_dir) if not f.startswith(".besser_")} == {"app.jar"}


def test_storing_takes_a_run_slot_and_gives_it_back(signed_in, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(current_runtime().slots, "try_acquire", lambda: False)
        assert _upload("library.sql", SQL, SESSION).status_code == 429
    assert current_runtime().registry._entries == {}

    released = []
    real_release = current_runtime().slots.release
    monkeypatch.setattr(current_runtime().slots, "release", lambda: (released.append(1), real_release()))
    assert _upload("library.sql", SQL, SESSION).status_code == 200
    assert _upload("bad.zip", b"PK not a zip", SESSION).status_code == 400
    assert released == [1, 1]


def test_a_besser_file_name_is_refused(signed_in):
    assert _upload(".besser_download.json", b"{}", SESSION).status_code == 400


def test_the_generated_sql_is_pushed_to_github(signed_in, monkeypatch):
    """The issue: a deterministic SQL run can be saved to GitHub like a Spec-Driven run."""
    fake = _CapturingGitHubService()
    pushed = {}

    async def push_directory_to_repo(owner, repo_name, directory_path, **kwargs):
        for path in _files(directory_path):
            with open(os.path.join(directory_path, path), "rb") as fh:
                pushed[path] = fh.read()
        return {"commit_sha": "d" * 40, "total_files": len(pushed), "uploaded_files": list(pushed),
                "status": "pushed", "own_commit_sha": "d" * 40}

    fake.push_directory_to_repo = push_directory_to_repo
    monkeypatch.setattr(github_sync, "create_github_service", lambda token: fake)
    run_id = _upload("library.sql", SQL, SESSION).json()["run_id"]

    async def _push() -> httpx.Response:
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as ac:
            return await ac.post(PUSH_URL, headers=SESSION, json={
                "run_id": run_id,
                "projectExport": _export("Library"),
                "deploy_config": {"repo_name": "library", "use_existing": False},
            })

    response = asyncio.run(_push())

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert pushed["library.sql"] == SQL
    assert "buml/diagrams.json" in pushed
    assert not [path for path in pushed if path.split("/")[-1].startswith(".besser_")]


def test_a_cancelled_upload_waits_for_its_writer_then_leaves_nothing(monkeypatch):
    """The writer thread cannot be stopped; removing the run while it writes left an orphan tree."""
    import threading

    started, release, written = threading.Event(), threading.Event(), []

    def slow_store(content, name, is_zip, run_dir):
        started.set()
        release.wait(5)
        os.makedirs(os.path.join(run_dir, "late"), exist_ok=True)
        written.append(run_dir)
        return os.path.join(run_dir, name)

    monkeypatch.setattr(github_sync, "_store_generated_code", slow_store)

    async def scenario():
        task = asyncio.create_task(
            github_sync.import_generated_code(SQL, "library.sql", current_runtime().registry))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()  # still waiting for the writer
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert written and not os.path.exists(written[0])
    assert current_runtime().registry._entries == {}
