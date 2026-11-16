"""What the app writes while the agent test-runs it is not part of the app.

The probe boots the generated backend inside the workspace. An app that
creates a signing key or an upload folder on startup leaves those files beside
its source, and a key delivered that way is shared by every deployment built
from the download. The probe records the files that appear under the backend
while it runs the app, each with its size and modification time; the run
deletes them before it describes and packages the output.

The record is written inside the sandbox, so the run trusts none of it: it
deletes only an unchanged file under the backend folder it finds itself, that
no generator or write tool wrote and that was not there when the run started.
"""
import json
import logging
import os
import subprocess
import sys

import pytest

from besser.spec_driven_agent.agent import runbook
from besser.spec_driven_agent.agent.runbook import (
    PROBE_FILENAME,
    PROBE_SCRIPT,
    RUNTIME_FILES_FILENAME,
    install_probe,
    remove_app_runtime_files,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state.checkpoint import load_checkpoint, save_checkpoint

KEY = "backend/data/signing.key"


def _backend(workspace, folder="backend"):
    backend = workspace / folder
    backend.mkdir(parents=True, exist_ok=True)
    (backend / "main_api.py").write_text("app = None\n", encoding="utf-8")
    (backend / "sql_alchemy.py").write_text("", encoding="utf-8")
    return backend


def _write(workspace, rel, text="0f1e2d3c" * 8):
    path = workspace / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _stat(workspace, rel):
    info = os.stat(workspace / rel)
    return [info.st_size, info.st_mtime_ns]


def _note(workspace, *rels, **explicit):
    """Write the probe's record: ``rels`` with their real size and time, ``explicit`` as given."""
    files = {rel: _stat(workspace, rel) for rel in rels}
    files.update(explicit)
    (workspace / RUNTIME_FILES_FILENAME).write_text(json.dumps({"files": files, "overflow": 0}), encoding="utf-8")


def _probe_functions(workspace):
    """The probe script loaded in-process: its functions run, nothing boots."""
    namespace = {"__file__": str(workspace / PROBE_FILENAME), "__name__": "besser_probe"}
    exec(compile(PROBE_SCRIPT, PROBE_FILENAME, "exec"), namespace)
    return namespace


def _record(workspace):
    return json.loads((workspace / RUNTIME_FILES_FILENAME).read_text(encoding="utf-8"))


def _tree(workspace):
    return sorted(
        os.path.relpath(os.path.join(root, name), workspace).replace("\\", "/")
        for root, _, names in os.walk(workspace) for name in names if not name.startswith(".besser_"))


# --- the probe's record ---------------------------------------------------

def test_the_probe_script_and_the_engine_agree_on_the_record_file():
    assert f'"{RUNTIME_FILES_FILENAME}"' in PROBE_SCRIPT
    assert f"MAX_RECORDED = {runbook._MAX_RUNTIME_RECORD_ENTRIES} " in PROBE_SCRIPT


def test_the_probe_records_what_appears_under_the_backend_while_the_app_runs(tmp_path):
    backend = _backend(tmp_path)
    probe = _probe_functions(tmp_path)

    def app_starts():
        _write(tmp_path, KEY)
        _write(tmp_path, "backend/__pycache__/main_api.cpython-311.pyc", "")
        _write(tmp_path, "backend/.pytest_cache/CACHEDIR.TAG")
        _write(tmp_path, "backend/htmlcov/index.html")
        _write(tmp_path, "notes.txt", "outside the backend\n")
        return "booted"

    assert probe["recording"](app_starts) == "booted"
    assert _record(tmp_path) == {"files": {KEY: _stat(tmp_path, KEY)}, "overflow": 0}

    probe["recording"](lambda: (backend / "uploads").mkdir() or _write(tmp_path, "backend/uploads/cover.png"))
    assert sorted(_record(tmp_path)["files"]) == [KEY, "backend/uploads/cover.png"]


def test_a_run_of_the_app_that_writes_nothing_leaves_no_record(tmp_path):
    _backend(tmp_path)
    _probe_functions(tmp_path)["recording"](lambda: None)
    assert not (tmp_path / RUNTIME_FILES_FILENAME).exists()


def test_a_file_edited_between_two_probe_calls_is_no_longer_the_apps(tmp_path):
    """The app created an empty catalogue at boot; the agent then filled it
    through a shell command. It is the agent's data now and ships."""
    _backend(tmp_path)
    probe = _probe_functions(tmp_path)
    probe["recording"](lambda: _write(tmp_path, "backend/data/catalog.json", "[]"))
    assert "backend/data/catalog.json" in _record(tmp_path)["files"]

    _write(tmp_path, "backend/data/catalog.json", json.dumps([{"id": n} for n in range(50)]))
    probe["recording"](lambda: None)

    assert _record(tmp_path)["files"] == {}


def test_the_probe_records_nothing_when_the_backend_is_the_workspace_root(tmp_path):
    """There "under the backend" is everything: a lock file or seed data that
    appears during a probe call is not the app's."""
    _backend(tmp_path, folder=".")
    _probe_functions(tmp_path)["recording"](
        lambda: _write(tmp_path, "frontend/package-lock.json", "{}") and _write(tmp_path, "seed_data.json", "[1]"))
    assert not (tmp_path / RUNTIME_FILES_FILENAME).exists()


# --- what the run deletes -------------------------------------------------

def test_recorded_files_are_removed_unless_a_tool_or_generator_wrote_them(tmp_path):
    backend = _backend(tmp_path)
    _write(tmp_path, KEY)
    _write(tmp_path, "backend/data/.jwt_secret")
    _write(tmp_path, "backend/data/seed.json", "{}")
    _note(tmp_path, KEY, "backend/data/.jwt_secret", "backend/data/seed.json", "backend/main_api.py")

    result = remove_app_runtime_files(str(tmp_path), {"backend/data/seed.json", "backend/main_api.py"})

    assert result.removed == ("backend/data/.jwt_secret", KEY)
    assert (result.not_removed, result.note) == ((), "")
    assert _tree(tmp_path) == ["backend/data/seed.json", "backend/main_api.py", "backend/sql_alchemy.py"]
    assert not (tmp_path / RUNTIME_FILES_FILENAME).exists()
    assert backend.is_dir()


@pytest.mark.parametrize("victim, entry", [
    ("README.md", "README.md"),
    ("frontend/src/App.tsx", "frontend/src/App.tsx"),
    (".git/config", ".git/config"),
    (".agent-trace/old-run.json", ".agent-trace/old-run.json"),
    (".besser_snapshot/backend/main_api.py", ".besser_snapshot/backend/main_api.py"),
    ("backend/.besser_state.json", "backend/.besser_state.json"),
    ("backend/.cache/token", "backend/.cache/token"),
    ("backend/data/x.key", "backend/data/../data/x.key"),
    ("backend/data/x.key", "backend\\data\\x.key"),
    ("backend/data/x.key", "BACKEND/data/x.key"),
])
def test_a_record_cannot_name_a_file_outside_what_the_probe_records(tmp_path, victim, entry):
    """The probe records plain paths under the backend, through no dot-folder."""
    _backend(tmp_path)
    _write(tmp_path, victim)
    _note(tmp_path, **{entry: _stat(tmp_path, victim)})

    assert remove_app_runtime_files(str(tmp_path), set()).removed == ()
    assert (tmp_path / victim).exists()


@pytest.mark.parametrize("alias", ["backend/main_api.py.", "backend/main_api.py ", "backend/main_api.py::$DATA",
                                   "backend/./main_api.py", "backend/data./../main_api.py"])
def test_another_spelling_of_a_written_file_does_not_get_it_deleted(tmp_path, alias):
    """Windows opens ``main_api.py`` for a name with a trailing dot or space."""
    _backend(tmp_path)
    _note(tmp_path, **{alias: _stat(tmp_path, "backend/main_api.py")})

    assert remove_app_runtime_files(str(tmp_path), {"backend/main_api.py"}).removed == ()
    assert (tmp_path / "backend" / "main_api.py").exists()


def test_a_recorded_path_outside_the_workspace_or_through_a_link_is_never_deleted(tmp_path):
    workspace = tmp_path / "run"
    backend = _backend(workspace)
    outside = _write(tmp_path, "other_run.txt")
    entries = {"../other_run.txt": _stat(tmp_path, "other_run.txt"), str(outside): _stat(tmp_path, "other_run.txt")}
    try:
        os.symlink(outside, backend / "link.key")
        entries["backend/link.key"] = _stat(tmp_path, "other_run.txt")
    except OSError:
        pass  # no symlink privilege on this Windows account
    _note(workspace, **entries)

    assert remove_app_runtime_files(str(workspace), set()).removed == ()
    assert outside.exists() and os.path.lexists(backend / "link.key") == ("backend/link.key" in entries)


def test_a_recorded_file_that_changed_since_the_app_wrote_it_is_kept(tmp_path):
    _backend(tmp_path)
    _write(tmp_path, "backend/data/catalog.json", "[]")
    _note(tmp_path, "backend/data/catalog.json")
    _write(tmp_path, "backend/data/catalog.json", '[{"id": 1}]')

    assert remove_app_runtime_files(str(tmp_path), set()).removed == ()
    assert (tmp_path / "backend/data/catalog.json").read_text(encoding="utf-8") == '[{"id": 1}]'


def test_a_file_the_output_leaves_out_by_name_is_not_deleted(tmp_path):
    """The database keeps the records between two boots of a resumed run."""
    _backend(tmp_path)
    _write(tmp_path, "backend/data/app.db")
    _note(tmp_path, "backend/data/app.db")

    assert remove_app_runtime_files(str(tmp_path), set()).removed == ()
    assert (tmp_path / "backend/data/app.db").exists()


def test_nothing_is_removed_when_the_backend_is_the_workspace_root(tmp_path):
    _backend(tmp_path, folder=".")
    _write(tmp_path, "migrations/0001_initial.py")
    _note(tmp_path, "migrations/0001_initial.py")

    result = remove_app_runtime_files(str(tmp_path), set())

    assert result.removed == () and "workspace root" in result.note
    assert (tmp_path / "migrations/0001_initial.py").exists()


_UNUSABLE_RECORDS = {
    "not json": ("not json", "could not be read"),
    "a bare list": ('["backend/data/signing.key"]', "could not be read"),
    "over the size limit": ('{"files": {}, "padding": "' + "x" * 1_000_001 + '"}', "too large"),
    "more files than it holds": (json.dumps({"files": {}, "overflow": 7}), "7 more"),
}


@pytest.mark.parametrize("case", _UNUSABLE_RECORDS)
def test_a_record_that_cannot_be_used_in_full_is_reported(tmp_path, caplog, case):
    body, reason = _UNUSABLE_RECORDS[case]
    _backend(tmp_path)
    _write(tmp_path, KEY)
    (tmp_path / RUNTIME_FILES_FILENAME).write_text(body, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=runbook.__name__):
        result = remove_app_runtime_files(str(tmp_path), set())

    assert reason in result.note and result.removed == ()
    assert reason in caplog.text
    assert not (tmp_path / RUNTIME_FILES_FILENAME).exists()


def test_a_file_that_cannot_be_deleted_is_reported(tmp_path, caplog, monkeypatch):
    """Windows refuses to delete a file a process still holds open."""
    _backend(tmp_path)
    _write(tmp_path, KEY)
    _write(tmp_path, "backend/data/session.key")
    _note(tmp_path, KEY, "backend/data/session.key")
    real_remove = runbook.remove_plain

    def remove(path, **kwargs):
        if str(path).endswith("session.key"):
            raise PermissionError(13, "The process cannot access the file")
        real_remove(path, **kwargs)

    monkeypatch.setattr(runbook, "remove_plain", remove)
    with caplog.at_level(logging.WARNING, logger=runbook.__name__):
        result = remove_app_runtime_files(str(tmp_path), set())

    assert result.removed == (KEY,) and result.not_removed == ("backend/data/session.key",)
    assert "could not remove backend/data/session.key" in caplog.text


# --- the run --------------------------------------------------------------

def _orchestrator(model, workspace):
    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    return LLMOrchestrator(llm_client=Client(), domain_model=model, output_dir=str(workspace),
                           enable_checkpointing=False)


def _recipe(orchestrator, workspace):
    orchestrator._save_recipe("add sign-in", 1.0)
    return json.loads((workspace / ".besser_recipe.json").read_text(encoding="utf-8"))


def test_the_delivered_output_holds_no_file_the_test_run_of_the_app_wrote(simple_library_book_model, tmp_path):
    """The recipe is written last; the download and a push copy the same folder."""
    _backend(tmp_path)
    orchestrator = _orchestrator(simple_library_book_model, tmp_path)
    (tmp_path / "backend" / "main_api.py").write_text("app = 1\n", encoding="utf-8")
    orchestrator.executor._generator_files.update({"backend/main_api.py", "backend/sql_alchemy.py"})
    _write(tmp_path, "backend/auth.py", "SECRET_FILE = 'data/signing.key'\n")
    orchestrator.tool_calls_log = [{"tool": "write_file", "input": {"path": "backend/auth.py"}, "success": True}]
    _write(tmp_path, KEY)
    _note(tmp_path, KEY, "backend/auth.py")

    recipe = _recipe(orchestrator, tmp_path)

    delivered = ["backend/auth.py", "backend/main_api.py", "backend/sql_alchemy.py"]
    assert sorted(entry["path"] for entry in recipe["output_files"]) == delivered
    assert (recipe["runtime_files_removed"], recipe["runtime_files_removed_count"]) == ([KEY], 1)
    assert _tree(tmp_path) == delivered
    # A second save (a fix pass after the first) keeps the account of the first.
    assert _recipe(orchestrator, tmp_path)["runtime_files_removed"] == [KEY]


@pytest.mark.parametrize("calls", [
    [("delete_file", True)],
    [("write_file", True), ("delete_file", True)],
    [("write_file", False)],
])
def test_a_file_the_agent_deleted_and_the_app_wrote_again_is_the_apps(simple_library_book_model, tmp_path, calls):
    _backend(tmp_path)
    orchestrator = _orchestrator(simple_library_book_model, tmp_path)
    orchestrator.tool_calls_log = [
        {"tool": tool, "input": {"path": KEY}, "success": success} for tool, success in calls]
    _write(tmp_path, KEY)
    _note(tmp_path, KEY)

    assert _recipe(orchestrator, tmp_path)["runtime_files_removed"] == [KEY]


def test_a_file_that_was_there_when_the_run_started_is_never_removed(simple_library_book_model, tmp_path):
    """A modify run's seed or an imported repository: the user's file, whatever a record says."""
    _backend(tmp_path)
    _write(tmp_path, "backend/keys/partner_public.pem", "seeded by the user\n")
    orchestrator = _orchestrator(simple_library_book_model, tmp_path)
    _write(tmp_path, KEY)
    _note(tmp_path, KEY, "backend/keys/partner_public.pem")

    assert _recipe(orchestrator, tmp_path)["runtime_files_removed"] == [KEY]
    assert (tmp_path / "backend/keys/partner_public.pem").exists()


def test_a_resumed_run_keeps_the_list_of_files_it_started_with(tmp_path):
    from tests.spec_driven_agent.test_checkpoint import _make_checkpoint

    save_checkpoint(str(tmp_path), _make_checkpoint(files_at_start=["backend/keys/partner_public.pem"]))
    assert load_checkpoint(str(tmp_path)).files_at_start == ["backend/keys/partner_public.pem"]


def test_the_recipe_counts_every_removed_file_and_names_the_first(simple_library_book_model, tmp_path):
    _backend(tmp_path)
    orchestrator = _orchestrator(simple_library_book_model, tmp_path)
    uploads = ["backend/uploads/%03d.bin" % n for n in range(120)]
    for rel in uploads:
        _write(tmp_path, rel)
    _note(tmp_path, *uploads)

    recipe = _recipe(orchestrator, tmp_path)

    assert recipe["runtime_files_removed_count"] == 120
    assert recipe["runtime_files_removed"] == uploads[:50]
    assert (recipe["runtime_files_not_removed_count"], recipe["runtime_files_note"]) == (0, "")


def test_a_real_boot_records_the_key_the_app_creates_on_import(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    backend = _backend(tmp_path)
    (backend / "main_api.py").write_text(
        "import os, secrets\n"
        "from fastapi import FastAPI\n"
        "os.makedirs('data', exist_ok=True)\n"
        "if not os.path.exists('data/signing.key'):\n"
        "    open('data/signing.key', 'w').write(secrets.token_hex(32))\n"
        "app = FastAPI()\n", encoding="utf-8")
    assert install_probe(str(tmp_path))

    def probe(*argv):
        return subprocess.run([sys.executable, PROBE_FILENAME, *argv], cwd=tmp_path,
                              capture_output=True, text=True, timeout=90)

    try:
        booted = probe("up")
        assert "BOOT_OK" in booted.stdout, booted.stdout + booted.stderr
    finally:
        probe("down")

    assert list(_record(tmp_path)["files"]) == [KEY]
    assert remove_app_runtime_files(str(tmp_path), set()).removed == (KEY,)
