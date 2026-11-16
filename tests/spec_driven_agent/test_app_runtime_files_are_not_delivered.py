"""What the app writes while the agent test-runs it is not part of the app.

The probe boots the generated backend inside the workspace. An app that
creates a signing key, a database or an upload folder on startup leaves those
files beside its source, and a key delivered that way is shared by every
deployment built from the download. The probe records the files that appear
under the backend while it runs the app; the run deletes them before it
describes and packages the output, unless a generator or a write tool wrote
them.
"""
import json
import os
import subprocess
import sys

import pytest

from besser.spec_driven_agent.agent.runbook import (
    PROBE_FILENAME,
    PROBE_SCRIPT,
    RUNTIME_FILES_FILENAME,
    install_probe,
    remove_app_runtime_files,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker


def _backend(workspace):
    backend = workspace / "backend"
    backend.mkdir(parents=True)
    (backend / "main_api.py").write_text("app = None\n", encoding="utf-8")
    (backend / "sql_alchemy.py").write_text("", encoding="utf-8")
    return backend


def _probe_functions(workspace):
    """The probe script loaded in-process: its functions run, nothing boots."""
    namespace = {"__file__": str(workspace / PROBE_FILENAME), "__name__": "besser_probe"}
    exec(compile(PROBE_SCRIPT, PROBE_FILENAME, "exec"), namespace)
    return namespace


def _record(workspace):
    return json.loads((workspace / RUNTIME_FILES_FILENAME).read_text(encoding="utf-8"))


def _normalised(*paths):
    return {os.path.normcase(os.path.normpath(str(path))) for path in paths}


def test_the_probe_script_and_the_engine_agree_on_the_record_file():
    assert f'"{RUNTIME_FILES_FILENAME}"' in PROBE_SCRIPT


def test_the_probe_records_what_appears_under_the_backend_while_the_app_runs(tmp_path):
    backend = _backend(tmp_path)
    probe = _probe_functions(tmp_path)

    def app_starts():
        (backend / "data").mkdir()
        (backend / "data" / "signing.key").write_text("0f" * 32, encoding="utf-8")
        (backend / "__pycache__").mkdir()
        (backend / "__pycache__" / "main_api.cpython-311.pyc").write_bytes(b"")
        (tmp_path / "notes.txt").write_text("outside the backend\n", encoding="utf-8")
        return "booted"

    assert probe["recording"](app_starts) == "booted"
    assert _record(tmp_path) == ["backend/data/signing.key"]

    def app_takes_an_upload():
        (backend / "uploads").mkdir()
        (backend / "uploads" / "cover.png").write_bytes(b"\x89PNG")

    probe["recording"](app_takes_an_upload)
    assert _record(tmp_path) == ["backend/data/signing.key", "backend/uploads/cover.png"]


def test_a_run_of_the_app_that_writes_nothing_leaves_no_record(tmp_path):
    _backend(tmp_path)
    _probe_functions(tmp_path)["recording"](lambda: None)
    assert not (tmp_path / RUNTIME_FILES_FILENAME).exists()


def test_recorded_files_are_removed_unless_a_tool_or_generator_wrote_them(tmp_path):
    backend = _backend(tmp_path)
    (backend / "data").mkdir()
    for name in ("signing.key", "seed.json"):
        (backend / "data" / name).write_text("{}", encoding="utf-8")
    (tmp_path / RUNTIME_FILES_FILENAME).write_text(
        json.dumps(["backend/data/signing.key", "backend/data/seed.json", "backend/main_api.py"]),
        encoding="utf-8")

    removed = remove_app_runtime_files(
        str(tmp_path), _normalised(backend / "data" / "seed.json", backend / "main_api.py"))

    assert removed == ["backend/data/signing.key"]
    assert not (backend / "data" / "signing.key").exists()
    assert (backend / "data" / "seed.json").exists() and (backend / "main_api.py").exists()
    assert not (tmp_path / RUNTIME_FILES_FILENAME).exists()


def test_a_recorded_path_outside_the_workspace_is_never_deleted(tmp_path):
    workspace = tmp_path / "run"
    _backend(workspace)
    outside = tmp_path / "other_run.txt"
    outside.write_text("someone else's\n", encoding="utf-8")
    (workspace / RUNTIME_FILES_FILENAME).write_text(
        json.dumps(["../other_run.txt", str(outside), ".besser_recipe.json", 7]), encoding="utf-8")
    (workspace / ".besser_recipe.json").write_text("{}", encoding="utf-8")

    assert remove_app_runtime_files(str(workspace), set()) == []
    assert outside.exists() and (workspace / ".besser_recipe.json").exists()


@pytest.fixture
def orchestrator(simple_library_book_model, tmp_path):
    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    return LLMOrchestrator(
        llm_client=Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
    )


def test_the_delivered_output_holds_no_file_the_test_run_of_the_app_wrote(orchestrator, tmp_path):
    """The recipe is written last; the download and a push copy the same folder."""
    backend = _backend(tmp_path)
    orchestrator.executor._generator_files.update({"backend/main_api.py", "backend/sql_alchemy.py"})
    (backend / "auth.py").write_text("SECRET_FILE = 'data/signing.key'\n", encoding="utf-8")
    orchestrator.tool_calls_log = [
        {"tool": "write_file", "input": {"path": "backend/auth.py"}, "success": True},
    ]
    (backend / "data").mkdir()
    (backend / "data" / "signing.key").write_text("0f" * 32, encoding="utf-8")
    (tmp_path / RUNTIME_FILES_FILENAME).write_text(
        json.dumps(["backend/data/signing.key", "backend/auth.py"]), encoding="utf-8")

    orchestrator._save_recipe("add sign-in", 1.0)

    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert sorted(entry["path"] for entry in recipe["output_files"]) == [
        "backend/auth.py", "backend/main_api.py", "backend/sql_alchemy.py"]
    assert recipe["runtime_files_removed"] == ["backend/data/signing.key"]
    delivered = sorted(
        os.path.relpath(os.path.join(root, name), tmp_path).replace("\\", "/")
        for root, _, names in os.walk(tmp_path) for name in names if not name.startswith(".besser_"))
    assert delivered == ["backend/auth.py", "backend/main_api.py", "backend/sql_alchemy.py"]


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

    assert _record(tmp_path) == ["backend/data/signing.key"]
    assert remove_app_runtime_files(str(tmp_path), set()) == ["backend/data/signing.key"]
