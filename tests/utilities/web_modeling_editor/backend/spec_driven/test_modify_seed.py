"""Tests for the runner-side incremental modify seed arm.

Covers:
  * request-model fields (``base_run_id`` pattern, ``mode`` literal);
  * the seed arm copies the base tree, strips checkpoint + snapshot, KEEPS
    the recipe, excludes build dirs, and leaves the base entry intact +
    downloadable, then drives ``orchestrator.modify``;
  * a base whose registry entry has expired warns (non-terminal
    ``INCOMPLETE``) and falls back to a from-scratch ``run()`` — not a
    crash.

The orchestrator is stubbed (same pattern as ``test_runner.py``) so the
runner's queue bridge and workspace allocation are exercised without a
real LLM or generator.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest
from pydantic import ValidationError as PydanticValidationError

from besser.spec_driven_agent.state.checkpoint import CHECKPOINT_FILENAME
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.models.spec_driven import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
    SmartRunEntry,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime


# ---------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------


def _build_request(**overrides) -> SmartGenerateRequest:
    project = ProjectInput(
        id="test-project",
        type="BesserProject",
        name="TestProject",
        createdAt="2026-04-15T00:00:00Z",
        diagrams={
            "ClassDiagram": [
                {"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}
            ]
        },
        currentDiagramIndices={"ClassDiagram": 0},
    )
    defaults = dict(
        project=project,
        instructions="Add a dark theme",
        api_key="sk-ant-test-NEVER-LEAK-1234567890",
        provider="anthropic",
        llm_model="claude-sonnet-4-5",
        max_cost_usd=1.0,
        max_runtime_seconds=60,
    )
    defaults.update(overrides)
    return SmartGenerateRequest(**defaults)


# ---------------------------------------------------------------------
# Orchestrator stub — records which entry point the runner invoked
# ---------------------------------------------------------------------


_CALLS: list[tuple[str, str]] = []  # (method, output_dir)


class _StubOrchestrator:
    def __init__(self, *, llm_client, output_dir, **kwargs):
        self.client = llm_client
        self.output_dir = output_dir
        self.total_turns = 1
        self._phase2_exited_cleanly = True
        self._phase2_stop_reason = "completed"
        self.max_cost_usd = kwargs.get("max_cost_usd", 1.0)
        self.max_runtime_seconds = kwargs.get("max_runtime_seconds", 60)

    def _finish(self, method: str) -> str:
        _CALLS.append((method, self.output_dir))
        self.client.usage.estimated_cost = 0.01
        os.makedirs(self.output_dir, exist_ok=True)
        # Write a NEW file so _package_result has user output to serve.
        with open(os.path.join(self.output_dir, "NEW.md"), "w", encoding="utf-8") as fh:
            fh.write("# added by stub\n")
        return self.output_dir

    def run(self, instructions: str) -> str:
        return self._finish("run")

    def modify(self, instructions: str) -> str:
        return self._finish("modify")

    def resume(self, instructions: str) -> str:  # pragma: no cover - unused
        return self._finish("resume")


class _FakeUsage:
    def __init__(self):
        self.estimated_cost = 0.0


class _FakeClient:
    def __init__(self):
        self.usage = _FakeUsage()


@pytest.fixture(autouse=True)
def _clear_registry_and_calls():
    _CALLS.clear()
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


async def _clear_registry():
    import shutil

    async with current_runtime().registry._lock:
        for entry in list(current_runtime().registry._entries.values()):
            shutil.rmtree(entry.temp_dir, ignore_errors=True)
        current_runtime().registry._entries.clear()


@pytest.fixture
def stub_orchestrator(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _StubOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())


async def _collect_frames(runner: SmartGenerationRunner) -> list[bytes]:
    return [frame async for frame in runner.generate_and_stream()]


def _parse(frame: bytes) -> dict:
    text = frame.decode("utf-8")
    data_line = [l for l in text.splitlines() if l.startswith("data: ")][0]
    return json.loads(data_line[len("data: "):])


# ---------------------------------------------------------------------
# Request model fields
# ---------------------------------------------------------------------


class TestRequestFields:
    def test_defaults_are_generate_and_none(self):
        req = _build_request()
        assert req.mode == "generate"
        assert req.base_run_id is None

    def test_valid_modify_pairing(self):
        req = _build_request(mode="modify", base_run_id="a" * 32)
        assert req.mode == "modify"
        assert req.base_run_id == "a" * 32

    def test_base_run_id_rejects_non_hex32(self):
        with pytest.raises(PydanticValidationError):
            _build_request(base_run_id="not-a-valid-id")
        with pytest.raises(PydanticValidationError):
            # Uppercase hex is outside [a-f0-9].
            _build_request(base_run_id="A" * 32)
        with pytest.raises(PydanticValidationError):
            _build_request(base_run_id="a" * 31)  # too short

    def test_mode_rejects_unknown_value(self):
        with pytest.raises(PydanticValidationError):
            _build_request(mode="delete")


# ---------------------------------------------------------------------
# (d) Seed arm
# ---------------------------------------------------------------------


def _make_base_dir(tmp_path) -> str:
    base = tmp_path / "base"
    (base / "app").mkdir(parents=True)
    (base / "app" / "main.py").write_text("print('base app')\n", encoding="utf-8")
    (base / CHECKPOINT_FILENAME).write_text('{"turn": 3}', encoding="utf-8")
    (base / ".besser_snapshot").mkdir()
    (base / ".besser_snapshot" / "old.txt").write_text("old", encoding="utf-8")
    (base / "node_modules").mkdir()
    (base / "node_modules" / "junk.js").write_text("junk", encoding="utf-8")
    recipe = {
        "generator_used": "generate_fastapi_backend",
        "output_files": [
            {"path": "app/main.py", "size": 17, "source": "generator"},
        ],
    }
    (base / ".besser_recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    return str(base)


class TestSeedArm:
    def test_modify_seeds_tree_strips_internals_keeps_recipe(
        self, tmp_path, stub_orchestrator
    ):
        base_dir = _make_base_dir(tmp_path)
        base_id = "a" * 32
        asyncio.run(current_runtime().registry.put(
            base_id,
            SmartRunEntry(
                file_path=os.path.join(base_dir, "app", "main.py"),
                file_name="main.py",
                is_zip=False,
                temp_dir=base_dir,
                created_at=time.time(),
            ),
        ))

        request = _build_request(mode="modify", base_run_id=base_id)
        runner = SmartGenerationRunner(
            request, runtime=current_runtime(), base_run_id=base_id, mode="modify",
        )
        frames = asyncio.run(_collect_frames(runner))
        parsed = [_parse(f) for f in frames]

        # The modify entry point ran (not run/resume).
        assert ("modify", runner.temp_dir) in _CALLS
        assert not any(m == "run" for m, _ in _CALLS)

        # Run succeeded → done event, no terminal error.
        assert parsed[-1]["event"] == "done"

        # The seeded workspace is a NEW dir, not the base.
        assert runner.temp_dir is not None
        assert os.path.realpath(runner.temp_dir) != os.path.realpath(base_dir)

        # Copied: the base's user file + the recipe.
        assert os.path.isfile(os.path.join(runner.temp_dir, "app", "main.py"))
        assert os.path.isfile(os.path.join(runner.temp_dir, ".besser_recipe.json"))
        # Stripped: checkpoint + snapshot.
        assert not os.path.exists(os.path.join(runner.temp_dir, CHECKPOINT_FILENAME))
        assert not os.path.exists(os.path.join(runner.temp_dir, ".besser_snapshot"))
        # Excluded: build/dependency dirs.
        assert not os.path.exists(os.path.join(runner.temp_dir, "node_modules"))

        # Line attribution (#620): the base has no attribution state, so its
        # lines are of unknown origin rather than credited to this run's AI.
        from besser.spec_driven_agent.state.agent_trace import AttributionState
        state = AttributionState.load(runner.temp_dir)
        assert state is not None
        assert {state.contributors[i]["kind"] for i in state.files["app/main.py"][1]} == {"unknown"}

        # The base is untouched and still fully downloadable.
        assert os.path.isfile(os.path.join(base_dir, "app", "main.py"))
        assert os.path.isfile(os.path.join(base_dir, CHECKPOINT_FILENAME))
        assert os.path.isdir(os.path.join(base_dir, ".besser_snapshot"))
        assert os.path.isdir(os.path.join(base_dir, "node_modules"))
        base_entry = asyncio.run(current_runtime().registry.get(base_id))
        assert base_entry is not None and os.path.isdir(base_entry.temp_dir)


# ---------------------------------------------------------------------
# (d2) Seed from the download once the base workspace has expired
# ---------------------------------------------------------------------


# What a packaged run's download holds: user files, the provenance note, the
# Agent Trace records and the run files a later run reads back.
_ARCHIVE_FILES = {
    "app/main.py": "print('base app')\n",
    "frontend/src/App.tsx": "export default function App() { return null; }\n",
    "BESSER_GENERATION.md": "# Generated by BESSER\n",
    ".agent-trace/run.json": "{}",
    ".besser_recipe.json": json.dumps({"generator_used": "generate_fastapi_backend",
                                       "model": {"user_model": {"classes": {}}}}),
    ".besser_attribution.json": "{}",
    ".besser_generator_output.json": "{}",
}


def _register_archived_base(tmp_path, run_id: str, extra: dict | None = None, *, packaged: bool = True) -> str:
    """Register a base run whose workspace is gone and whose ZIP download remains:
    one BESSER packaged (``packaged``), else an uploaded generator ZIP."""
    import zipfile

    workspace = str(tmp_path / f"besser_llm_{run_id}_x")  # never created: swept
    download = runner_module.run_download_dir(workspace)
    os.makedirs(download)
    artifact = os.path.join(download, "artifact")
    with zipfile.ZipFile(artifact, "w") as zf:
        for name, text in {**_ARCHIVE_FILES, **(extra or {})}.items():
            zf.writestr(name, text)
    asyncio.run(current_runtime().registry.put(run_id, SmartRunEntry(
        file_path=artifact, file_name=f"besser_smart_{run_id}.zip", is_zip=True,
        temp_dir=workspace, created_at=time.time(), packaged=packaged,
    )))
    return workspace


def _run_modify(base_id: str) -> tuple[SmartGenerationRunner, list[dict], dict[str, str]]:
    """Run a stubbed modify of ``base_id``: the runner, its events and the tree modify() got."""
    runner = SmartGenerationRunner(
        _build_request(mode="modify", base_run_id=base_id),
        runtime=current_runtime(), base_run_id=base_id, mode="modify",
    )
    seeded = {}
    real_modify = _StubOrchestrator.modify

    def modify(self, instructions):
        seeded.update(_tree(self.output_dir))
        return real_modify(self, instructions)

    _StubOrchestrator.modify = modify
    try:
        parsed = [_parse(f) for f in asyncio.run(_collect_frames(runner))]
    finally:
        _StubOrchestrator.modify = real_modify
    return runner, parsed, seeded


def _tree(root: str) -> dict[str, str]:
    files = {}
    for folder, _dirs, names in os.walk(root):
        for name in names:
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as fh:
                files[os.path.relpath(path, root).replace(os.sep, "/")] = fh.read()
    return files


class TestArchiveSeed:
    def test_expired_workspace_seeds_from_the_download_and_stays_a_modify(self, tmp_path, stub_orchestrator):
        base_id = "c" * 32
        _register_archived_base(tmp_path, base_id)
        runner, parsed, seeded = _run_modify(base_id)

        assert [m for m, _ in _CALLS] == ["modify"]
        assert not [p for p in parsed if p["event"] == "error"]
        assert parsed[-1]["event"] == "done"
        assert (runner._facts.mode, runner._facts.mode_reason) == ("modify", "edit of a previous result")
        # The seeded tree is the download's tree, prepared as a workspace seed is:
        # the provenance note is written again at packaging, the attribution is updated.
        prepared = ("BESSER_GENERATION.md", ".besser_attribution.json")
        assert {k: v for k, v in seeded.items() if k not in prepared} == {
            k: v for k, v in _ARCHIVE_FILES.items() if k not in prepared}
        assert ".besser_attribution.json" in seeded
        assert json.loads(seeded[".besser_recipe.json"])["generator_used"] == "generate_fastapi_backend"

    def test_a_download_with_an_unsafe_path_seeds_nothing(self, tmp_path):
        import zipfile

        archive = tmp_path / "artifact"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("app/main.py", "x = 1\n")
            zf.writestr("../outside.py", "x = 2\n")
        dest = tmp_path / "dest"
        dest.mkdir()
        with pytest.raises(ValueError):
            runner_module._seed_workspace_from_archive(str(archive), str(dest), True)
        assert os.listdir(dest) == []
        assert not (tmp_path / "outside.py").exists()

    def test_a_download_that_cannot_be_unpacked_is_not_edited(self, tmp_path, stub_orchestrator):
        base_id = "d" * 32
        _register_archived_base(tmp_path, base_id, extra={"../outside.py": "x = 2\n"})
        runner = SmartGenerationRunner(
            _build_request(mode="modify", base_run_id=base_id),
            runtime=current_runtime(), base_run_id=base_id, mode="modify",
        )
        parsed = [_parse(f) for f in asyncio.run(_collect_frames(runner))]
        assert [m for m, _ in _CALLS] == ["run"]
        assert runner._facts.mode_reason == "previous result could not be copied, rebuilt from scratch"
        assert any(p["event"] == "error" and p["code"] == "INCOMPLETE" for p in parsed)
        assert not (tmp_path / "outside.py").exists()

    def test_executable_files_stay_executable_and_nothing_else_is_set(self, tmp_path, monkeypatch):
        import stat
        import zipfile

        archive = tmp_path / "artifact"
        with zipfile.ZipFile(archive, "w") as zf:
            for name, mode in (("mvnw", 0o755), ("scripts/run.sh", 0o744), ("README.md", 0o644),
                               ("tool", 0o6777)):
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFREG | mode) << 16
                zf.writestr(info, "x\n")
        dest = tmp_path / "dest"
        dest.mkdir()
        real_chmod, chmods = os.chmod, {}

        def chmod(path, mode, *args, **kwargs):
            if not isinstance(path, int):
                chmods[os.path.relpath(os.fspath(path), dest).replace(os.sep, "/")] = mode
            else:
                chmods[f"fd{path}"] = mode
            return real_chmod(path, mode, *args, **kwargs)

        monkeypatch.setattr(os, "chmod", chmod)
        runner_module._seed_workspace_from_archive(str(archive), str(dest), True)
        assert len(chmods) == 3  # mvnw, scripts/run.sh, tool; never README.md
        assert "README.md" not in chmods
        for mode in chmods.values():
            assert mode & 0o100 and not mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX | 0o002)
        if os.name == "posix":
            assert os.stat(dest / "mvnw").st_mode & 0o111 == 0o111
            assert os.stat(dest / "scripts" / "run.sh").st_mode & 0o111 == 0o100
            assert os.stat(dest / "README.md").st_mode & 0o111 == 0
            assert not os.stat(dest / "tool").st_mode & (stat.S_ISUID | stat.S_ISGID | 0o002)

    def test_an_uploaded_zip_seeds_without_its_run_records(self, tmp_path, stub_orchestrator):
        base_id = "e" * 32
        _register_archived_base(tmp_path, base_id, packaged=False)
        _runner, _parsed, seeded = _run_modify(base_id)
        assert [m for m, _ in _CALLS] == ["modify"]
        assert ".besser_recipe.json" not in seeded and ".besser_generator_output.json" not in seeded
        assert "app/main.py" in seeded

    def test_the_github_source_of_the_base_is_kept_beside_its_download(self, tmp_path, stub_orchestrator):
        source = {"owner": "acme", "repo": "shop", "commitSha": "a" * 40, "paths": {"app/main.py": "h"}}
        base_dir = _make_base_dir(tmp_path)
        with open(os.path.join(base_dir, runner_module.GITHUB_SOURCE_FILENAME), "w", encoding="utf-8") as fh:
            json.dump(source, fh)
        base_id = "a" * 32
        asyncio.run(current_runtime().registry.put(base_id, SmartRunEntry(
            file_path=os.path.join(base_dir, "app", "main.py"), file_name="main.py", is_zip=False,
            temp_dir=base_dir, created_at=time.time())))
        runner, _parsed, _seeded = _run_modify(base_id)
        # The run started from that base keeps the record beside its own download...
        kept = os.path.join(runner_module.run_download_dir(runner.temp_dir), runner_module.GITHUB_SOURCE_FILENAME)
        with open(kept, encoding="utf-8") as fh:
            assert json.load(fh) == source

        # ...and once a workspace is gone, a modify seeds the record from there.
        _CALLS.clear()
        archived_id = "9" * 32
        workspace = _register_archived_base(tmp_path, archived_id)
        with open(os.path.join(runner_module.run_download_dir(workspace), runner_module.GITHUB_SOURCE_FILENAME),
                  "w", encoding="utf-8") as fh:
            json.dump(source, fh)
        _runner, _parsed, seeded = _run_modify(archived_id)
        assert json.loads(seeded[runner_module.GITHUB_SOURCE_FILENAME]) == source

    def test_a_push_records_its_commit_beside_the_download_too(self, tmp_path):
        from besser.utilities.web_modeling_editor.backend.services.spec_driven import github_sync

        workspace = tmp_path / "besser_llm_run_x"
        workspace.mkdir()
        os.makedirs(runner_module.run_download_dir(str(workspace)))
        github_sync._record_run_seed(str(workspace), "acme", "shop", "b" * 40, {"app/main.py": "h"})
        kept = os.path.join(runner_module.run_download_dir(str(workspace)), runner_module.GITHUB_SOURCE_FILENAME)
        with open(kept, encoding="utf-8") as fh:
            assert json.load(fh)["commitSha"] == "b" * 40

    def test_a_download_removed_before_it_is_opened_reads_as_expired(self, tmp_path, stub_orchestrator,
                                                                      monkeypatch):
        import shutil

        base_id = "8" * 32
        workspace = _register_archived_base(tmp_path, base_id)
        real_seed = runner_module._seed_workspace_from_archive

        def swept_first(archive, dest, packaged):
            shutil.rmtree(runner_module.run_download_dir(workspace))
            return real_seed(archive, dest, packaged)

        monkeypatch.setattr(runner_module, "_seed_workspace_from_archive", swept_first)
        runner, parsed, _seeded = _run_modify(base_id)
        assert [m for m, _ in _CALLS] == ["run"]
        assert runner._facts.mode_reason == "previous result expired, rebuilt from scratch"
        warnings = [p["message"] for p in parsed if p["event"] == "error" and p["code"] == "INCOMPLETE"]
        assert warnings and "expired" in warnings[0]

    def test_the_download_kind_survives_a_restart(self, tmp_path):
        registry = runner_module.SmartRunRegistry()
        for run_id, packaged in (("1" * 32, True), ("2" * 32, False)):
            workspace = str(tmp_path / f"besser_llm_{run_id}_x")
            os.makedirs(runner_module.run_download_dir(workspace))
            artifact = os.path.join(runner_module.run_download_dir(workspace), "artifact")
            open(artifact, "wb").close()
            registry._store_download(run_id, SmartRunEntry(
                file_path=artifact, file_name="out.zip", is_zip=True, temp_dir=workspace,
                created_at=time.time(), packaged=packaged))
            name = os.path.basename(runner_module.run_download_dir(workspace))
            restored = registry._entry_from_download_dir(str(tmp_path), name, time.time())
            assert restored is not None and restored[1].packaged is packaged

    def test_a_download_from_before_the_packaged_flag_is_known_by_its_name(self, tmp_path):
        registry = runner_module.SmartRunRegistry()
        for run_id, file_name, packaged in (("3" * 32, f"besser_smart_{'3' * 32}.zip", True),
                                            ("4" * 32, f"besser_smart_{'3' * 32}.zip", False),
                                            ("5" * 32, "generated.zip", False)):
            workspace = str(tmp_path / f"besser_llm_{run_id}_x")
            download = runner_module.run_download_dir(workspace)
            os.makedirs(download)
            open(os.path.join(download, "artifact"), "wb").close()
            with open(os.path.join(download, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump({"version": 2, "runId": run_id, "fileName": file_name, "isZip": True,
                           "createdAt": time.time()}, fh)
            restored = registry._entry_from_download_dir(str(tmp_path), os.path.basename(download), time.time())
            assert restored is not None and restored[1].packaged is packaged, file_name


# ---------------------------------------------------------------------
# (e) Base-expired fallback
# ---------------------------------------------------------------------


class TestBaseExpiredFallback:
    def test_expired_base_warns_and_runs_from_scratch(
        self, tmp_path, stub_orchestrator
    ):
        missing_id = "f" * 32  # never registered
        request = _build_request(mode="modify", base_run_id=missing_id)
        runner = SmartGenerationRunner(
            request, runtime=current_runtime(), base_run_id=missing_id, mode="modify",
        )
        frames = asyncio.run(_collect_frames(runner))
        parsed = [_parse(f) for f in frames]

        # A non-terminal INCOMPLETE warning is emitted about the expiry.
        warnings = [
            p for p in parsed
            if p["event"] == "error" and p["code"] == "INCOMPLETE"
        ]
        assert warnings, "expected an INCOMPLETE warning for the expired base"
        assert "expired" in warnings[0]["message"].lower()

        # It fell back to from-scratch run() (never modify), and still
        # produced a downloadable result — no crash.
        assert any(m == "run" for m, _ in _CALLS)
        assert not any(m == "modify" for m, _ in _CALLS)
        assert parsed[-1]["event"] == "done"
        # The result says it is a new app, not an edit.
        assert parsed[-1]["mode"] == "create"
        assert runner._facts.mode_reason == "previous result expired, rebuilt from scratch"
