"""The generator's stored output and its model ride with the run (#659, stage 3).

A modify run merges a model change into edited generated files against the
output the generator wrote. That output must come back with the run: in the
ZIP, in the GitHub push (test_github_round_trip.py) and in the seed. After a
BESSER upgrade the merge base is rebuilt from the model the output came from,
so that model is stored with it, exactly as the run received it.
"""
import asyncio
import json
import os
import zipfile

from besser.generators.backend import BackendGenerator
from besser.spec_driven_agent.state.generator_output import (
    GENERATOR_OUTPUT_FILENAME,
    attach_model,
    store_generator_output,
    stored_model,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.github_sync import _hidden_from_run
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.model_assembly import (
    assemble_models_from_project,
    class_diagram_model,
    generation_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.settings import get_settings
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _build_request,
    _collect_frames,
    _FakeClient,
    _MultiFileOrchestrator,
)


class _StoringOrchestrator(_MultiFileOrchestrator):
    """Writes a multi-file app and a stored generator output, as Phase 1 does."""

    _updated_project_export = None  # as LLMOrchestrator: set only by a model sync

    def run(self, instructions: str) -> str:
        out = super().run(instructions)
        store_generator_output(out, "backend", {"backend/main_api.py": b"app = 1\n"}, "8.0.0")
        return out


def _generate(model, out) -> dict:
    BackendGenerator(model=model, output_dir=str(out)).generate()
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_the_zip_carries_the_stored_output_with_the_runs_class_diagram(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _StoringOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    request = _build_request()
    runner = SmartGenerationRunner(request, runtime=current_runtime())

    asyncio.run(_collect_frames(runner))

    entry = asyncio.run(current_runtime().registry.pop(runner.run_id))
    with zipfile.ZipFile(entry.file_path) as zf:
        names = set(zf.namelist())
        stored = json.loads(zf.read(GENERATOR_OUTPUT_FILENAME))
    assert {GENERATOR_OUTPUT_FILENAME, ".besser_recipe.json"} <= names
    assert stored["files"] == {"backend/main_api.py": "app = 1\n"}
    assert stored["model"] == generation_class_diagram(request.project)


def test_the_stored_class_diagram_generates_what_the_run_generated(tmp_path):
    """The diagram is stored as the run received it: a round trip through the
    model reorders attributes and literals, and the rebuilt base would differ."""
    project = _build_request().project
    received = assemble_models_from_project(project).domain_model
    stored = json.loads(json.dumps(generation_class_diagram(project)))

    rebuilt = class_diagram_model(stored)

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert _generate(received, tmp_path / "a") == _generate(rebuilt, tmp_path / "b")


class _SyncingOrchestrator(_StoringOrchestrator):
    """A modify run whose model sync added classes: its export, not the request's diagram, generated the app."""

    def run(self, instructions: str) -> str:
        export = _build_request().project.model_dump(mode="json")
        export["diagrams"]["ClassDiagram"][0]["title"] = "Synced"
        self._updated_project_export = export
        return super().run(instructions)


def test_a_model_synced_run_stores_the_diagram_it_generated_from(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _SyncingOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())

    asyncio.run(_collect_frames(runner))

    entry = asyncio.run(current_runtime().registry.pop(runner.run_id))
    with zipfile.ZipFile(entry.file_path) as zf:
        assert json.loads(zf.read(GENERATOR_OUTPUT_FILENAME))["model"]["title"] == "Synced"


def test_a_store_this_run_did_not_write_keeps_the_model_it_came_from(tmp_path):
    """A run whose regeneration was skipped must not label the older output with its new model."""
    store_generator_output(str(tmp_path), "backend", {"backend/main_api.py": b"app = 1\n"}, "8.0.0")
    attach_model(str(tmp_path), {"title": "Before"})

    attach_model(str(tmp_path), {"title": "After"})

    assert stored_model(str(tmp_path)) == {"title": "Before"}


def test_a_seeded_modify_run_reads_the_previous_model(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_run_workspace_root", str(tmp_path.parent))
    project = _build_request().project
    store_generator_output(str(tmp_path), "backend", {"backend/main_api.py": b"app = 1\n"}, "8.0.0")
    attach_model(str(tmp_path), generation_class_diagram(project))
    runner = SmartGenerationRunner(_build_request(), mode="modify", runtime=current_runtime())
    runner.temp_dir, runner._seeded = str(tmp_path), True

    previous = runner._previous_domain_model()

    assert {c.name for c in previous.get_classes()} == {
        c.name for c in assemble_models_from_project(project).domain_model.get_classes()}


def test_the_orchestrator_receives_the_previous_model(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _StoringOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    monkeypatch.setattr(SmartGenerationRunner, "_previous_domain_model", lambda self: sentinel)
    received = {}
    original = _StoringOrchestrator.__init__

    def capture(self, **kwargs):
        received.update(kwargs)
        original(self, **kwargs)

    monkeypatch.setattr(_StoringOrchestrator, "__init__", capture)
    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())

    asyncio.run(_collect_frames(runner))

    asyncio.run(current_runtime().registry.pop(runner.run_id))
    assert received["previous_domain_model"] is sentinel


def test_a_stored_class_diagram_that_does_not_convert_is_reported(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_run_workspace_root", str(tmp_path.parent))
    store_generator_output(str(tmp_path), "backend", {"backend/main_api.py": b"app = 1\n"}, "8.0.0")
    store = json.loads((tmp_path / GENERATOR_OUTPUT_FILENAME).read_text(encoding="utf-8"))
    store["model"] = {"model": "not a diagram"}
    (tmp_path / GENERATOR_OUTPUT_FILENAME).write_text(json.dumps(store), encoding="utf-8")
    runner = SmartGenerationRunner(_build_request(), mode="modify", runtime=current_runtime())
    runner.temp_dir, runner._seeded = str(tmp_path), True

    assert runner._previous_domain_model() is None
    assert "cannot rebuild the merge base" in caplog.text


def test_a_run_owns_the_files_it_carries_but_not_its_other_records():
    """A carried file missing from the run is deleted from the repo on push; the
    checkpoint and snapshot never reach it, so their absence deletes nothing."""
    assert not _hidden_from_run(GENERATOR_OUTPUT_FILENAME, [])
    assert not _hidden_from_run(".besser_attribution.json", [])
    assert _hidden_from_run(".besser_checkpoint.json", [])
    assert _hidden_from_run(os.path.join(".besser_snapshot", "x.py").replace(os.sep, "/"), [])


def test_a_workspace_outside_the_run_root_is_not_read(tmp_path, monkeypatch, caplog):
    """A resumed run finds its workspace from the request's run id; a path that resolves
    outside the shared run root is never read."""
    monkeypatch.setattr(get_settings(), "llm_run_workspace_root", str(tmp_path / "runs"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    store_generator_output(str(elsewhere), "backend", {"backend/main_api.py": b"app = 1\n"}, "8.0.0")
    attach_model(str(elsewhere), generation_class_diagram(_build_request().project))
    runner = SmartGenerationRunner(_build_request(), mode="modify", runtime=current_runtime())
    runner.temp_dir, runner._seeded = str(elsewhere), True

    assert runner._previous_domain_model() is None
    assert "outside the run root" in caplog.text
