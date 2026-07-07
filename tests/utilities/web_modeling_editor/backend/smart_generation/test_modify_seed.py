"""Tests for the runner-side incremental vibe-modify seed arm.

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

from besser.generators.llm.checkpoint import CHECKPOINT_FILENAME
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    SMART_RUN_REGISTRY,
    SmartGenerationRunner,
    SmartRunEntry,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)


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

    async with SMART_RUN_REGISTRY._lock:
        for entry in list(SMART_RUN_REGISTRY._entries.values()):
            shutil.rmtree(entry.temp_dir, ignore_errors=True)
        SMART_RUN_REGISTRY._entries.clear()


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
        asyncio.run(SMART_RUN_REGISTRY.put(
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
            request, base_run_id=base_id, mode="modify",
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

        # The base is untouched and still fully downloadable.
        assert os.path.isfile(os.path.join(base_dir, "app", "main.py"))
        assert os.path.isfile(os.path.join(base_dir, CHECKPOINT_FILENAME))
        assert os.path.isdir(os.path.join(base_dir, ".besser_snapshot"))
        assert os.path.isdir(os.path.join(base_dir, "node_modules"))
        base_entry = asyncio.run(SMART_RUN_REGISTRY.get(base_id))
        assert base_entry is not None and os.path.isdir(base_entry.temp_dir)


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
            request, base_run_id=missing_id, mode="modify",
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
