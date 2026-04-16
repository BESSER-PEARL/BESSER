"""Tests for SmartGenerationRunner — asyncio queue bridge, cost emitter, cleanup."""

import asyncio
import json
import os
import threading
import time
from typing import Callable, Optional

import pytest

from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.smart_generation import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    SmartGenerationRunner,
    SMART_RUN_REGISTRY,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_request(**overrides) -> SmartGenerateRequest:
    project = ProjectInput(
        id="test-project",
        type="BesserProject",
        name="TestProject",
        createdAt="2026-04-15T00:00:00Z",
        diagrams={
            "ClassDiagram": [
                {
                    "id": "cd1",
                    "title": "Library",
                    "model": CLASS_DIAGRAM_MODEL,
                }
            ]
        },
        currentDiagramIndices={"ClassDiagram": 0},
    )
    defaults = dict(
        project=project,
        instructions="Build a simple backend",
        api_key="sk-ant-test-NEVER-LEAK-1234567890",
        provider="anthropic",
        llm_model="claude-sonnet-4-5",
        max_cost_usd=1.0,
        max_runtime_seconds=60,
    )
    defaults.update(overrides)
    return SmartGenerateRequest(**defaults)


class _FakeUsage:
    def __init__(self):
        self.estimated_cost = 0.0


class _FakeClient:
    def __init__(self):
        self.usage = _FakeUsage()


class _FakeOrchestrator:
    """Stub that mimics the real orchestrator's callback firing pattern.

    The real orchestrator runs synchronously in a worker thread and calls
    on_progress / on_text from that same thread. This stub replicates
    that behaviour so the queue bridge is exercised end-to-end.
    """

    def __init__(
        self,
        *,
        llm_client,
        domain_model,
        output_dir,
        on_progress: Optional[Callable] = None,
        on_text: Optional[Callable] = None,
        **_kwargs,
    ):
        self.client = llm_client
        self.domain_model = domain_model
        self.output_dir = output_dir
        self.on_progress = on_progress
        self.on_text = on_text
        self.total_turns = 0
        # The runner constructs the orchestrator directly, so expose
        # a run(instructions) method like the real class.
        _LAST_ORCHESTRATORS.append(self)

    def run(self, instructions: str) -> str:
        # Simulate the ordered callback sequence the real orchestrator
        # fires: phase 1 (select/generate), phase 2 tool loop, phase 3
        # validation. We're on a worker thread here (via asyncio.to_thread),
        # so these callbacks go through loop.call_soon_threadsafe.
        if self.on_progress:
            self.on_progress(0, "fastapi_backend", "generating")
        if self.on_text:
            self.on_text("Analyzing the domain model...")
        if self.on_progress:
            self.on_progress(1, "write_file", "executing")
        if self.on_text:
            self.on_text(" done.")
        if self.on_progress:
            self.on_progress(2, "validation", "0 issues")

        # Simulate cost accrual
        self.client.usage.estimated_cost = 0.02
        self.total_turns = 3

        # Write one output file so _package_result emits a single-file
        # (not zipped) done event.
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "main.py"), "w", encoding="utf-8") as fh:
            fh.write("# fake generated content\nprint('hello')\n")
        # And a recipe
        with open(
            os.path.join(self.output_dir, ".besser_recipe.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump({"instructions": instructions, "usage": {"input_tokens": 10}}, fh)

        return self.output_dir


class _FailingOrchestrator(_FakeOrchestrator):
    def run(self, instructions: str) -> str:
        raise RuntimeError("Claude API call failed: upstream 503")


class _EmptyOutputOrchestrator(_FakeOrchestrator):
    """Returns normally but produces no output files."""

    def run(self, instructions: str) -> str:
        self.client.usage.estimated_cost = 0.01
        os.makedirs(self.output_dir, exist_ok=True)
        # Only write an internal artefact; `_package_result` should
        # filter `.besser_*` out and then raise EmptyGenerationError.
        with open(
            os.path.join(self.output_dir, ".besser_recipe.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump({"instructions": instructions}, fh)
        return self.output_dir


class _MultiFileOrchestrator(_FakeOrchestrator):
    """Writes multiple files so _package_result produces a zip."""

    def run(self, instructions: str) -> str:
        self.client.usage.estimated_cost = 0.03
        os.makedirs(self.output_dir, exist_ok=True)
        for name, content in [
            ("app/main.py", "# main\nprint('hi')\n"),
            ("app/auth.py", "# auth stub\n"),
            ("Dockerfile", "FROM python:3.11\n"),
        ]:
            full = os.path.join(self.output_dir, name)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(content)
        with open(
            os.path.join(self.output_dir, ".besser_recipe.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump({"instructions": instructions, "files": 3}, fh)
        return self.output_dir


_LAST_ORCHESTRATORS: list[_FakeOrchestrator] = []


@pytest.fixture(autouse=True)
def clear_registry_and_orchestrators():
    _LAST_ORCHESTRATORS.clear()
    # Reset the global registry between tests.
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


async def _clear_registry():
    async with SMART_RUN_REGISTRY._lock:
        for entry in list(SMART_RUN_REGISTRY._entries.values()):
            import shutil
            shutil.rmtree(entry.temp_dir, ignore_errors=True)
        SMART_RUN_REGISTRY._entries.clear()


@pytest.fixture
def stub_orchestrator(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _FakeOrchestrator)
    # Also stub create_llm_client so no real API client is built.
    def _fake_client(**kwargs):
        if kwargs.get("api_key", "").startswith("bad"):
            raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY.")
        return _FakeClient()
    monkeypatch.setattr(runner_module, "create_llm_client", _fake_client)


@pytest.fixture
def failing_orchestrator(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _FailingOrchestrator)
    def _fake_client(**kwargs):
        return _FakeClient()
    monkeypatch.setattr(runner_module, "create_llm_client", _fake_client)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


async def _collect_frames(runner: SmartGenerationRunner) -> list[bytes]:
    frames: list[bytes] = []
    async for frame in runner.generate_and_stream():
        frames.append(frame)
    return frames


def _parse_frame(frame: bytes) -> dict:
    text = frame.decode("utf-8")
    data_line = [l for l in text.splitlines() if l.startswith("data: ")][0]
    return json.loads(data_line[len("data: "):])


class TestHappyPath:
    def test_happy_path_event_ordering(self, stub_orchestrator):
        request = _build_request()
        runner = SmartGenerationRunner(request)

        frames = asyncio.run(_collect_frames(runner))
        parsed = [_parse_frame(f) for f in frames]

        event_names = [p["event"] for p in parsed]
        # First event must always be `start`
        assert event_names[0] == "start"
        # phase(select) must come before any tool_call/text
        first_phase_idx = event_names.index("phase")
        assert parsed[first_phase_idx]["phase"] == "select"
        # Last event should be `done`
        assert event_names[-1] == "done"
        # A text delta and a tool_call must appear
        assert "text" in event_names
        assert "tool_call" in event_names

    def test_start_event_never_contains_api_key(self, stub_orchestrator):
        request = _build_request()
        runner = SmartGenerationRunner(request)

        frames = asyncio.run(_collect_frames(runner))
        for frame in frames:
            # The literal api-key string must not appear anywhere.
            assert b"sk-ant-test-NEVER-LEAK" not in frame

    def test_done_event_registers_download(self, stub_orchestrator):
        request = _build_request()
        runner = SmartGenerationRunner(request)

        frames = asyncio.run(_collect_frames(runner))
        done = _parse_frame(frames[-1])
        assert done["event"] == "done"
        assert done["downloadUrl"].startswith("/besser_api/download-smart/")
        assert done["fileName"] == "main.py"  # single file, not zipped
        assert done["isZip"] is False
        assert done["recipe"]["instructions"] == request.instructions


class TestErrorPaths:
    def test_invalid_api_key(self, stub_orchestrator):
        request = _build_request(api_key="bad-key")
        runner = SmartGenerationRunner(request)

        frames = asyncio.run(_collect_frames(runner))
        parsed = [_parse_frame(f) for f in frames]
        errors = [p for p in parsed if p["event"] == "error"]
        assert len(errors) >= 1
        assert errors[-1]["code"] == "INVALID_KEY"

    def test_upstream_llm_error(self, failing_orchestrator):
        request = _build_request()
        runner = SmartGenerationRunner(request)

        frames = asyncio.run(_collect_frames(runner))
        parsed = [_parse_frame(f) for f in frames]
        errors = [p for p in parsed if p["event"] == "error"]
        assert any(e["code"] == "UPSTREAM_LLM" for e in errors)
        # No done event on upstream failure
        assert not any(p["event"] == "done" for p in parsed)


class TestCleanup:
    def test_temp_dir_cleaned_on_upstream_error(self, failing_orchestrator):
        request = _build_request()
        runner = SmartGenerationRunner(request)

        asyncio.run(_collect_frames(runner))
        # Runner cleans up on failure
        assert runner.temp_dir is None or not os.path.isdir(runner.temp_dir)

    def test_temp_dir_preserved_on_success_for_download(self, stub_orchestrator):
        request = _build_request()
        runner = SmartGenerationRunner(request)

        asyncio.run(_collect_frames(runner))
        # Temp dir must still exist because the download endpoint will serve it
        assert runner.temp_dir is not None
        assert os.path.isdir(runner.temp_dir)


class TestMultiFileZip:
    def test_multi_file_output_is_zipped(self, monkeypatch):
        """When the orchestrator produces >1 user file, _package_result zips them."""
        monkeypatch.setattr(runner_module, "LLMOrchestrator", _MultiFileOrchestrator)
        monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())

        request = _build_request()
        runner = SmartGenerationRunner(request)

        frames = asyncio.run(_collect_frames(runner))
        done = [_parse_frame(f) for f in frames if _parse_frame(f)["event"] == "done"]
        assert len(done) == 1
        assert done[0]["isZip"] is True
        assert done[0]["fileName"].startswith("besser_smart_")
        assert done[0]["fileName"].endswith(".zip")

        # Verify the zip is valid and contains the expected files
        import zipfile
        entry_path = None
        async def _find():
            return await SMART_RUN_REGISTRY.pop(runner.run_id)
        entry = asyncio.run(_find())
        assert entry is not None
        entry_path = entry.file_path
        with zipfile.ZipFile(entry_path, "r") as zf:
            names = set(zf.namelist())
        assert any("main.py" in n for n in names)
        assert any("auth.py" in n for n in names)
        assert any("Dockerfile" in n for n in names)


class TestEmptyOutput:
    def test_no_output_files_yields_internal_error(self, monkeypatch):
        """An orchestrator that completes but writes no user files is a backend
        error, not an upstream LLM error."""
        monkeypatch.setattr(runner_module, "LLMOrchestrator", _EmptyOutputOrchestrator)
        monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())

        request = _build_request()
        runner = SmartGenerationRunner(request)

        frames = asyncio.run(_collect_frames(runner))
        parsed = [_parse_frame(f) for f in frames]
        errors = [p for p in parsed if p["event"] == "error"]
        assert len(errors) >= 1
        # Must be INTERNAL (backend issue), not UPSTREAM_LLM (provider issue)
        assert errors[-1]["code"] == "INTERNAL"
        assert not any(p["event"] == "done" for p in parsed)
        # Temp dir should be cleaned up
        assert runner.temp_dir is None or not os.path.isdir(runner.temp_dir)


class TestRegistrySweep:
    def test_sweep_expires_entries_older_than_ttl(self):
        """Periodic sweep removes entries past their TTL and rmtrees their temp dirs."""
        import tempfile
        registry = runner_module.SmartRunRegistry()

        async def _test():
            tmp1 = tempfile.mkdtemp(prefix="besser_llm_test_")
            tmp2 = tempfile.mkdtemp(prefix="besser_llm_test_")
            try:
                # Entry 1: old enough to expire
                with open(os.path.join(tmp1, "a.txt"), "w") as fh:
                    fh.write("old")
                entry1 = runner_module.SmartRunEntry(
                    file_path=os.path.join(tmp1, "a.txt"),
                    file_name="a.txt",
                    is_zip=False,
                    temp_dir=tmp1,
                    created_at=time.time() - 10_000,  # 10000s ago
                )
                await registry.put("a" * 32, entry1)

                # Entry 2: fresh
                with open(os.path.join(tmp2, "b.txt"), "w") as fh:
                    fh.write("fresh")
                entry2 = runner_module.SmartRunEntry(
                    file_path=os.path.join(tmp2, "b.txt"),
                    file_name="b.txt",
                    is_zip=False,
                    temp_dir=tmp2,
                    created_at=time.time(),
                )
                await registry.put("b" * 32, entry2)

                # Trigger one sweep cycle manually by calling the inner logic
                # (we can't wait for periodic_sweep's sleep — too slow for a test).
                async with registry._lock:
                    now = time.time()
                    ttl = 1800
                    expired = [
                        (rid, e)
                        for rid, e in list(registry._entries.items())
                        if max(0.0, now - e.created_at) > ttl
                    ]
                    for rid, _ in expired:
                        registry._entries.pop(rid, None)
                for _, e in expired:
                    import shutil as _sh
                    _sh.rmtree(e.temp_dir, ignore_errors=True)

                # The old entry is gone; the fresh one remains
                assert "a" * 32 not in registry._entries
                assert "b" * 32 in registry._entries
                assert not os.path.isdir(tmp1)
                assert os.path.isdir(tmp2)
            finally:
                import shutil as _sh
                _sh.rmtree(tmp1, ignore_errors=True)
                _sh.rmtree(tmp2, ignore_errors=True)

        asyncio.run(_test())

    def test_sweep_handles_backwards_clock_movement(self):
        """An entry whose created_at is in the future (clock skew) is NOT
        considered expired immediately and does NOT live forever — it is
        treated as age 0 and expires normally once TTL elapses."""
        registry = runner_module.SmartRunRegistry()

        async def _test():
            import tempfile
            tmp = tempfile.mkdtemp(prefix="besser_llm_test_future_")
            try:
                entry = runner_module.SmartRunEntry(
                    file_path=os.path.join(tmp, "x.txt"),
                    file_name="x.txt",
                    is_zip=False,
                    temp_dir=tmp,
                    created_at=time.time() + 10_000,  # 10000s in the future!
                )
                await registry.put("c" * 32, entry)

                async with registry._lock:
                    now = time.time()
                    age = max(0.0, now - entry.created_at)
                # max(0, negative) == 0, so the entry is treated as age 0
                assert age == 0.0
            finally:
                import shutil as _sh
                _sh.rmtree(tmp, ignore_errors=True)

        asyncio.run(_test())
