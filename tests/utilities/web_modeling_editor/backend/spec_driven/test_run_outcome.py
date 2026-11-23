"""The run store, telemetry and the done event say how a run ended in the run record's words.

A run that finished with a requirement found missing is ``completed_with_blockers`` in its
run record, yet ``GET /runs/{id}`` said ``succeeded`` and its telemetry said ``success``:
both keyed on ``incomplete`` alone, which such a blocker never sets. The done event now
carries the record's ``outcome``, and the store and telemetry derive from it, adding values
rather than renaming the ones clients compare against.
"""

import asyncio
from types import SimpleNamespace

import pytest

from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.run_manager import (
    TERMINAL_RUN_STATUSES,
    DurableRunManager,
    SqliteRunEventStore,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import SmartGenerationRunner
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime
from tests.utilities.web_modeling_editor.backend.spec_driven.test_durable_runs import _start
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _clear_registry,
    _collect_frames,
    _parse_frame,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_telemetry import (
    _ClientWithTokens,
    _read_events,
    _SplitOrchestrator,
    _telemetry_request,
)

MISSING = "requirement: R1 — An order total is never negative — the delivered code does not implement it"


class _MissingRequirementOrchestrator(_SplitOrchestrator):
    """A finished run whose one blocker is a requirement the judge found missing."""

    def run(self, instructions: str) -> str:
        out = super().run(instructions)
        self._validation_issues = [SimpleNamespace(severity="blocker", message=MISSING)]
        return out


@pytest.fixture(autouse=True)
def reset_registry():
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


@pytest.fixture
def telemetry_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("BESSER_TELEMETRY_ENABLED", "1")
    monkeypatch.setenv("BESSER_TELEMETRY_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def missing_requirement(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _MissingRequirementOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _ClientWithTokens())


def _stored_status(done_frame: bytes) -> str:
    async def exercise() -> str:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "f" * 32

        async def source():
            yield _start(run_id)
            yield done_frame

        await manager.start(run_id, source())
        await manager.wait(run_id)
        status = manager.get_run(run_id).status
        store.close()
        return status

    return asyncio.run(exercise())


def test_a_run_with_a_requirement_blocker_is_not_reported_as_success(missing_requirement, telemetry_dir):
    runner = SmartGenerationRunner(_telemetry_request(), runtime=current_runtime())
    frames = asyncio.run(_collect_frames(runner))

    [done_frame] = [f for f in frames if _parse_frame(f)["event"] == "done"]
    done = _parse_frame(done_frame)
    assert done["incomplete"] is False
    assert done["outcome"] == "completed_with_blockers"

    [summary] = [e for e in _read_events(telemetry_dir, "study-sess-1") if e["kind"] == "run_summary"]
    assert summary["payload"]["outcome"] == "completed_with_blockers"

    status = _stored_status(done_frame)
    assert status == "completed_with_blockers"
    assert status in TERMINAL_RUN_STATUSES


@pytest.mark.parametrize("outcome, status", [
    ("completed", "succeeded"),
    ("completed_unverified", "completed_unverified"),
    (None, "succeeded"),  # an event written before the field existed
])
def test_the_stored_status_follows_the_outcome(outcome, status):
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import DoneEvent, format_sse

    frame = format_sse(DoneEvent(runId="f" * 32, downloadUrl="/d", fileName="r.zip", isZip=True, outcome=outcome))
    assert _stored_status(frame) == status
