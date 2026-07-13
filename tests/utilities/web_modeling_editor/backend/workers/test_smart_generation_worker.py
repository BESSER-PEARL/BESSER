from __future__ import annotations

import asyncio
import io
import json
import tempfile
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import pytest

from besser.generators.llm.checkpoint import CHECKPOINT_FILENAME
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_jobs import (
    canonical_request_hash,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_state import (
    DurableStateConfig,
    DurableStateFoundation,
    DurableStateMode,
    FileSystemBlobStore,
    InMemoryJobQueue,
    InMemoryStateStore,
    JobMessage,
    OptimisticLockError,
    ReplayCursor,
    RunRecord,
    RunStatus,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    SMART_RUN_REGISTRY,
    SmartRunEntry,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.secret_envelope import (
    LocalSecretEnvelope,
)
from besser.utilities.web_modeling_editor.backend.workers.smart_generation_worker import (
    WorkerDisposition,
    _EcsTaskProtection,
    _TaskProtectionError,
    _build_task_protection,
    process_one_job,
    run_worker_until_job,
)


OWNER_ID = "github:1234"


def _request(**overrides: Any) -> SmartGenerateRequest:
    values: dict[str, Any] = {
        "project": ProjectInput(
            id="project-1",
            type="BesserProject",
            name="Worker Test",
            createdAt="2026-07-13T00:00:00Z",
            diagrams={},
        ),
        "instructions": "Build a small service",
        "api_key": "sk-test-never-log-123456",
        "provider": "openai",
        "llm_model": "gpt-4o-mini",
        "max_cost_usd": 1.0,
        "max_runtime_seconds": 60,
    }
    values.update(overrides)
    return SmartGenerateRequest(**values)


def _foundation(
    tmp_path: Path,
    *,
    state: InMemoryStateStore | None = None,
    queue: InMemoryJobQueue | None = None,
    max_storage_bytes: int = 1024 * 1024,
    lease_ttl_seconds: int = 2,
) -> DurableStateFoundation:
    config = DurableStateConfig.from_env({
        "BESSER_SMARTGEN_STATE_MODE": "local",
        "BESSER_SMARTGEN_SQLITE_PATH": str(tmp_path / "state.sqlite3"),
        "BESSER_SMARTGEN_STORAGE_DIR": str(tmp_path / "blobs"),
        "BESSER_SMARTGEN_LEASE_TTL_SECONDS": str(lease_ttl_seconds),
        "BESSER_SMARTGEN_MAX_STORAGE_BYTES_PER_OWNER": str(max_storage_bytes),
    })
    blobs = FileSystemBlobStore(str(tmp_path / "blobs"))
    return DurableStateFoundation(
        state=state or InMemoryStateStore(),
        artifacts=blobs,
        checkpoints=blobs,
        queue=queue or InMemoryJobQueue(),
        config=config,
    )


async def _enqueue(
    foundation: DurableStateFoundation,
    envelope: LocalSecretEnvelope,
    request: SmartGenerateRequest,
    *,
    run_id: str,
    status: RunStatus = RunStatus.QUEUED,
    metadata_overrides: dict[str, Any] | None = None,
) -> RunRecord:
    reservation_id = f"concurrent:{run_id}:test"
    quota = await foundation.state.reserve_quota(
        OWNER_ID,
        "concurrent-runs",
        reservation_id,
        amount=1,
        limit=1,
        ttl_seconds=3600,
    )
    assert quota.allowed
    metadata: dict[str, Any] = {
        "request": request.model_dump(mode="json", exclude={"api_key"}),
        "api_key_envelope": envelope.encrypt(
            request.resolved_api_key(),
            run_id=run_id,
            owner_id=OWNER_ID,
        ).to_dict(),
        "concurrency_reservation": reservation_id,
    }
    metadata.update(metadata_overrides or {})
    record = RunRecord(
        run_id=run_id,
        owner_id=OWNER_ID,
        request_hash=canonical_request_hash(request, OWNER_ID),
        status=status,
        mode=request.mode,
        provider=request.provider,
        model=request.llm_model,
        max_cost_usd=request.max_cost_usd,
        max_runtime_seconds=request.max_runtime_seconds,
        metadata=metadata,
    )
    await foundation.state.create_run(record)
    await foundation.queue.enqueue(JobMessage(run_id, OWNER_ID, {}))
    return record


def _frame(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


class _ScriptRunner:
    def __init__(
        self,
        request: SmartGenerateRequest,
        *,
        script: Callable[["_ScriptRunner"], AsyncIterator[dict[str, Any]]],
        **options: Any,
    ) -> None:
        self.request = request
        self.run_id = options.get("run_id") or options.get("resume_run_id")
        self.owner_id = options["owner_id"]
        self.temp_dir: str | None = None
        self.options = options
        self._script = script

    async def generate_and_stream(self, http_request: Any = None):
        del http_request
        async for payload in self._script(self):
            yield _frame(payload)


class _RecordingTaskProtection:
    def __init__(
        self,
        events: list[str],
        *,
        fail_enable: bool = False,
        fail_refresh: bool = False,
    ) -> None:
        self.events = events
        self.fail_enable = fail_enable
        self.fail_refresh = fail_refresh
        self.active = False

    async def enable(self) -> None:
        self.events.append("enable")
        self.active = True
        if self.fail_enable:
            raise _TaskProtectionError("agent unavailable")

    async def refresh(self) -> None:
        if self.active:
            self.events.append("refresh")
            if self.fail_refresh:
                raise _TaskProtectionError("agent refresh unavailable")

    async def disable(self) -> None:
        if self.active:
            self.events.append("disable")
            self.active = False


async def _register_output(
    runner: _ScriptRunner,
    *,
    incomplete: bool = False,
    checkpoint: bool = False,
) -> dict[str, Any]:
    workspace = Path(tempfile.mkdtemp(prefix=f"worker_test_{runner.run_id}_"))
    runner.temp_dir = str(workspace)
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    if checkpoint:
        (workspace / CHECKPOINT_FILENAME).write_text(
            '{"turn": 2}', encoding="utf-8",
        )
        (workspace / "node_modules").mkdir()
        (workspace / "node_modules" / "skip.js").write_text("skip")
    artifact_path = workspace / "result.zip"
    artifact_path.write_bytes(b"generated-artifact")
    await SMART_RUN_REGISTRY.put(
        runner.run_id,
        SmartRunEntry(
            file_path=str(artifact_path),
            file_name="result.zip",
            is_zip=True,
            temp_dir=str(workspace),
            created_at=time.time(),
            owner_id=runner.owner_id,
        ),
    )
    return {
        "event": "done",
        "runId": runner.run_id,
        "downloadUrl": f"/besser_api/download-smart/{runner.run_id}",
        "fileName": "result.zip",
        "isZip": True,
        "recipe": {},
        "incomplete": incomplete,
        "incompleteReason": "more work remains" if incomplete else None,
    }


async def _events(foundation: DurableStateFoundation, run_id: str):
    page = await foundation.state.read_events(
        run_id,
        cursor=ReplayCursor(run_id),
        limit=1000,
    )
    return list(page.events)


def test_worker_protects_paid_execution_refreshes_and_clears(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        run_id = "c2" * 16
        await _enqueue(foundation, envelope, _request(), run_id=run_id)
        lifecycle: list[str] = []
        protection = _RecordingTaskProtection(lifecycle)

        async def script(runner: _ScriptRunner):
            lifecycle.append("runner")
            await asyncio.sleep(0.06)
            yield {"event": "start", "runId": runner.run_id}
            yield await _register_output(runner)

        result = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=lambda request, **options: _ScriptRunner(
                request,
                script=script,
                **options,
            ),
            heartbeat_interval_seconds=0.01,
            task_protection=protection,
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert result.disposition == WorkerDisposition.ACKNOWLEDGED
        assert lifecycle[0:2] == ["enable", "runner"]
        assert "refresh" in lifecycle
        assert lifecycle[-1] == "disable"
        record = await foundation.state.get_run(run_id)
        assert record is not None and record.status == RunStatus.SUCCEEDED
        assert record.metadata["execution_started"] is True
        await foundation.close()

    asyncio.run(scenario())


def test_task_protection_failure_prevents_paid_generation(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        run_id = "c3" * 16
        await _enqueue(foundation, envelope, _request(), run_id=run_id)
        lifecycle: list[str] = []
        protection = _RecordingTaskProtection(lifecycle, fail_enable=True)
        runner_started = False

        async def script(runner: _ScriptRunner):
            nonlocal runner_started
            runner_started = True
            yield {"event": "start", "runId": runner.run_id}

        result = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=lambda request, **options: _ScriptRunner(
                request,
                script=script,
                **options,
            ),
            task_protection=protection,
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert result.disposition == WorkerDisposition.RETRY_RELEASED
        assert runner_started is False
        assert lifecycle == ["enable", "disable"]
        record = await foundation.state.get_run(run_id)
        assert record is not None and record.status == RunStatus.RUNNING
        assert record.metadata["execution_started"] is False
        await foundation.close()

    asyncio.run(scenario())


def test_task_protection_refresh_failure_releases_job_for_recovery(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        run_id = "c4" * 16
        await _enqueue(foundation, envelope, _request(), run_id=run_id)
        lifecycle: list[str] = []
        protection = _RecordingTaskProtection(
            lifecycle,
            fail_refresh=True,
        )

        async def script(runner: _ScriptRunner):
            await asyncio.sleep(0.05)
            yield {"event": "start", "runId": runner.run_id}

        result = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=lambda request, **options: _ScriptRunner(
                request,
                script=script,
                **options,
            ),
            heartbeat_interval_seconds=0.01,
            task_protection=protection,
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert result.disposition == WorkerDisposition.RETRY_RELEASED
        assert lifecycle[0] == "enable"
        assert "refresh" in lifecycle
        assert lifecycle[-1] == "disable"
        record = await foundation.state.get_run(run_id)
        assert record is not None and record.status == RunStatus.RUNNING
        assert record.metadata["execution_started"] is True
        await foundation.close()

    asyncio.run(scenario())


def test_ecs_task_protection_request_contract_and_bounded_timeout(monkeypatch):
    class Response:
        status = 200

        def __init__(self, enabled: bool) -> None:
            self.enabled = enabled

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int) -> bytes:
            return json.dumps({
                "protection": {"ProtectionEnabled": self.enabled},
            }).encode()

    calls: list[tuple[str, dict[str, Any], float]] = []

    class Opener:
        def open(self, request, *, timeout: float):
            payload = json.loads(request.data.decode())
            calls.append((request.full_url, payload, timeout))
            return Response(payload["ProtectionEnabled"])

    from besser.utilities.web_modeling_editor.backend.workers import (
        smart_generation_worker as worker_module,
    )

    monkeypatch.setattr(
        worker_module.urllib.request,
        "build_opener",
        lambda *_args: Opener(),
    )
    client = _EcsTaskProtection(
        "http://169.254.170.2/api/task-token",
        expires_in_minutes=7,
        request_timeout_seconds=0.1,
    )

    async def lifecycle() -> None:
        await client.enable()
        client.refresh_interval_seconds = 0
        await client.refresh()
        await client.disable()

    asyncio.run(lifecycle())
    assert [payload for _, payload, _ in calls] == [
        {"ProtectionEnabled": True, "ExpiresInMinutes": 7},
        {"ProtectionEnabled": True, "ExpiresInMinutes": 7},
        {"ProtectionEnabled": False},
    ]
    assert all(
        url.endswith("/task-protection/v1/state") for url, _, _ in calls
    )
    assert all(timeout == 0.1 for _, _, timeout in calls)

    async def stalled_to_thread(*_args, **_kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(worker_module.asyncio, "to_thread", stalled_to_thread)
    timed_client = _EcsTaskProtection(
        "http://169.254.170.2/api/task-token",
        expires_in_minutes=1,
        request_timeout_seconds=0.01,
    )
    started = time.monotonic()
    with pytest.raises(_TaskProtectionError):
        asyncio.run(timed_client.enable())
    assert time.monotonic() - started < 1.0


def test_production_task_protection_is_fail_closed_without_agent_uri():
    protection = _build_task_protection("production", {})
    with pytest.raises(_TaskProtectionError):
        asyncio.run(protection.enable())
    asyncio.run(_build_task_protection("local", {}).enable())


def test_worker_persists_ordered_events_artifact_and_workspace(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        request = _request()
        await _enqueue(foundation, envelope, request, run_id="a" * 32)
        created: list[_ScriptRunner] = []

        async def script(runner: _ScriptRunner):
            workspace = Path(tempfile.mkdtemp(prefix="worker_success_"))
            runner.temp_dir = str(workspace)
            (workspace / "src").mkdir()
            (workspace / "src" / "app.py").write_text("print('ok')\n")
            yield {"event": "start", "runId": runner.run_id}
            yield {"event": "text", "delta": "building"}
            yield {"event": "cost", "usd": 0.25, "turns": 2, "elapsedSeconds": 1.0}
            artifact = workspace / "result.zip"
            artifact.write_bytes(b"generated-artifact")
            await SMART_RUN_REGISTRY.put(
                runner.run_id,
                SmartRunEntry(
                    str(artifact), "result.zip", True, str(workspace), time.time(),
                    owner_id=runner.owner_id,
                ),
            )
            yield {
                "event": "done", "runId": runner.run_id,
                "downloadUrl": "/legacy", "fileName": "result.zip",
                "isZip": True, "recipe": {}, "incomplete": False,
                "incompleteReason": None,
            }

        def factory(request_value: SmartGenerateRequest, **options: Any):
            runner = _ScriptRunner(request_value, script=script, **options)
            created.append(runner)
            return runner

        result = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=factory,
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert result.disposition == WorkerDisposition.ACKNOWLEDGED
        record = await foundation.state.get_run("a" * 32)
        assert record is not None and record.status == RunStatus.SUCCEEDED
        assert record.estimated_cost_usd == 0.25
        assert record.artifact_key
        assert record.checkpoint_key
        assert "api_key_envelope" not in record.metadata
        assert record.metadata["retry_count"] == 0
        assert record.metadata["worker_id"]
        assert record.metadata["lease"]["fencing_token"] == 1
        assert record.metadata["checkpoint"]["purpose"] == "final-workspace"
        assert record.metadata["checkpoint"]["resumable"] is False
        assert created[0].request.resolved_api_key() == request.resolved_api_key()
        assert created[0].run_id == "a" * 32
        assert created[0].options["allow_shell_tools"] is False
        assert callable(created[0].options["request_tool_approval"])

        events = await _events(foundation, "a" * 32)
        assert [event.event_type for event in events] == [
            "start", "text", "cost", "done",
        ]
        done = dict(events[-1].payload)
        assert done["downloadUrl"] == (
            f"/besser_api/smart-gen/runs/{'a' * 32}/artifact"
        )
        assert "artifact" not in done
        assert not any(key.startswith("_") for key in done)
        assert record.metadata["terminal_event_sequence"] == events[-1].sequence

        checkpoint = await foundation.checkpoints.get_checkpoint(OWNER_ID, "a" * 32)
        assert checkpoint is not None
        with zipfile.ZipFile(io.BytesIO(checkpoint)) as archive:
            assert "src/app.py" in archive.namelist()
            assert "result.zip" not in archive.namelist()

        second_slot = await foundation.state.reserve_quota(
            OWNER_ID, "concurrent-runs", "concurrent:second:test",
            amount=1, limit=1, ttl_seconds=60,
        )
        assert second_slot.allowed
        await foundation.close()

    asyncio.run(scenario())


def test_incomplete_run_keeps_resumable_workspace_bundle(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        await _enqueue(foundation, envelope, _request(), run_id="b" * 32)

        async def script(runner: _ScriptRunner):
            done = await _register_output(runner, incomplete=True, checkpoint=True)
            yield {"event": "start", "runId": runner.run_id}
            yield {"event": "phase", "phase": "customize", "message": "working"}
            yield done

        result = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=lambda request, **options: _ScriptRunner(
                request, script=script, **options,
            ),
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert result.disposition == WorkerDisposition.ACKNOWLEDGED
        record = await foundation.state.get_run("b" * 32)
        assert record is not None and record.status == RunStatus.INCOMPLETE
        assert record.artifact_key and record.checkpoint_key
        assert record.metadata["checkpoint"]["resumable"] is True
        bundle = await foundation.checkpoints.get_checkpoint(OWNER_ID, "b" * 32)
        assert bundle is not None
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            assert CHECKPOINT_FILENAME in archive.namelist()
            assert "src/app.py" in archive.namelist()
            assert not any(name.startswith("node_modules/") for name in archive.namelist())
        await foundation.close()

    asyncio.run(scenario())


async def _wait_for_run(
    foundation: DurableStateFoundation,
    run_id: str,
    predicate: Callable[[RunRecord], bool],
) -> RunRecord:
    for _ in range(300):
        record = await foundation.state.get_run(run_id)
        if record is not None and predicate(record):
            return record
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for durable run state")


async def _update_run(
    foundation: DurableStateFoundation,
    run_id: str,
    build_changes: Callable[[RunRecord], dict[str, Any]],
) -> RunRecord:
    while True:
        current = await foundation.state.get_run(run_id)
        assert current is not None
        try:
            return await foundation.state.update_run(
                run_id,
                current.version,
                build_changes(current),
            )
        except OptimisticLockError:
            await asyncio.sleep(0)


def test_worker_observes_durable_cancellation(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        run_id = "c" * 32
        await _enqueue(foundation, envelope, _request(), run_id=run_id)

        async def script(runner: _ScriptRunner):
            yield {"event": "start", "runId": runner.run_id}
            cancel_event = runner.options["reserved_cancel_event"]
            while not cancel_event.is_set():
                await asyncio.sleep(0.01)
            yield {
                "event": "error",
                "code": "CANCELLED",
                "message": "Smart generation cancelled by user request",
            }

        worker = asyncio.create_task(process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=lambda request, **options: _ScriptRunner(
                request, script=script, **options,
            ),
            wait_seconds=0,
            heartbeat_interval_seconds=0.02,
            retry_delay_seconds=0,
        ))
        await _wait_for_run(
            foundation,
            run_id,
            lambda record: record.status == RunStatus.RUNNING,
        )
        await _update_run(
            foundation,
            run_id,
            lambda _: {"status": RunStatus.CANCEL_REQUESTED},
        )
        result = await asyncio.wait_for(worker, timeout=3)
        assert result.disposition == WorkerDisposition.ACKNOWLEDGED
        record = await foundation.state.get_run(run_id)
        assert record is not None and record.status == RunStatus.CANCELLED
        assert record.error_code == "CANCELLED"
        assert "api_key_envelope" not in record.metadata
        events = await _events(foundation, run_id)
        assert [event.event_type for event in events] == ["start", "error"]
        await foundation.close()

    asyncio.run(scenario())


def test_shell_tool_approval_is_durable_and_owner_resolvable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BESSER_SMARTGEN_ALLOW_SHELL_TOOLS", "true")
    monkeypatch.setenv("BESSER_SMARTGEN_ALLOW_LOCAL_SHELL_TOOLS", "true")

    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        run_id = "d" * 32
        await _enqueue(foundation, envelope, _request(), run_id=run_id)
        created: list[_ScriptRunner] = []

        async def script(runner: _ScriptRunner):
            yield {"event": "start", "runId": runner.run_id}
            callback = runner.options["request_tool_approval"]
            approved = await asyncio.to_thread(
                callback,
                3,
                "run_command",
                {"command": "echo safe"},
            )
            assert approved is True
            yield await _register_output(runner)

        def factory(request: SmartGenerateRequest, **options: Any):
            runner = _ScriptRunner(request, script=script, **options)
            created.append(runner)
            return runner

        worker = asyncio.create_task(process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=factory,
            wait_seconds=0,
            approval_timeout_seconds=2,
            heartbeat_interval_seconds=0.05,
            retry_delay_seconds=0,
        ))
        pending = await _wait_for_run(
            foundation,
            run_id,
            lambda record: bool(record.metadata.get("approvals")),
        )
        approvals = pending.metadata["approvals"]
        approval_id = next(iter(approvals))

        def approve(current: RunRecord) -> dict[str, Any]:
            metadata = dict(current.metadata)
            updated_approvals = dict(metadata["approvals"])
            approval = dict(updated_approvals[approval_id])
            approval.update({"status": "approved", "resolved_at": time.time()})
            updated_approvals[approval_id] = approval
            metadata["approvals"] = updated_approvals
            return {"metadata": metadata}

        await _update_run(foundation, run_id, approve)
        result = await asyncio.wait_for(worker, timeout=4)
        assert result.disposition == WorkerDisposition.ACKNOWLEDGED
        assert created[0].options["allow_shell_tools"] is True
        events = await _events(foundation, run_id)
        approval_event = next(
            event for event in events if event.event_type == "approval_required"
        )
        assert approval_event.payload["approvalId"] == approval_id
        assert approval_event.payload["arguments"] == {"command": "echo safe"}
        await foundation.close()

    asyncio.run(scenario())


class _FakeClock:
    def __init__(self, value: float = 1_800_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_fence_loss_stops_runner_and_releases_message_without_stale_writes(tmp_path):
    async def scenario() -> None:
        clock = _FakeClock()
        state = InMemoryStateStore(clock=clock)
        foundation = _foundation(
            tmp_path,
            state=state,
            lease_ttl_seconds=1,
        )
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        run_id = "e" * 32
        await _enqueue(foundation, envelope, _request(), run_id=run_id)
        reached_barrier = asyncio.Event()
        continue_runner = asyncio.Event()

        async def script(runner: _ScriptRunner):
            yield {"event": "start", "runId": runner.run_id}
            reached_barrier.set()
            await continue_runner.wait()
            yield {"event": "text", "delta": "must not persist"}

        worker = asyncio.create_task(process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=lambda request, **options: _ScriptRunner(
                request, script=script, **options,
            ),
            worker_id="worker-one",
            wait_seconds=0,
            heartbeat_interval_seconds=100,
            retry_delay_seconds=0,
        ))
        await asyncio.wait_for(reached_barrier.wait(), timeout=2)
        clock.advance(2)
        replacement = await state.acquire_lease(
            run_id,
            "worker-two",
            ttl_seconds=30,
        )
        assert replacement is not None and replacement.fencing_token == 2
        continue_runner.set()
        result = await asyncio.wait_for(worker, timeout=2)
        assert result.disposition == WorkerDisposition.RETRY_RELEASED
        record = await state.get_run(run_id)
        assert record is not None and record.status == RunStatus.RUNNING
        assert record.metadata.get("worker_finalized") is not True
        events = await _events(foundation, run_id)
        assert [event.event_type for event in events] == ["start"]
        retried = await foundation.queue.receive(wait_seconds=0)
        assert retried and retried[0].receive_count == 2
        await state.release_lease(
            run_id,
            "worker-two",
            replacement.fencing_token,
        )
        await foundation.close()

    asyncio.run(scenario())


class _FailAckOnceQueue(InMemoryJobQueue):
    def __init__(self) -> None:
        super().__init__()
        self.acknowledgements = 0

    async def acknowledge(self, job) -> None:
        self.acknowledgements += 1
        if self.acknowledgements == 1:
            raise RuntimeError("simulated acknowledgement failure")
        await super().acknowledge(job)


class _EmptyOnceQueue(InMemoryJobQueue):
    def __init__(self) -> None:
        super().__init__()
        self.receive_calls = 0

    async def receive(self, **options):
        self.receive_calls += 1
        if self.receive_calls == 1:
            return ()
        return await super().receive(**options)


def test_ack_failure_redelivery_repairs_without_rerunning_generation(tmp_path):
    async def scenario() -> None:
        queue = _FailAckOnceQueue()
        foundation = _foundation(tmp_path, queue=queue)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        run_id = "f" * 32
        await _enqueue(foundation, envelope, _request(), run_id=run_id)
        runner_calls = 0

        async def script(runner: _ScriptRunner):
            yield {"event": "start", "runId": runner.run_id}
            yield await _register_output(runner)

        def factory(request: SmartGenerateRequest, **options: Any):
            nonlocal runner_calls
            runner_calls += 1
            return _ScriptRunner(request, script=script, **options)

        first = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=factory,
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert first.disposition == WorkerDisposition.RETRY_RELEASED
        second = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=factory,
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert second.disposition == WorkerDisposition.ACKNOWLEDGED
        assert runner_calls == 1
        record = await foundation.state.get_run(run_id)
        assert record is not None and record.status == RunStatus.SUCCEEDED
        events = await _events(foundation, run_id)
        assert [event.event_type for event in events] == ["start", "done"]
        assert sum(event.event_type == "done" for event in events) == 1
        assert record.metadata["terminal_event_sequence"] == events[-1].sequence
        await foundation.close()

    asyncio.run(scenario())


def test_storage_quota_denial_fails_without_publishing_artifact(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path, max_storage_bytes=8)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        run_id = "1" * 32
        await _enqueue(foundation, envelope, _request(), run_id=run_id)

        async def script(runner: _ScriptRunner):
            yield {"event": "start", "runId": runner.run_id}
            yield await _register_output(runner)

        result = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=lambda request, **options: _ScriptRunner(
                request, script=script, **options,
            ),
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert result.disposition == WorkerDisposition.ACKNOWLEDGED
        record = await foundation.state.get_run(run_id)
        assert record is not None and record.status == RunStatus.FAILED
        assert record.error_code == "STORAGE_QUOTA"
        assert record.artifact_key is None
        events = await _events(foundation, run_id)
        assert events[-1].payload == {
            "event": "error",
            "code": "QUOTA",
            "message": "Generated output exceeds the available storage quota.",
        }
        await foundation.close()

    asyncio.run(scenario())


def test_modify_hydrates_owned_base_workspace_from_durable_storage(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        base_run_id = "2" * 32
        await _enqueue(foundation, envelope, _request(), run_id=base_run_id)

        async def base_script(runner: _ScriptRunner):
            yield {"event": "start", "runId": runner.run_id}
            yield await _register_output(runner)

        base_result = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=lambda request, **options: _ScriptRunner(
                request, script=base_script, **options,
            ),
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert base_result.disposition == WorkerDisposition.ACKNOWLEDGED
        assert await SMART_RUN_REGISTRY.get(base_run_id) is None

        modify_run_id = "3" * 32
        modify_request = _request(
            mode="modify",
            base_run_id=base_run_id,
            instructions="Add an audit endpoint",
        )
        await _enqueue(
            foundation,
            envelope,
            modify_request,
            run_id=modify_run_id,
        )
        hydrated_directory: list[str] = []

        async def modify_script(runner: _ScriptRunner):
            base_entry = await SMART_RUN_REGISTRY.get(base_run_id)
            assert base_entry is not None and base_entry.owner_id == OWNER_ID
            hydrated_directory.append(base_entry.temp_dir)
            assert (Path(base_entry.temp_dir) / "src" / "app.py").read_text() == (
                "print('ok')\n"
            )
            yield {"event": "start", "runId": runner.run_id}
            yield await _register_output(runner)

        result = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=lambda request, **options: _ScriptRunner(
                request, script=modify_script, **options,
            ),
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert result.disposition == WorkerDisposition.ACKNOWLEDGED
        record = await foundation.state.get_run(modify_run_id)
        assert record is not None and record.status == RunStatus.SUCCEEDED
        assert hydrated_directory and not Path(hydrated_directory[0]).exists()
        assert await SMART_RUN_REGISTRY.get(base_run_id) is None
        await foundation.close()

    asyncio.run(scenario())


def test_interrupted_run_resumes_from_durable_workspace_bundle(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        run_id = "4" * 32
        request = _request()
        await _enqueue(
            foundation,
            envelope,
            request,
            run_id=run_id,
            status=RunStatus.RUNNING,
            metadata_overrides={"execution_started": True},
        )
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(CHECKPOINT_FILENAME, '{"turn": 4}')
            archive.writestr("src/app.py", "print('before resume')\n")
        checkpoint = await foundation.checkpoints.put_checkpoint(
            OWNER_ID,
            run_id,
            output.getvalue(),
        )

        def attach_checkpoint(current: RunRecord) -> dict[str, Any]:
            metadata = dict(current.metadata)
            metadata["checkpoint"] = {
                "storage_key": checkpoint.storage_key,
                "size_bytes": checkpoint.size_bytes,
                "sha256": checkpoint.sha256,
                "created_at": checkpoint.created_at,
                "format": "workspace-zip-v1",
                "checkpoint_file": CHECKPOINT_FILENAME,
                "purpose": "recovery",
                "resumable": True,
            }
            return {"checkpoint_key": checkpoint.storage_key, "metadata": metadata}

        await _update_run(foundation, run_id, attach_checkpoint)
        created: list[_ScriptRunner] = []

        async def resume_script(runner: _ScriptRunner):
            assert runner.temp_dir is not None
            source = Path(runner.temp_dir) / "src" / "app.py"
            assert source.read_text() == "print('before resume')\n"
            source.write_text("print('after resume')\n")
            yield {"event": "start", "runId": runner.run_id}
            artifact = Path(runner.temp_dir) / "resumed.zip"
            artifact.write_bytes(b"resumed-artifact")
            await SMART_RUN_REGISTRY.put(
                runner.run_id,
                SmartRunEntry(
                    str(artifact), "resumed.zip", True, runner.temp_dir, time.time(),
                    owner_id=runner.owner_id,
                ),
            )
            yield {
                "event": "done", "runId": runner.run_id,
                "downloadUrl": "/legacy", "fileName": "resumed.zip",
                "isZip": True, "recipe": {}, "incomplete": False,
                "incompleteReason": None,
            }

        def factory(request_value: SmartGenerateRequest, **options: Any):
            assert options["resume_run_id"] == run_id
            assert "run_id" not in options
            runner = _ScriptRunner(request_value, script=resume_script, **options)
            candidates = [
                path for path in Path(tempfile.gettempdir()).glob(
                    f"besser_llm_{run_id}_*",
                )
                if (path / CHECKPOINT_FILENAME).is_file()
            ]
            assert candidates
            runner.temp_dir = str(candidates[-1])
            created.append(runner)
            return runner

        result = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=factory,
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert result.disposition == WorkerDisposition.ACKNOWLEDGED
        record = await foundation.state.get_run(run_id)
        assert record is not None and record.status == RunStatus.SUCCEEDED
        assert created and not Path(created[0].temp_dir or "").exists()
        final_bundle = await foundation.checkpoints.get_checkpoint(OWNER_ID, run_id)
        assert final_bundle is not None
        with zipfile.ZipFile(io.BytesIO(final_bundle)) as archive:
            assert archive.read("src/app.py").decode().splitlines() == [
                "print('after resume')",
            ]
        await foundation.close()

    asyncio.run(scenario())


def test_service_worker_long_polls_until_one_job_then_exits(tmp_path):
    async def scenario() -> None:
        queue = _EmptyOnceQueue()
        foundation = _foundation(tmp_path, queue=queue)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        run_id = "5" * 32
        await _enqueue(foundation, envelope, _request(), run_id=run_id)
        runner_calls = 0

        async def script(runner: _ScriptRunner):
            yield {"event": "start", "runId": runner.run_id}
            yield await _register_output(runner)

        def factory(request: SmartGenerateRequest, **options: Any):
            nonlocal runner_calls
            runner_calls += 1
            return _ScriptRunner(request, script=script, **options)

        result = await run_worker_until_job(
            foundation,
            envelope=envelope,
            runner_factory=factory,
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert result.disposition == WorkerDisposition.ACKNOWLEDGED
        assert queue.receive_calls == 2
        assert runner_calls == 1
        assert await queue.receive(wait_seconds=0) == ()
        await foundation.close()

    asyncio.run(scenario())


async def _assert_modify_rejected_without_runner(
    foundation: DurableStateFoundation,
    envelope: LocalSecretEnvelope,
    *,
    base_run_id: str,
    run_id: str,
) -> None:
    await _enqueue(
        foundation,
        envelope,
        _request(mode="modify", base_run_id=base_run_id),
        run_id=run_id,
    )
    runner_calls = 0

    def factory(request: SmartGenerateRequest, **options: Any):
        nonlocal runner_calls
        runner_calls += 1
        raise AssertionError("runner must not start for an untrusted base")

    result = await process_one_job(
        foundation,
        envelope=envelope,
        runner_factory=factory,
        wait_seconds=0,
        retry_delay_seconds=0,
    )
    assert result.disposition == WorkerDisposition.ACKNOWLEDGED
    record = await foundation.state.get_run(run_id)
    assert record is not None and record.status == RunStatus.FAILED
    assert record.error_code == "BASE_ARTIFACT_UNAVAILABLE"
    assert runner_calls == 0
    assert await SMART_RUN_REGISTRY.get(base_run_id) is None


def test_modify_rejects_cross_owner_base_without_hydrating(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        base_run_id = "6" * 32
        await foundation.state.create_run(RunRecord(
            run_id=base_run_id,
            owner_id="github:9999",
            request_hash="a" * 64,
            status=RunStatus.SUCCEEDED,
            provider="openai",
            metadata={},
        ))
        await _assert_modify_rejected_without_runner(
            foundation,
            envelope,
            base_run_id=base_run_id,
            run_id="7" * 32,
        )
        await foundation.close()

    asyncio.run(scenario())


def test_modify_rejects_missing_base_without_hydrating(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        await _assert_modify_rejected_without_runner(
            foundation,
            envelope,
            base_run_id="8" * 32,
            run_id="9" * 32,
        )
        await foundation.close()

    asyncio.run(scenario())


def test_modify_rejects_corrupt_bundle_and_cleans_temp_workspace(tmp_path):
    async def scenario() -> None:
        foundation = _foundation(tmp_path)
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        base_run_id = "a1" * 16
        checkpoint = await foundation.checkpoints.put_checkpoint(
            OWNER_ID,
            base_run_id,
            b"not-a-zip-workspace",
        )
        await foundation.state.create_run(RunRecord(
            run_id=base_run_id,
            owner_id=OWNER_ID,
            request_hash="b" * 64,
            status=RunStatus.SUCCEEDED,
            provider="openai",
            checkpoint_key=checkpoint.storage_key,
            metadata={
                "checkpoint": {
                    "storage_key": checkpoint.storage_key,
                    "size_bytes": checkpoint.size_bytes,
                    "sha256": checkpoint.sha256,
                    "created_at": checkpoint.created_at,
                    "format": "workspace-zip-v1",
                    "purpose": "final-workspace",
                    "resumable": False,
                },
            },
        ))
        prefix = f"besser_durable_base_{base_run_id}_*"
        before = set(Path(tempfile.gettempdir()).glob(prefix))
        await _assert_modify_rejected_without_runner(
            foundation,
            envelope,
            base_run_id=base_run_id,
            run_id="b1" * 16,
        )
        after = set(Path(tempfile.gettempdir()).glob(prefix))
        assert after == before
        await foundation.close()

    asyncio.run(scenario())


def test_production_shell_tools_are_fail_closed_and_always_approval_gated(
    tmp_path,
    monkeypatch,
):
    async def run_once(directory: Path, run_id: str, expected: bool) -> None:
        foundation = _foundation(directory)
        foundation.config = replace(
            foundation.config,
            mode=DurableStateMode.PRODUCTION,
        )
        envelope = LocalSecretEnvelope()
        await foundation.initialize()
        await _enqueue(foundation, envelope, _request(), run_id=run_id)
        captured: dict[str, Any] = {}

        async def script(runner: _ScriptRunner):
            yield {"event": "start", "runId": runner.run_id}
            yield await _register_output(runner)

        def factory(request: SmartGenerateRequest, **options: Any):
            captured.update(options)
            return _ScriptRunner(request, script=script, **options)

        result = await process_one_job(
            foundation,
            envelope=envelope,
            runner_factory=factory,
            task_protection=_RecordingTaskProtection([]),
            wait_seconds=0,
            retry_delay_seconds=0,
        )
        assert result.disposition == WorkerDisposition.ACKNOWLEDGED
        assert captured["allow_shell_tools"] is expected
        assert callable(captured["request_tool_approval"])
        await foundation.close()

    async def scenario() -> None:
        monkeypatch.delenv("BESSER_SMARTGEN_ALLOW_SHELL_TOOLS", raising=False)
        await run_once(tmp_path / "disabled", "c1" * 16, False)
        monkeypatch.setenv("BESSER_SMARTGEN_ALLOW_SHELL_TOOLS", "true")
        await run_once(tmp_path / "enabled", "d1" * 16, True)

    asyncio.run(scenario())
