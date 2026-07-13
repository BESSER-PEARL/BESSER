from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import asdict

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.routers import (
    smart_generation_jobs_router as router_module,
)
from besser.utilities.web_modeling_editor.backend.services.principal import Principal
from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_jobs import (
    DurableJobRuntime,
    canonical_request_hash,
    durable_jobs_enabled,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation import (
    durable_jobs as durable_jobs_module,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_state import (
    DurableStateConfigurationError,
    InMemoryJobQueue,
    RunRecord,
    RunStatus,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_smart_generation_router import (
    _build_project_body,
)


class FakeLauncher:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def launch(self, run_id: str) -> None:
        self.calls.append(run_id)
        if self.fail:
            raise RuntimeError("worker unavailable")


class BlockingLauncher(FakeLauncher):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def launch(self, run_id: str) -> None:
        self.calls.append(run_id)
        self.started.set()
        await self.release.wait()


class FailingQueue:
    async def enqueue(self, *_args, **_kwargs) -> str:
        raise RuntimeError("queue unavailable")

    async def receive(self, **_kwargs):
        return ()


@asynccontextmanager
async def _client(monkeypatch, tmp_path, *, launcher: FakeLauncher | None = None):
    monkeypatch.setenv("BESSER_SMARTGEN_DURABLE_JOBS", "true")
    monkeypatch.setenv("BESSER_SMARTGEN_STATE_MODE", "local")
    monkeypatch.setenv(
        "BESSER_SMARTGEN_SQLITE_PATH",
        str(tmp_path / "smartgen.sqlite3"),
    )
    monkeypatch.setenv(
        "BESSER_SMARTGEN_STORAGE_DIR",
        str(tmp_path / "blobs"),
    )
    runtime = DurableJobRuntime()
    await runtime.initialize()
    selected_launcher = launcher or FakeLauncher()
    runtime._launcher = selected_launcher
    monkeypatch.setattr(router_module, "DURABLE_JOB_RUNTIME", runtime)

    app = FastAPI()
    app.include_router(router_module.router)
    transport = ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            yield client, runtime, selected_launcher, app
    finally:
        await runtime.close()


def _frames(response: httpx.Response) -> list[dict]:
    frames = []
    for raw_frame in response.text.split("\n\n"):
        if not raw_frame.strip() or raw_frame.startswith(":"):
            continue
        parsed: dict[str, object] = {}
        for line in raw_frame.splitlines():
            name, _, value = line.partition(":")
            if name == "data":
                parsed[name] = json.loads(value.strip())
            elif name in {"id", "event"}:
                parsed[name] = value.strip()
        frames.append(parsed)
    return frames


def test_enqueue_is_idempotent_and_owner_metadata_is_sanitized(monkeypatch, tmp_path):
    async def scenario() -> None:
        async with _client(monkeypatch, tmp_path) as (client, _runtime, launcher, _app):
            missing = await client.post(
                "/besser_api/smart-gen/runs",
                json=_build_project_body(),
            )
            assert missing.status_code == 400

            headers = {"Idempotency-Key": "request-0001"}
            first = await client.post(
                "/besser_api/smart-gen/runs",
                headers=headers,
                json=_build_project_body(),
            )
            assert first.status_code == 202
            assert set(first.json()) == {"run_id", "status"}
            assert first.json()["status"] == "queued"
            assert first.headers["location"].endswith(first.json()["run_id"])

            replay = await client.post(
                "/besser_api/smart-gen/runs",
                headers=headers,
                json=_build_project_body(),
            )
            assert replay.status_code == 202
            assert replay.json() == first.json()
            assert launcher.calls == [first.json()["run_id"]]

            conflict = await client.post(
                "/besser_api/smart-gen/runs",
                headers=headers,
                json=_build_project_body(instructions="Build something else"),
            )
            assert conflict.status_code == 409

            listed = await client.get("/besser_api/smart-gen/runs")
            detail = await client.get(first.headers["location"])
            assert listed.status_code == detail.status_code == 200
            combined = listed.text + detail.text
            assert "api_key" not in combined
            assert "api_key_envelope" not in combined
            assert "request" not in detail.json()
            assert detail.json()["events_url"].endswith("/events")

    asyncio.run(scenario())


def test_durable_feature_configuration_fails_closed():
    assert durable_jobs_enabled({"BESSER_SMARTGEN_STATE_MODE": "production"}) is True
    assert durable_jobs_enabled({"BESSER_SMARTGEN_DURABLE_JOBS": "false"}) is False
    with pytest.raises(DurableStateConfigurationError, match="must be a boolean"):
        durable_jobs_enabled({"BESSER_SMARTGEN_DURABLE_JOBS": "tru"})


def test_orphaned_idempotency_claim_is_recovered(monkeypatch, tmp_path):
    async def scenario() -> None:
        async with _client(monkeypatch, tmp_path) as (client, runtime, launcher, _app):
            body = _build_project_body()
            request = SmartGenerateRequest.model_validate(body)
            orphan_id = "f" * 32
            await runtime.foundation.state.claim_idempotency(
                "local:anonymous",
                "request-orphan",
                canonical_request_hash(request, "local:anonymous"),
                orphan_id,
                ttl_seconds=3600,
            )
            response = await client.post(
                "/besser_api/smart-gen/runs",
                headers={"Idempotency-Key": "request-orphan"},
                json=body,
            )
            assert response.status_code == 202
            assert response.json() == {"run_id": orphan_id, "status": "queued"}
            assert launcher.calls == [orphan_id]

    asyncio.run(scenario())


def test_pending_database_run_repairs_missing_queue_publication(monkeypatch, tmp_path):
    async def scenario() -> None:
        async with _client(monkeypatch, tmp_path) as (client, runtime, launcher, _app):
            body = _build_project_body()
            request = SmartGenerateRequest.model_validate(body)
            run_id = "e" * 32
            request_hash = canonical_request_hash(request, "local:anonymous")
            await runtime.foundation.state.claim_idempotency(
                "local:anonymous",
                "request-pending",
                request_hash,
                run_id,
                ttl_seconds=3600,
            )
            await runtime.foundation.state.create_run(RunRecord(
                run_id=run_id,
                owner_id="local:anonymous",
                request_hash=request_hash,
                provider=request.provider,
                model=request.llm_model,
                max_cost_usd=request.max_cost_usd,
                max_runtime_seconds=request.max_runtime_seconds,
                metadata={"dispatch_state": "pending"},
            ))

            response = await client.post(
                "/besser_api/smart-gen/runs",
                headers={"Idempotency-Key": "request-pending"},
                json=body,
            )
            assert response.status_code == 202
            assert response.json()["run_id"] == run_id
            queued = await runtime.foundation.queue.receive(wait_seconds=0)
            assert len(queued) == 1 and queued[0].message.run_id == run_id
            repaired = await runtime.foundation.state.get_run(run_id)
            assert repaired is not None
            assert repaired.metadata["dispatch_state"] == "published"
            events = await runtime.foundation.state.read_events(run_id)
            assert events.events[0].payload["message"] == "Run queued"
            assert launcher.calls == [run_id]

    asyncio.run(scenario())


def test_durable_modify_is_queued_for_worker_workspace_hydration(
    monkeypatch,
    tmp_path,
):
    async def scenario() -> None:
        async with _client(monkeypatch, tmp_path) as (client, runtime, launcher, _app):
            response = await client.post(
                "/besser_api/smart-gen/runs",
                headers={"Idempotency-Key": "request-modify"},
                json=_build_project_body(
                    mode="modify",
                    base_run_id="a" * 32,
                ),
            )
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            record = await runtime.foundation.state.get_run(run_id)
            assert record is not None
            assert record.mode == "modify"
            assert record.metadata["request"]["base_run_id"] == "a" * 32
            assert launcher.calls == [run_id]

    asyncio.run(scenario())


def test_sse_replays_after_last_event_id_and_hides_foreign_runs(monkeypatch, tmp_path):
    async def scenario() -> None:
        async with _client(monkeypatch, tmp_path) as (client, runtime, _launcher, app):
            created = await client.post(
                "/besser_api/smart-gen/runs",
                headers={"Idempotency-Key": "request-events"},
                json=_build_project_body(),
            )
            run_id = created.json()["run_id"]
            await runtime.foundation.state.append_event(
                run_id,
                "text",
                {"event": "text", "delta": "hello"},
            )
            await runtime.foundation.state.append_event(
                run_id,
                "done",
                {
                    "event": "done",
                    "runId": run_id,
                    "downloadUrl": f"/besser_api/smart-gen/runs/{run_id}/artifact",
                    "fileName": "app.zip",
                    "isZip": True,
                    "recipe": {},
                },
            )

            replay = await client.get(
                f"/besser_api/smart-gen/runs/{run_id}/events?follow=false",
                headers={"Last-Event-ID": "1"},
            )
            assert replay.status_code == 200
            frames = _frames(replay)
            assert [frame["id"] for frame in frames] == ["2", "3"]
            assert [frame["event"] for frame in frames] == ["text", "done"]
            assert frames[0]["data"] == {"event": "text", "delta": "hello"}

            invalid = await client.get(
                f"/besser_api/smart-gen/runs/{run_id}/events?follow=false",
                headers={"Last-Event-ID": "not-a-sequence"},
            )
            assert invalid.status_code == 400

            app.dependency_overrides[router_module.get_current_principal] = lambda: Principal(
                subject="github:9999",
                provider="github",
            )
            hidden_detail = await client.get(f"/besser_api/smart-gen/runs/{run_id}")
            hidden_events = await client.get(
                f"/besser_api/smart-gen/runs/{run_id}/events?follow=false"
            )
            assert hidden_detail.status_code == hidden_events.status_code == 404

    asyncio.run(scenario())


def test_cancel_and_approval_resolution_are_optimistic_and_idempotent(
    monkeypatch,
    tmp_path,
):
    async def scenario() -> None:
        async with _client(monkeypatch, tmp_path) as (client, runtime, _launcher, _app):
            cancelled_run = await client.post(
                "/besser_api/smart-gen/runs",
                headers={"Idempotency-Key": "request-cancel"},
                json=_build_project_body(),
            )
            cancel_url = f"{cancelled_run.headers['location']}/cancel"
            cancelled = await client.post(cancel_url)
            cancelled_again = await client.post(cancel_url)
            assert cancelled.status_code == cancelled_again.status_code == 202
            assert cancelled.json()["status"] == "cancel_requested"

            approval_run = await client.post(
                "/besser_api/smart-gen/runs",
                headers={"Idempotency-Key": "request-approval"},
                json=_build_project_body(),
            )
            run_id = approval_run.json()["run_id"]
            record = await runtime.foundation.state.get_run(run_id)
            assert record is not None
            metadata = dict(record.metadata)
            metadata["approvals"] = {
                "shell-001": {
                    "status": "pending",
                    "title": "Run project tests",
                    "arguments": {"command": "pytest"},
                },
                "shell-expired": {
                    "status": "timed_out",
                    "arguments": {"command": "do-not-expose"},
                },
            }
            await runtime.foundation.state.update_run(
                run_id,
                record.version,
                {"metadata": metadata},
            )
            approval_url = (
                f"/besser_api/smart-gen/runs/{run_id}/approvals/shell-001"
            )
            approved = await client.post(approval_url, json={"decision": "approved"})
            approved_again = await client.post(
                approval_url,
                json={"decision": "approved"},
            )
            rejected_after = await client.post(
                approval_url,
                json={"decision": "rejected"},
            )
            assert approved.status_code == approved_again.status_code == 200
            assert approved.json() == {
                "run_id": run_id,
                "approval_id": "shell-001",
                "status": "approved",
            }
            assert rejected_after.status_code == 409

            stored = await runtime.foundation.state.get_run(run_id)
            assert stored is not None
            approval = stored.metadata["approvals"]["shell-001"]
            assert approval["status"] == "approved"
            assert approval["resolved_by"] == "local:anonymous"
            events = await runtime.foundation.state.read_events(run_id, limit=100)
            resolution_events = [
                event
                for event in events.events
                if event.event_type == "approval_resolved"
            ]
            assert len(resolution_events) == 1
            assert resolution_events[0].payload["decision"] == "approved"
            detail = await client.get(f"/besser_api/smart-gen/runs/{run_id}")
            assert detail.json()["approvals"] == {
                "shell-001": "approved",
                "shell-expired": "timed_out",
            }
            assert "arguments" not in detail.text
            assert "do-not-expose" not in detail.text

    asyncio.run(scenario())


def test_artifact_download_uses_full_durable_reference(monkeypatch, tmp_path):
    async def scenario() -> None:
        async with _client(monkeypatch, tmp_path) as (client, runtime, _launcher, _app):
            created = await client.post(
                "/besser_api/smart-gen/runs",
                headers={"Idempotency-Key": "request-artifact"},
                json=_build_project_body(),
            )
            run_id = created.json()["run_id"]
            source = tmp_path / "app.zip"
            source.write_bytes(b"durable-generated-app")
            artifact = await runtime.foundation.artifacts.put_artifact(
                "local:anonymous",
                run_id,
                str(source),
                file_name="app.zip",
                content_type="application/zip",
            )
            record = await runtime.foundation.state.get_run(run_id)
            assert record is not None
            running = await runtime.foundation.state.update_run(
                run_id,
                record.version,
                {"status": RunStatus.RUNNING},
            )
            metadata = dict(running.metadata)
            metadata["artifact"] = {**asdict(artifact), "is_zip": True}
            await runtime.foundation.state.update_run(
                run_id,
                running.version,
                {
                    "status": RunStatus.SUCCEEDED,
                    "artifact_key": artifact.storage_key,
                    "metadata": metadata,
                },
            )

            detail = await client.get(f"/besser_api/smart-gen/runs/{run_id}")
            downloaded = await client.get(
                f"/besser_api/smart-gen/runs/{run_id}/artifact"
            )
            assert detail.json()["artifact_available"] is True
            assert detail.json()["artifact_url"].endswith("/artifact")
            assert downloaded.status_code == 200
            assert downloaded.content == b"durable-generated-app"
            assert "app.zip" in downloaded.headers["content-disposition"]

    asyncio.run(scenario())


def test_managed_worker_mode_accepts_enqueue_without_per_request_ecs_dispatch(
    monkeypatch,
    tmp_path,
):
    async def scenario() -> None:
        async with _client(monkeypatch, tmp_path) as (client, runtime, _launcher, _app):
            runtime._launcher = durable_jobs_module._ManagedWorkerLauncher()
            response = await client.post(
                "/besser_api/smart-gen/runs",
                headers={"Idempotency-Key": "request-managed-worker"},
                json=_build_project_body(),
            )
            assert response.status_code == 202
            queued = await runtime.foundation.queue.receive(wait_seconds=0)
            assert len(queued) == 1
            assert queued[0].message.run_id == response.json()["run_id"]

    asyncio.run(scenario())


def test_queue_failure_is_durable_and_releases_admission(monkeypatch, tmp_path):
    async def scenario() -> None:
        monkeypatch.setenv("BESSER_SMARTGEN_MAX_CONCURRENT_RUNS_PER_OWNER", "1")
        async with _client(monkeypatch, tmp_path) as (client, runtime, launcher, _app):
            runtime.foundation.queue = FailingQueue()
            failed = await client.post(
                "/besser_api/smart-gen/runs",
                headers={"Idempotency-Key": "request-failure"},
                json=_build_project_body(),
            )
            assert failed.status_code == 503
            records = await runtime.foundation.state.list_runs("local:anonymous")
            assert len(records) == 1
            assert records[0].status == RunStatus.FAILED
            assert records[0].error_code == "DISPATCH_FAILED"
            assert await runtime.foundation.queue.receive(wait_seconds=0) == ()
            assert launcher.calls == []

            replacement_queue = InMemoryJobQueue()
            await replacement_queue.initialize()
            runtime.foundation.queue = replacement_queue
            replacement = FakeLauncher()
            runtime._launcher = replacement
            admitted = await client.post(
                "/besser_api/smart-gen/runs",
                headers={"Idempotency-Key": "request-after-failure"},
                json=_build_project_body(),
            )
            assert admitted.status_code == 202
            assert replacement.calls == [admitted.json()["run_id"]]

    asyncio.run(scenario())


def test_client_cancellation_does_not_cancel_durable_admission(monkeypatch, tmp_path):
    async def scenario() -> None:
        launcher = BlockingLauncher()
        async with _client(
            monkeypatch,
            tmp_path,
            launcher=launcher,
        ) as (_client_instance, runtime, _launcher, _app):
            operation = asyncio.create_task(
                runtime.enqueue(
                    SmartGenerateRequest.model_validate(_build_project_body()),
                    owner_id="local:anonymous",
                    idempotency_key="request-disconnect",
                )
            )
            await launcher.started.wait()
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
            launcher.release.set()
            for _ in range(100):
                if not runtime._admission_tasks:
                    break
                await asyncio.sleep(0.01)
            records = await runtime.foundation.state.list_runs("local:anonymous")
            assert len(records) == 1
            assert records[0].status == RunStatus.QUEUED
            assert launcher.calls == [records[0].run_id]

    asyncio.run(scenario())
