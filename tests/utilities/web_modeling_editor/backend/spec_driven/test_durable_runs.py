"""Durable spec-driven run ownership and event replay tests."""

from __future__ import annotations

import asyncio
import json
import os
import time

from besser.utilities.web_modeling_editor.backend.services.spec_driven.run_manager import (
    DurableRunManager,
    SqliteRunEventStore,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartRunEntry,
    SmartRunRegistry,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import (
    DoneEvent,
    ErrorEvent,
    PhaseEvent,
    StartEvent,
    format_sse,
)


def _payload(frame: bytes) -> dict:
    data = next(
        line[5:].strip()
        for line in frame.decode("utf-8").splitlines()
        if line.startswith("data:")
    )
    return json.loads(data)


def _start(run_id: str) -> bytes:
    return format_sse(
        StartEvent(
            runId=run_id,
            provider="openai",
            llmModel="test-model",
            maxCost=1.0,
            maxRuntime=60,
        )
    )


def _done(run_id: str, *, incomplete: bool = False) -> bytes:
    return format_sse(
        DoneEvent(
            runId=run_id,
            downloadUrl=f"/download/{run_id}",
            fileName="result.zip",
            isZip=True,
            incomplete=incomplete,
        )
    )


def test_store_persists_sequenced_events_across_instances(tmp_path):
    path = str(tmp_path / "runs.sqlite3")
    run_id = "a" * 32

    first = SqliteRunEventStore(path)
    first.begin_run(run_id)
    first.set_status(run_id, "running")
    event = first.append(run_id, _start(run_id))
    first.set_status(run_id, "succeeded", terminal_event="done")
    first.close()

    second = SqliteRunEventStore(path)
    record = second.get_run(run_id)
    replay = second.events_after(run_id, 0)
    second.close()

    assert event.sequence == 1
    assert b"id: 1\n" in event.frame
    assert _payload(event.frame)["sequence"] == 1
    assert record is not None
    assert record.status == "succeeded"
    assert [item.sequence for item in replay] == [1]


def test_store_marks_orphaned_running_record_interrupted(tmp_path):
    path = str(tmp_path / "runs.sqlite3")
    run_id = "b" * 32

    first = SqliteRunEventStore(path)
    first.begin_run(run_id)
    first.set_status(run_id, "running")
    first.close()

    restarted = SqliteRunEventStore(path)
    record = restarted.get_run(run_id)
    restarted.close()

    assert record is not None
    assert record.status == "interrupted"
    assert record.terminal_event == "interrupted"


def test_subscriber_disconnect_does_not_stop_producer_and_replay_is_exact():
    async def exercise() -> None:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "c" * 32
        continue_producer = asyncio.Event()

        async def source():
            yield _start(run_id)
            await continue_producer.wait()
            yield format_sse(PhaseEvent(phase="generate", message="building"))
            yield _done(run_id)

        await manager.start(run_id, source())

        first_subscription = manager.subscribe(run_id)
        first_frame = await anext(first_subscription)
        await first_subscription.aclose()

        # Closing the only subscriber must not cancel the producer task.
        assert _payload(first_frame)["sequence"] == 1
        assert manager.get_run(run_id).status == "running"

        continue_producer.set()
        await manager.wait(run_id)

        replayed = []
        async for frame in manager.subscribe(run_id, after_sequence=1):
            replayed.append(_payload(frame))

        record = manager.get_run(run_id)
        assert record is not None
        assert record.status == "succeeded"
        assert [event["sequence"] for event in replayed] == [2, 3]
        assert [event["event"] for event in replayed] == ["phase", "done"]
        store.close()

    asyncio.run(exercise())


def test_incomplete_done_maps_to_partial_status():
    async def exercise() -> None:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "d" * 32

        async def source():
            yield _start(run_id)
            yield _done(run_id, incomplete=True)

        await manager.start(run_id, source())
        await manager.wait(run_id)
        assert manager.get_run(run_id).status == "partial"
        store.close()

    asyncio.run(exercise())


def test_run_without_subscribers_is_stopped_after_grace_period():
    async def exercise() -> None:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "e" * 32
        cancel_signal = asyncio.Event()

        async def source():
            yield _start(run_id)
            await cancel_signal.wait()
            yield format_sse(
                ErrorEvent(code="CANCELLED", message="cancelled")
            )

        await manager.start(
            run_id,
            source(),
            on_abandoned=cancel_signal.set,
            disconnect_grace_seconds=0.01,
        )
        await manager.wait(run_id)

        record = manager.get_run(run_id)
        replay = store.events_after(run_id, 1)
        assert record is not None
        assert record.status == "abandoned"
        assert record.terminal_event == "ABANDONED"
        assert record.subscriber_count == 0
        assert record.disconnected_at is not None
        assert record.abandonment_requested_at is not None
        assert len(replay) == 1
        assert replay[0].payload["code"] == "CANCELLED"
        assert replay[0].payload["reason"] == "abandoned"
        assert replay[0].payload["resumeAvailable"] is False
        assert record.resume_available is False
        assert "No resumable checkpoint" in replay[0].payload["message"]
        store.close()

    asyncio.run(exercise())


def test_abandonment_only_advertises_a_verified_checkpoint():
    async def exercise() -> None:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "1" * 32
        cancel_signal = asyncio.Event()

        async def source():
            yield _start(run_id)
            await cancel_signal.wait()
            yield format_sse(ErrorEvent(code="CANCELLED", message="cancelled"))

        await manager.start(
            run_id,
            source(),
            on_abandoned=cancel_signal.set,
            resume_available=lambda: True,
            disconnect_grace_seconds=0.01,
        )
        await manager.wait(run_id)

        record = manager.get_run(run_id)
        terminal = store.events_after(run_id, 1)[0].payload
        assert record is not None
        assert record.resume_available is True
        assert terminal["resumeAvailable"] is True
        assert "verified checkpoint was retained" in terminal["message"]
        store.close()

    asyncio.run(exercise())


def test_abandoned_real_runner_shape_keeps_partial_download():
    """Cooperative runner cancellation emits INCOMPLETE then done, not CANCELLED."""

    async def exercise() -> None:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "7" * 32
        cancel_signal = asyncio.Event()

        async def source():
            yield _start(run_id)
            await cancel_signal.wait()
            yield format_sse(
                ErrorEvent(code="INCOMPLETE", message="run was cancelled")
            )
            yield _done(run_id, incomplete=True)

        await manager.start(
            run_id,
            source(),
            on_abandoned=cancel_signal.set,
            resume_available=lambda: False,
            disconnect_grace_seconds=0.01,
        )
        await manager.wait(run_id)

        record = manager.get_run(run_id)
        replay = store.events_after(run_id, 0)
        assert record is not None
        assert record.status == "abandoned"
        assert record.terminal_event == "done"
        assert record.resume_available is False
        assert [event.event_type for event in replay] == [
            "start",
            "error",
            "done",
        ]
        assert replay[-1].payload["downloadUrl"] == f"/download/{run_id}"
        store.close()

    asyncio.run(exercise())


def test_resume_starts_a_fresh_event_sequence():
    async def exercise() -> None:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "2" * 32

        async def first_source():
            yield _start(run_id)
            yield format_sse(ErrorEvent(code="CANCELLED", message="stopped"))

        async def resumed_source():
            yield _start(run_id)
            yield _done(run_id)

        await manager.start(run_id, first_source())
        await manager.wait(run_id)
        assert manager.get_run(run_id).last_sequence == 2

        await manager.start(run_id, resumed_source(), resume=True)
        await manager.wait(run_id)
        replay = store.events_after(run_id, 0)

        assert [event.sequence for event in replay] == [1, 2]
        assert [event.event_type for event in replay] == ["start", "done"]
        assert manager.get_run(run_id).status == "succeeded"
        store.close()

    asyncio.run(exercise())


def test_old_subscriber_finalizer_cannot_touch_resumed_attempt():
    async def exercise() -> None:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "3" * 32
        finish = asyncio.Event()
        cancel_signal = asyncio.Event()

        async def first_source():
            yield _start(run_id)
            yield _done(run_id)

        async def resumed_source():
            yield _start(run_id)
            await finish.wait()
            yield _done(run_id)

        await manager.start(run_id, first_source())
        old_subscription = manager.subscribe(run_id)
        await anext(old_subscription)
        await manager.wait(run_id)

        await manager.start(
            run_id,
            resumed_source(),
            resume=True,
            on_abandoned=cancel_signal.set,
            disconnect_grace_seconds=0.02,
        )
        new_subscription = manager.subscribe(run_id)
        await anext(new_subscription)
        await old_subscription.aclose()
        await asyncio.sleep(0.04)

        record = manager.get_run(run_id)
        assert record is not None
        assert record.subscriber_count == 1
        assert cancel_signal.is_set() is False

        finish.set()
        await anext(new_subscription)
        await new_subscription.aclose()
        await manager.wait(run_id)
        store.close()

    asyncio.run(exercise())


def test_internal_producer_exception_is_redacted_from_status():
    async def exercise() -> None:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "4" * 32

        async def source():
            yield _start(run_id)
            raise RuntimeError("secret sentinel")

        await manager.start(run_id, source())
        await manager.wait(run_id)
        record = manager.get_run(run_id)

        assert record is not None
        assert record.status == "failed"
        assert record.error == "Internal server error"
        assert "secret sentinel" not in json.dumps(record.to_api_dict())
        store.close()

    asyncio.run(exercise())


def test_download_registry_restores_artifact_after_restart(tmp_path, monkeypatch):
    async def exercise() -> None:
        from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
            runner,
        )

        monkeypatch.setattr(runner, "LLM_RUN_WORKSPACE_ROOT", str(tmp_path))
        run_id = "5" * 32
        workspace = tmp_path / f"besser_llm_{run_id}_test"
        workspace.mkdir()
        artifact = workspace / "result.txt"
        artifact.write_text("result", encoding="utf-8")
        created_at = time.time()

        first = SmartRunRegistry()
        await first.put(
            run_id,
            SmartRunEntry(
                file_path=str(artifact),
                file_name="result.txt",
                is_zip=False,
                temp_dir=str(workspace),
                created_at=created_at,
            ),
        )

        restarted = SmartRunRegistry()
        assert await restarted.restore_persisted() == 1
        restored = await restarted.get(run_id)
        assert restored is not None
        assert os.path.samefile(restored.file_path, artifact)
        assert restored.file_name == "result.txt"
        assert restored.created_at == created_at

    asyncio.run(exercise())


def test_cleanup_sweeps_only_stale_smart_runs_from_persistent_root(
    tmp_path, monkeypatch
):
    from besser.utilities.web_modeling_editor.backend.services import cleanup

    system_root = tmp_path / "system"
    persistent_root = tmp_path / "persistent"
    system_root.mkdir()
    persistent_root.mkdir()
    stale_run = persistent_root / f"besser_llm_{'6' * 32}_old"
    unrelated = persistent_root / "keep_me"
    stale_run.mkdir()
    unrelated.mkdir()
    old = time.time() - 7200
    os.utime(stale_run, (old, old))
    os.utime(unrelated, (old, old))

    monkeypatch.setattr(cleanup.tempfile, "gettempdir", lambda: str(system_root))
    monkeypatch.setattr(cleanup, "LLM_RUN_WORKSPACE_ROOT", str(persistent_root))
    cleanup.cleanup_old_temp_files(max_age_hours=1)

    assert stale_run.exists() is False
    assert unrelated.is_dir()


def test_reconnect_within_grace_cancels_abandonment_timer():
    async def exercise() -> None:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "f" * 32
        cancel_signal = asyncio.Event()
        finish = asyncio.Event()

        async def source():
            yield _start(run_id)
            await finish.wait()
            yield _done(run_id)

        await manager.start(
            run_id,
            source(),
            on_abandoned=cancel_signal.set,
            disconnect_grace_seconds=0.1,
        )

        first = manager.subscribe(run_id)
        await anext(first)
        connected = manager.get_run(run_id)
        assert connected is not None
        assert connected.subscriber_count == 1
        await first.aclose()

        disconnected = manager.get_run(run_id)
        assert disconnected is not None
        assert disconnected.subscriber_count == 0
        assert disconnected.disconnected_at is not None

        await asyncio.sleep(0.02)
        second = manager.subscribe(run_id, after_sequence=1)
        next_frame = asyncio.create_task(anext(second))
        await asyncio.sleep(0.12)
        assert not cancel_signal.is_set()

        finish.set()
        assert _payload(await next_frame)["event"] == "done"
        await second.aclose()
        await manager.wait(run_id)
        assert manager.get_run(run_id).status == "succeeded"
        store.close()

    asyncio.run(exercise())


def test_one_of_two_subscribers_leaving_does_not_start_abandonment():
    async def exercise() -> None:
        store = SqliteRunEventStore(":memory:")
        manager = DurableRunManager(store)
        run_id = "8" * 32
        cancel_signal = asyncio.Event()
        finish = asyncio.Event()

        async def source():
            yield _start(run_id)
            await finish.wait()
            yield _done(run_id)

        await manager.start(
            run_id,
            source(),
            on_abandoned=cancel_signal.set,
            disconnect_grace_seconds=0.02,
        )
        first = manager.subscribe(run_id)
        second = manager.subscribe(run_id)
        await anext(first)
        await anext(second)
        await first.aclose()
        await asyncio.sleep(0.04)

        record = manager.get_run(run_id)
        assert record is not None
        assert record.subscriber_count == 1
        assert cancel_signal.is_set() is False

        finish.set()
        await anext(second)
        await second.aclose()
        await manager.wait(run_id)
        store.close()

    asyncio.run(exercise())
