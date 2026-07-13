import asyncio

import pytest

from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_state import (
    IdempotencyConflictError,
    InMemoryStateStore,
    InvalidRunTransitionError,
    LeaseLostError,
    OptimisticLockError,
    RecordAlreadyExistsError,
    ReplayCursor,
    RunRecord,
    RunStatus,
    SQLiteStateStore,
)


class FakeClock:
    def __init__(self, value: float = 1_800_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _run_record(run_id: str = "a" * 32) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        owner_id="github:1234",
        request_hash="b" * 64,
        provider="openai",
        model="gpt-4o-mini",
        max_cost_usd=1.0,
        max_runtime_seconds=600,
        created_at=1_800_000_000.0,
        updated_at=1_800_000_000.0,
        metadata={"project_id": "project-1"},
    )


@pytest.mark.parametrize("adapter_kind", ["memory", "sqlite"])
def test_state_store_contract(adapter_kind, tmp_path):
    clock = FakeClock()
    if adapter_kind == "memory":
        store = InMemoryStateStore(clock=clock)
    else:
        store = SQLiteStateStore(str(tmp_path / "state.sqlite3"), clock=clock)

    async def scenario():
        await store.initialize()
        created = await store.create_run(_run_record())
        assert created.version == 1
        assert await store.get_owned_run(created.owner_id, created.run_id) == created
        assert await store.get_owned_run("github:9999", created.run_id) is None
        with pytest.raises(RecordAlreadyExistsError):
            await store.create_run(_run_record())

        running = await store.update_run(
            created.run_id,
            created.version,
            {"status": RunStatus.RUNNING, "estimated_cost_usd": 0.1},
        )
        assert running.version == 2
        assert running.started_at == clock.value
        with pytest.raises(OptimisticLockError):
            await store.update_run(created.run_id, 1, {"estimated_cost_usd": 0.2})
        with pytest.raises(InvalidRunTransitionError):
            await store.update_run(running.run_id, running.version, {"status": RunStatus.QUEUED})

        emitted = await asyncio.gather(*(
            store.append_event(created.run_id, "text", {"index": index})
            for index in range(12)
        ))
        assert sorted(event.sequence for event in emitted) == list(range(1, 13))

        first_page = await store.read_events(created.run_id, limit=5)
        assert [event.sequence for event in first_page.events] == [1, 2, 3, 4, 5]
        assert first_page.has_more is True
        restored_cursor = ReplayCursor.decode(first_page.cursor.encode())
        second_page = await store.read_events(created.run_id, cursor=restored_cursor, limit=20)
        assert [event.sequence for event in second_page.events] == list(range(6, 13))
        assert second_page.has_more is False

        first_claim = await store.claim_idempotency(
            created.owner_id,
            "request-123",
            created.request_hash,
            created.run_id,
            ttl_seconds=60,
        )
        replayed_claim = await store.claim_idempotency(
            created.owner_id,
            "request-123",
            created.request_hash,
            "c" * 32,
            ttl_seconds=60,
        )
        assert first_claim.created is True
        assert replayed_claim.created is False
        assert replayed_claim.record.run_id == created.run_id
        with pytest.raises(IdempotencyConflictError):
            await store.claim_idempotency(
                created.owner_id,
                "request-123",
                "d" * 64,
                created.run_id,
                ttl_seconds=60,
            )

        lease = await store.acquire_lease(created.run_id, "worker-1", ttl_seconds=30)
        assert lease is not None and lease.fencing_token == 1
        assert await store.acquire_lease(created.run_id, "worker-2", ttl_seconds=30) is None
        fenced = await store.update_run_fenced(
            running.run_id,
            running.version,
            "worker-1",
            lease.fencing_token,
            {"estimated_cost_usd": 0.2},
        )
        assert fenced.estimated_cost_usd == 0.2
        fenced_event = await store.append_event_fenced(
            running.run_id,
            "worker-1",
            lease.fencing_token,
            "cost",
            {"event": "cost", "usd": 0.2},
        )
        assert fenced_event.sequence == 13
        with pytest.raises(LeaseLostError):
            await store.append_event_fenced(
                running.run_id,
                "worker-2",
                lease.fencing_token,
                "text",
                {"event": "text", "delta": "stale"},
            )
        clock.advance(31)
        replacement = await store.acquire_lease(created.run_id, "worker-2", ttl_seconds=30)
        assert replacement is not None and replacement.fencing_token == 2
        assert await store.renew_lease(
            created.run_id,
            "worker-1",
            lease.fencing_token,
            ttl_seconds=30,
        ) is None
        with pytest.raises(LeaseLostError):
            await store.update_run_fenced(
                fenced.run_id,
                fenced.version,
                "worker-1",
                lease.fencing_token,
                {"estimated_cost_usd": 0.3},
            )

        decisions = await asyncio.gather(*(
            store.reserve_quota(
                created.owner_id,
                "concurrent-runs",
                f"quota-{index}",
                amount=1,
                limit=2,
                ttl_seconds=60,
            )
            for index in range(2)
        ))
        assert all(decision.allowed for decision in decisions)
        denied = await store.reserve_quota(
            created.owner_id,
            "concurrent-runs",
            "quota-3",
            amount=1,
            limit=2,
            ttl_seconds=60,
        )
        assert denied.allowed is False
        assert denied.retry_after_seconds == 60
        assert await store.release_quota("quota-0") is True
        admitted = await store.reserve_quota(
            created.owner_id,
            "concurrent-runs",
            "quota-4",
            amount=1,
            limit=2,
            ttl_seconds=60,
        )
        assert admitted.allowed is True

        listed = await store.list_runs(created.owner_id)
        assert [record.run_id for record in listed] == [created.run_id]
        await store.close()

    asyncio.run(scenario())


def test_sqlite_records_survive_adapter_recreation(tmp_path):
    path = str(tmp_path / "state.sqlite3")

    async def scenario():
        first = SQLiteStateStore(path)
        await first.initialize()
        await first.create_run(_run_record())
        await first.append_event("a" * 32, "phase", {"name": "queued"})
        await first.close()

        second = SQLiteStateStore(path)
        await second.initialize()
        record = await second.get_run("a" * 32)
        events = await second.read_events("a" * 32)
        assert record is not None and record.owner_id == "github:1234"
        assert events.events[0].payload == {"name": "queued"}

    asyncio.run(scenario())
