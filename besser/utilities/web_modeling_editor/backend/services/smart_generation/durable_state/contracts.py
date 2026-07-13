"""Async interfaces for SmartGen durable state, blobs, and job dispatch."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol

from .models import (
    ArtifactRef,
    CheckpointRef,
    EventPage,
    EventRecord,
    IdempotencyClaim,
    JobMessage,
    LeaseRecord,
    QueuedJob,
    QuotaDecision,
    ReplayCursor,
    RunRecord,
)


class DurableStateStore(Protocol):
    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def create_run(self, record: RunRecord) -> RunRecord: ...

    async def get_run(self, run_id: str) -> Optional[RunRecord]: ...

    async def get_owned_run(self, owner_id: str, run_id: str) -> Optional[RunRecord]: ...

    async def update_run(
        self,
        run_id: str,
        expected_version: int,
        changes: Mapping[str, Any],
    ) -> RunRecord: ...

    async def update_run_fenced(
        self,
        run_id: str,
        expected_version: int,
        worker_id: str,
        fencing_token: int,
        changes: Mapping[str, Any],
    ) -> RunRecord: ...

    async def list_runs(self, owner_id: str, *, limit: int = 50) -> tuple[RunRecord, ...]: ...

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> EventRecord: ...

    async def append_event_fenced(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> EventRecord: ...

    async def read_events(
        self,
        run_id: str,
        *,
        cursor: Optional[ReplayCursor] = None,
        limit: int = 200,
    ) -> EventPage: ...

    async def claim_idempotency(
        self,
        owner_id: str,
        key: str,
        request_hash: str,
        run_id: str,
        *,
        ttl_seconds: int,
    ) -> IdempotencyClaim: ...

    async def acquire_lease(
        self,
        run_id: str,
        worker_id: str,
        *,
        ttl_seconds: int,
    ) -> Optional[LeaseRecord]: ...

    async def renew_lease(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
        *,
        ttl_seconds: int,
    ) -> Optional[LeaseRecord]: ...

    async def release_lease(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
    ) -> bool: ...

    async def reserve_quota(
        self,
        owner_id: str,
        resource: str,
        reservation_id: str,
        *,
        amount: int,
        limit: int,
        ttl_seconds: int,
    ) -> QuotaDecision: ...

    async def release_quota(self, reservation_id: str) -> bool: ...


class ArtifactStorage(Protocol):
    async def initialize(self) -> None: ...

    async def put_artifact(
        self,
        owner_id: str,
        run_id: str,
        source_path: str,
        *,
        file_name: str,
        content_type: str,
    ) -> ArtifactRef: ...

    async def download_artifact(self, artifact: ArtifactRef, destination_path: str) -> None: ...

    async def delete_artifact(self, artifact: ArtifactRef) -> None: ...

    async def create_download_url(
        self,
        artifact: ArtifactRef,
        *,
        expires_seconds: int,
    ) -> Optional[str]: ...


class CheckpointStorage(Protocol):
    async def put_checkpoint(
        self,
        owner_id: str,
        run_id: str,
        data: bytes,
    ) -> CheckpointRef: ...

    async def get_checkpoint(self, owner_id: str, run_id: str) -> Optional[bytes]: ...

    async def delete_checkpoint(self, owner_id: str, run_id: str) -> None: ...


class JobQueue(Protocol):
    async def initialize(self) -> None: ...

    async def enqueue(self, message: JobMessage, *, deduplication_id: Optional[str] = None) -> str: ...

    async def receive(
        self,
        *,
        max_messages: int = 1,
        wait_seconds: int = 10,
        visibility_timeout: int = 60,
    ) -> tuple[QueuedJob, ...]: ...

    async def acknowledge(self, job: QueuedJob) -> None: ...

    async def release(self, job: QueuedJob, *, delay_seconds: int = 0) -> None: ...

    async def extend_visibility(
        self,
        job: QueuedJob,
        *,
        visibility_timeout: int,
    ) -> bool: ...
