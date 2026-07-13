"""Public factory for local and production SmartGen durability adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .blob_store import FileSystemBlobStore, S3BlobStore
from .config import DurableStateConfig, DurableStateMode
from .contracts import ArtifactStorage, CheckpointStorage, DurableStateStore, JobQueue
from .job_queue import InMemoryJobQueue, SQSJobQueue
from .postgres_store import PostgresStateStore
from .sqlite_store import SQLiteStateStore


@dataclass(slots=True)
class DurableStateFoundation:
    """One fully selected adapter set; production never silently downgrades."""

    state: DurableStateStore
    artifacts: ArtifactStorage
    checkpoints: CheckpointStorage
    queue: JobQueue
    config: DurableStateConfig

    async def initialize(self) -> None:
        await self.state.initialize()
        await self.artifacts.initialize()
        await self.queue.initialize()

    async def close(self) -> None:
        await self.state.close()


def build_durable_state(
    config: Optional[DurableStateConfig] = None,
) -> DurableStateFoundation:
    """Build the configured adapters without any production fallback."""

    selected = config or DurableStateConfig.from_env()
    if selected.mode == DurableStateMode.LOCAL:
        assert selected.sqlite_path is not None
        assert selected.storage_dir is not None
        state = SQLiteStateStore(selected.sqlite_path)
        blobs = FileSystemBlobStore(selected.storage_dir)
        queue = InMemoryJobQueue()
    else:
        assert selected.database_url is not None
        assert selected.s3_bucket is not None
        assert selected.sqs_queue_url is not None
        state = PostgresStateStore(selected.database_url)
        blobs = S3BlobStore(
            selected.s3_bucket,
            prefix=selected.s3_prefix,
            region_name=selected.aws_region,
            kms_key_id=selected.s3_kms_key_id,
        )
        queue = SQSJobQueue(
            selected.sqs_queue_url,
            region_name=selected.aws_region,
        )
    return DurableStateFoundation(
        state=state,
        artifacts=blobs,
        checkpoints=blobs,
        queue=queue,
        config=selected,
    )
