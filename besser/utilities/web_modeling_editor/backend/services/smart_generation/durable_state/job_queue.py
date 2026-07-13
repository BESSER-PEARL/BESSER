"""In-memory and SQS job queues for detached SmartGen workers."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, Mapping, Optional

from .dependencies import require_optional_dependency
from .errors import DurableStateConfigurationError, StorageIntegrityError
from .models import JobMessage, QueuedJob

_SENSITIVE_JOB_KEYS = frozenset({
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "client_secret",
    "private_key",
    "credential",
    "credentials",
})
_SAFE_SECRET_SUFFIXES = ("_ciphertext", "_reference", "_ref", "_secret_id")


def _assert_no_plaintext_secrets(value: Any, path: str = "payload") -> None:
    """Reject common plaintext credential fields before a job reaches a queue."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized in _SENSITIVE_JOB_KEYS and not normalized.endswith(_SAFE_SECRET_SUFFIXES):
                raise ValueError(
                    f"Queued SmartGen jobs must not contain plaintext secrets ({path}.{key})"
                )
            _assert_no_plaintext_secrets(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_plaintext_secrets(child, f"{path}[{index}]")


def _serialize_message(message: JobMessage) -> str:
    _assert_no_plaintext_secrets(message.payload)
    return json.dumps(
        {
            "schema_version": 1,
            "run_id": message.run_id,
            "owner_id": message.owner_id,
            "payload": dict(message.payload),
            "created_at": message.created_at,
        },
        separators=(",", ":"),
        default=str,
    )


def _deserialize_message(body: str) -> JobMessage:
    try:
        value = json.loads(body)
        if value.get("schema_version") != 1:
            raise ValueError("unsupported schema version")
        message = JobMessage(
            run_id=value["run_id"],
            owner_id=value["owner_id"],
            payload=value.get("payload") or {},
            created_at=float(value["created_at"]),
        )
        _assert_no_plaintext_secrets(message.payload)
        return message
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("SmartGen queue message has an invalid shape") from exc


class InMemoryJobQueue:
    """Local queue with explicit acknowledge/release semantics."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[JobMessage, str, int]] = asyncio.Queue()
        self._inflight: dict[str, tuple[JobMessage, str, int]] = {}
        self._deduplication: dict[str, str] = {}

    async def initialize(self) -> None:
        return None

    async def enqueue(self, message: JobMessage, *, deduplication_id: Optional[str] = None) -> str:
        _assert_no_plaintext_secrets(message.payload)
        if deduplication_id and deduplication_id in self._deduplication:
            return self._deduplication[deduplication_id]
        message_id = uuid.uuid4().hex
        if deduplication_id:
            self._deduplication[deduplication_id] = message_id
        await self._queue.put((message, message_id, 0))
        return message_id

    async def receive(
        self,
        *,
        max_messages: int = 1,
        wait_seconds: int = 10,
        visibility_timeout: int = 60,
    ) -> tuple[QueuedJob, ...]:
        del visibility_timeout
        if not 1 <= max_messages <= 10:
            raise ValueError("max_messages must be between 1 and 10")
        if not 0 <= wait_seconds <= 20:
            raise ValueError("wait_seconds must be between 0 and 20")
        items: list[tuple[JobMessage, str, int]] = []
        try:
            if wait_seconds == 0:
                items.append(self._queue.get_nowait())
            else:
                items.append(await asyncio.wait_for(self._queue.get(), timeout=wait_seconds))
        except (asyncio.QueueEmpty, asyncio.TimeoutError):
            return ()
        while len(items) < max_messages:
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        jobs = []
        for message, message_id, previous_count in items:
            receipt = uuid.uuid4().hex
            receive_count = previous_count + 1
            self._inflight[receipt] = (message, message_id, receive_count)
            jobs.append(QueuedJob(message, receipt, message_id, receive_count))
        return tuple(jobs)

    async def acknowledge(self, job: QueuedJob) -> None:
        self._inflight.pop(job.receipt_handle, None)

    async def release(self, job: QueuedJob, *, delay_seconds: int = 0) -> None:
        if not 0 <= delay_seconds <= 900:
            raise ValueError("delay_seconds must be between 0 and 900")
        item = self._inflight.pop(job.receipt_handle, None)
        if item is None:
            return
        if delay_seconds:
            await asyncio.sleep(delay_seconds)
        await self._queue.put(item)

    async def extend_visibility(
        self,
        job: QueuedJob,
        *,
        visibility_timeout: int,
    ) -> bool:
        if not 1 <= visibility_timeout <= 43200:
            raise ValueError("visibility_timeout must be between 1 and 43200")
        return job.receipt_handle in self._inflight


class SQSJobQueue:
    """SQS adapter; queue encryption and a dead-letter policy are expected in IaC."""

    def __init__(
        self,
        queue_url: str,
        *,
        region_name: Optional[str] = None,
        client: Any = None,
    ) -> None:
        if not queue_url or not queue_url.startswith("https://"):
            raise DurableStateConfigurationError("A valid HTTPS SmartGen SQS queue URL is required")
        self.queue_url = queue_url
        self._fifo = queue_url.endswith(".fifo")
        if client is None:
            boto3 = require_optional_dependency("boto3", "the 'boto3' package")
            client = boto3.client("sqs", region_name=region_name)
        self._client = client

    async def initialize(self) -> None:
        response = await asyncio.to_thread(
            self._client.get_queue_attributes,
            QueueUrl=self.queue_url,
            AttributeNames=["QueueArn", "SqsManagedSseEnabled", "KmsMasterKeyId"],
        )
        attributes = response.get("Attributes") or {}
        if (
            str(attributes.get("SqsManagedSseEnabled", "")).lower() != "true"
            and not attributes.get("KmsMasterKeyId")
        ):
            raise DurableStateConfigurationError(
                "The SmartGen SQS queue must enable SQS-managed SSE or an AWS KMS key"
            )

    async def enqueue(self, message: JobMessage, *, deduplication_id: Optional[str] = None) -> str:
        body = _serialize_message(message)
        request: dict[str, Any] = {
            "QueueUrl": self.queue_url,
            "MessageBody": body,
            "MessageAttributes": {
                "run_id": {"DataType": "String", "StringValue": message.run_id},
            },
        }
        if self._fifo:
            request["MessageGroupId"] = f"run-{message.run_id}"
            request["MessageDeduplicationId"] = deduplication_id or message.run_id
        response = await asyncio.to_thread(self._client.send_message, **request)
        message_id = response.get("MessageId")
        if not message_id:
            raise StorageIntegrityError("SQS did not return a message ID")
        return str(message_id)

    async def receive(
        self,
        *,
        max_messages: int = 1,
        wait_seconds: int = 10,
        visibility_timeout: int = 60,
    ) -> tuple[QueuedJob, ...]:
        if not 1 <= max_messages <= 10:
            raise ValueError("max_messages must be between 1 and 10")
        if not 0 <= wait_seconds <= 20:
            raise ValueError("wait_seconds must be between 0 and 20")
        if not 1 <= visibility_timeout <= 43200:
            raise ValueError("visibility_timeout must be between 1 and 43200")
        response = await asyncio.to_thread(
            self._client.receive_message,
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_seconds,
            VisibilityTimeout=visibility_timeout,
            AttributeNames=["ApproximateReceiveCount"],
        )
        jobs = []
        for raw in response.get("Messages") or []:
            jobs.append(QueuedJob(
                message=_deserialize_message(raw.get("Body", "")),
                receipt_handle=raw["ReceiptHandle"],
                message_id=raw["MessageId"],
                receive_count=int((raw.get("Attributes") or {}).get("ApproximateReceiveCount", 1)),
            ))
        return tuple(jobs)

    async def acknowledge(self, job: QueuedJob) -> None:
        await asyncio.to_thread(
            self._client.delete_message,
            QueueUrl=self.queue_url,
            ReceiptHandle=job.receipt_handle,
        )

    async def release(self, job: QueuedJob, *, delay_seconds: int = 0) -> None:
        if not 0 <= delay_seconds <= 43200:
            raise ValueError("delay_seconds must be between 0 and 43200")
        await asyncio.to_thread(
            self._client.change_message_visibility,
            QueueUrl=self.queue_url,
            ReceiptHandle=job.receipt_handle,
            VisibilityTimeout=delay_seconds,
        )

    async def extend_visibility(
        self,
        job: QueuedJob,
        *,
        visibility_timeout: int,
    ) -> bool:
        if not 1 <= visibility_timeout <= 43200:
            raise ValueError("visibility_timeout must be between 1 and 43200")
        await asyncio.to_thread(
            self._client.change_message_visibility,
            QueueUrl=self.queue_url,
            ReceiptHandle=job.receipt_handle,
            VisibilityTimeout=visibility_timeout,
        )
        return True
