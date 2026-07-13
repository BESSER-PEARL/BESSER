"""Filesystem and S3 storage for SmartGen artifacts and checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .dependencies import require_optional_dependency
from .errors import DurableStateConfigurationError, StorageIntegrityError
from .models import (
    ArtifactRef,
    CheckpointRef,
    validate_identifier,
    validate_run_id,
)

_SAFE_FILE_NAME_RE = re.compile(r"[^A-Za-z0-9_.()\- ]+")


def _owner_key(owner_id: str) -> str:
    validate_identifier(owner_id, "owner_id")
    return hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:32]


def _safe_file_name(value: str) -> str:
    candidate = _SAFE_FILE_NAME_RE.sub("_", os.path.basename(value or "")).strip(". ")
    return (candidate or "artifact.bin")[:160]


def _sha256_file(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FileSystemBlobStore:
    """Atomic local blob store suitable for development and one-node installs."""

    def __init__(self, root_dir: str, *, clock=time.time) -> None:
        if not root_dir:
            raise DurableStateConfigurationError("Local SmartGen storage directory is required")
        self.root_dir = os.path.abspath(root_dir)
        self._clock = clock
        self._initialized = False

    async def initialize(self) -> None:
        await asyncio.to_thread(os.makedirs, self.root_dir, 0o700, True)
        self._initialized = True

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise DurableStateConfigurationError(
                "FileSystemBlobStore.initialize() must complete before use"
            )

    def _run_prefix(self, owner_id: str, run_id: str) -> str:
        return os.path.join("owners", _owner_key(owner_id), "runs", validate_run_id(run_id))

    def _absolute_key(self, storage_key: str) -> str:
        candidate = os.path.realpath(os.path.join(self.root_dir, storage_key))
        root = os.path.realpath(self.root_dir)
        candidate_cmp = os.path.normcase(candidate)
        root_cmp = os.path.normcase(root)
        if candidate_cmp != root_cmp and not candidate_cmp.startswith(root_cmp + os.sep):
            raise StorageIntegrityError("Blob storage key escapes the configured root")
        return candidate

    async def put_artifact(
        self,
        owner_id: str,
        run_id: str,
        source_path: str,
        *,
        file_name: str,
        content_type: str,
    ) -> ArtifactRef:
        self._require_initialized()
        return await asyncio.to_thread(
            self._put_artifact_sync,
            owner_id,
            run_id,
            source_path,
            file_name,
            content_type,
        )

    def _put_artifact_sync(
        self,
        owner_id: str,
        run_id: str,
        source_path: str,
        file_name: str,
        content_type: str,
    ) -> ArtifactRef:
        source = os.path.realpath(source_path)
        if not os.path.isfile(source):
            raise FileNotFoundError(source_path)
        safe_name = _safe_file_name(file_name)
        storage_key = os.path.join(
            self._run_prefix(owner_id, run_id),
            "artifacts",
            uuid.uuid4().hex,
            safe_name,
        ).replace("\\", "/")
        destination = self._absolute_key(storage_key)
        os.makedirs(os.path.dirname(destination), mode=0o700, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".upload_", dir=os.path.dirname(destination))
        os.close(fd)
        try:
            shutil.copyfile(source, temp_path)
            digest, size = _sha256_file(temp_path)
            os.replace(temp_path, destination)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return ArtifactRef(
            storage_key=storage_key,
            file_name=safe_name,
            size_bytes=size,
            sha256=digest,
            content_type=content_type or "application/octet-stream",
            created_at=self._clock(),
        )

    async def download_artifact(self, artifact: ArtifactRef, destination_path: str) -> None:
        self._require_initialized()
        await asyncio.to_thread(self._download_artifact_sync, artifact, destination_path)

    def _download_artifact_sync(self, artifact: ArtifactRef, destination_path: str) -> None:
        source = self._absolute_key(artifact.storage_key)
        if not os.path.isfile(source):
            raise FileNotFoundError(artifact.storage_key)
        destination = os.path.abspath(destination_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(source, destination)
        digest, size = _sha256_file(destination)
        if digest != artifact.sha256 or size != artifact.size_bytes:
            try:
                os.remove(destination)
            finally:
                raise StorageIntegrityError("Artifact checksum or size does not match its durable record")

    async def delete_artifact(self, artifact: ArtifactRef) -> None:
        self._require_initialized()
        path = self._absolute_key(artifact.storage_key)
        await asyncio.to_thread(self._delete_file_if_present, path)

    async def create_download_url(
        self,
        artifact: ArtifactRef,
        *,
        expires_seconds: int,
    ) -> Optional[str]:
        if expires_seconds <= 0:
            raise ValueError("expires_seconds must be positive")
        return None

    async def put_checkpoint(
        self,
        owner_id: str,
        run_id: str,
        data: bytes,
    ) -> CheckpointRef:
        self._require_initialized()
        return await asyncio.to_thread(self._put_checkpoint_sync, owner_id, run_id, data)

    def _put_checkpoint_sync(self, owner_id: str, run_id: str, data: bytes) -> CheckpointRef:
        if not isinstance(data, bytes):
            raise TypeError("checkpoint data must be bytes")
        storage_key = os.path.join(
            self._run_prefix(owner_id, run_id),
            "checkpoints",
            "latest.bin",
        ).replace("\\", "/")
        destination = self._absolute_key(storage_key)
        os.makedirs(os.path.dirname(destination), mode=0o700, exist_ok=True)
        digest = _sha256_bytes(data)
        self._atomic_write(destination, data)
        self._atomic_write(destination + ".sha256", digest.encode("ascii"))
        return CheckpointRef(
            storage_key=storage_key,
            size_bytes=len(data),
            sha256=digest,
            created_at=self._clock(),
        )

    async def get_checkpoint(self, owner_id: str, run_id: str) -> Optional[bytes]:
        self._require_initialized()
        storage_key = os.path.join(
            self._run_prefix(owner_id, run_id),
            "checkpoints",
            "latest.bin",
        ).replace("\\", "/")
        return await asyncio.to_thread(
            self._read_checkpoint_sync,
            self._absolute_key(storage_key),
        )

    async def delete_checkpoint(self, owner_id: str, run_id: str) -> None:
        self._require_initialized()
        storage_key = os.path.join(
            self._run_prefix(owner_id, run_id),
            "checkpoints",
            "latest.bin",
        ).replace("\\", "/")
        path = self._absolute_key(storage_key)
        await asyncio.to_thread(self._delete_file_if_present, path)
        await asyncio.to_thread(self._delete_file_if_present, path + ".sha256")

    @staticmethod
    def _atomic_write(path: str, data: bytes) -> None:
        fd, temp_path = tempfile.mkstemp(prefix=".write_", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def _read_checkpoint_sync(path: str) -> Optional[bytes]:
        try:
            data = Path(path).read_bytes()
        except FileNotFoundError:
            return None
        try:
            expected = Path(path + ".sha256").read_text(encoding="ascii").strip()
        except FileNotFoundError as exc:
            raise StorageIntegrityError("Checkpoint checksum sidecar is missing") from exc
        if _sha256_bytes(data) != expected:
            raise StorageIntegrityError("Checkpoint checksum does not match its durable sidecar")
        return data

    @staticmethod
    def _delete_file_if_present(path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


class S3BlobStore:
    """Encrypted S3 adapter for artifacts and resumable checkpoints."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "smartgen",
        region_name: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        client: Any = None,
        clock=time.time,
    ) -> None:
        if not bucket or not re.fullmatch(r"[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]", bucket):
            raise DurableStateConfigurationError("A valid SmartGen S3 bucket is required")
        self.bucket = bucket
        self.prefix = prefix.strip("/") or "smartgen"
        self.kms_key_id = kms_key_id
        self._clock = clock
        if client is None:
            boto3 = require_optional_dependency("boto3", "the 'boto3' package")
            client = boto3.client("s3", region_name=region_name)
        self._client = client

    async def initialize(self) -> None:
        await asyncio.to_thread(self._client.head_bucket, Bucket=self.bucket)

    def _run_prefix(self, owner_id: str, run_id: str) -> str:
        return (
            f"{self.prefix}/owners/{_owner_key(owner_id)}/runs/{validate_run_id(run_id)}"
        )

    def _encryption_args(self) -> dict[str, str]:
        if self.kms_key_id:
            return {
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": self.kms_key_id,
            }
        return {"ServerSideEncryption": "AES256"}

    async def put_artifact(
        self,
        owner_id: str,
        run_id: str,
        source_path: str,
        *,
        file_name: str,
        content_type: str,
    ) -> ArtifactRef:
        source = os.path.realpath(source_path)
        if not os.path.isfile(source):
            raise FileNotFoundError(source_path)
        digest, size = await asyncio.to_thread(_sha256_file, source)
        safe_name = _safe_file_name(file_name)
        key = f"{self._run_prefix(owner_id, run_id)}/artifacts/{uuid.uuid4().hex}/{safe_name}"
        extra_args = {
            **self._encryption_args(),
            "ContentType": content_type or "application/octet-stream",
            "Metadata": {"sha256": digest, "run-id": validate_run_id(run_id)},
        }
        await asyncio.to_thread(
            self._client.upload_file,
            source,
            self.bucket,
            key,
            ExtraArgs=extra_args,
        )
        return ArtifactRef(
            key,
            safe_name,
            size,
            digest,
            content_type or "application/octet-stream",
            self._clock(),
        )

    async def download_artifact(self, artifact: ArtifactRef, destination_path: str) -> None:
        destination = os.path.abspath(destination_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        await asyncio.to_thread(
            self._client.download_file,
            self.bucket,
            artifact.storage_key,
            destination,
        )
        digest, size = await asyncio.to_thread(_sha256_file, destination)
        if digest != artifact.sha256 or size != artifact.size_bytes:
            await asyncio.to_thread(FileSystemBlobStore._delete_file_if_present, destination)
            raise StorageIntegrityError("S3 artifact checksum or size does not match its durable record")

    async def delete_artifact(self, artifact: ArtifactRef) -> None:
        await asyncio.to_thread(
            self._client.delete_object,
            Bucket=self.bucket,
            Key=artifact.storage_key,
        )

    async def create_download_url(
        self,
        artifact: ArtifactRef,
        *,
        expires_seconds: int,
    ) -> Optional[str]:
        if not 1 <= expires_seconds <= 3600:
            raise ValueError("expires_seconds must be between 1 and 3600")
        return await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            Params={
                "Bucket": self.bucket,
                "Key": artifact.storage_key,
                "ResponseContentDisposition": f'attachment; filename="{_safe_file_name(artifact.file_name)}"',
            },
            ExpiresIn=expires_seconds,
        )

    async def put_checkpoint(
        self,
        owner_id: str,
        run_id: str,
        data: bytes,
    ) -> CheckpointRef:
        if not isinstance(data, bytes):
            raise TypeError("checkpoint data must be bytes")
        digest = _sha256_bytes(data)
        key = f"{self._run_prefix(owner_id, run_id)}/checkpoints/latest.bin"
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType="application/octet-stream",
            Metadata={"sha256": digest, "run-id": validate_run_id(run_id)},
            **self._encryption_args(),
        )
        return CheckpointRef(key, len(data), digest, self._clock())

    async def get_checkpoint(self, owner_id: str, run_id: str) -> Optional[bytes]:
        key = f"{self._run_prefix(owner_id, run_id)}/checkpoints/latest.bin"
        try:
            response = await asyncio.to_thread(
                self._client.get_object,
                Bucket=self.bucket,
                Key=key,
            )
        except Exception as exc:
            response_meta = getattr(exc, "response", {}) or {}
            code = str(response_meta.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        body = await asyncio.to_thread(response["Body"].read)
        expected = (response.get("Metadata") or {}).get("sha256")
        if not expected:
            raise StorageIntegrityError("S3 checkpoint is missing checksum metadata")
        if _sha256_bytes(body) != expected:
            raise StorageIntegrityError("S3 checkpoint checksum does not match object metadata")
        return body

    async def delete_checkpoint(self, owner_id: str, run_id: str) -> None:
        key = f"{self._run_prefix(owner_id, run_id)}/checkpoints/latest.bin"
        await asyncio.to_thread(self._client.delete_object, Bucket=self.bucket, Key=key)
