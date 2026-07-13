"""Strict environment configuration for SmartGen durable infrastructure."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional
from urllib.parse import parse_qs, urlsplit

from .errors import DurableStateConfigurationError


class DurableStateMode(str, Enum):
    LOCAL = "local"
    PRODUCTION = "production"


def _positive_int(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise DurableStateConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise DurableStateConfigurationError(f"{name} must be positive")
    return value


def _required(source: Mapping[str, str], name: str) -> str:
    value = (source.get(name) or "").strip()
    if not value:
        raise DurableStateConfigurationError(
            f"{name} is required when BESSER_SMARTGEN_STATE_MODE=production"
        )
    return value


@dataclass(frozen=True, slots=True)
class DurableStateConfig:
    mode: DurableStateMode
    sqlite_path: Optional[str]
    storage_dir: Optional[str]
    database_url: Optional[str]
    s3_bucket: Optional[str]
    s3_prefix: str
    s3_kms_key_id: Optional[str]
    sqs_queue_url: Optional[str]
    aws_region: Optional[str]
    idempotency_ttl_seconds: int
    lease_ttl_seconds: int
    max_concurrent_runs_per_owner: int
    max_starts_per_hour: int
    max_storage_bytes_per_owner: int
    max_request_bytes: int
    event_page_size: int
    artifact_url_ttl_seconds: int

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "DurableStateConfig":
        source = os.environ if environ is None else environ
        raw_mode = (source.get("BESSER_SMARTGEN_STATE_MODE") or "local").strip().lower()
        try:
            mode = DurableStateMode(raw_mode)
        except ValueError as exc:
            raise DurableStateConfigurationError(
                "BESSER_SMARTGEN_STATE_MODE must be 'local' or 'production'"
            ) from exc

        default_data_dir = os.path.join(tempfile.gettempdir(), "besser_smartgen_durable")
        sqlite_path: Optional[str] = None
        storage_dir: Optional[str] = None
        database_url: Optional[str] = None
        s3_bucket: Optional[str] = None
        sqs_queue_url: Optional[str] = None

        if mode == DurableStateMode.LOCAL:
            sqlite_path = os.path.abspath(
                source.get("BESSER_SMARTGEN_SQLITE_PATH")
                or os.path.join(default_data_dir, "state.sqlite3")
            )
            storage_dir = os.path.abspath(
                source.get("BESSER_SMARTGEN_STORAGE_DIR")
                or os.path.join(default_data_dir, "blobs")
            )
        else:
            database_url = _required(source, "BESSER_SMARTGEN_DATABASE_URL")
            if not database_url.startswith(("postgres://", "postgresql://")):
                raise DurableStateConfigurationError(
                    "BESSER_SMARTGEN_DATABASE_URL must use postgres:// or postgresql://"
                )
            try:
                database_parameters = parse_qs(urlsplit(database_url).query)
            except ValueError as exc:
                raise DurableStateConfigurationError(
                    "BESSER_SMARTGEN_DATABASE_URL is invalid"
                ) from exc
            ssl_mode = (database_parameters.get("sslmode") or [""])[-1].lower()
            if ssl_mode != "verify-full":
                raise DurableStateConfigurationError(
                    "BESSER_SMARTGEN_DATABASE_URL must use sslmode=verify-full"
                )
            ssl_root_cert = (database_parameters.get("sslrootcert") or [""])[-1]
            if not ssl_root_cert.startswith("/"):
                raise DurableStateConfigurationError(
                    "BESSER_SMARTGEN_DATABASE_URL must include an absolute "
                    "sslrootcert path"
                )
            s3_bucket = _required(source, "BESSER_SMARTGEN_S3_BUCKET")
            sqs_queue_url = _required(source, "BESSER_SMARTGEN_SQS_QUEUE_URL")
            if not sqs_queue_url.startswith("https://"):
                raise DurableStateConfigurationError(
                    "BESSER_SMARTGEN_SQS_QUEUE_URL must use HTTPS"
                )

        s3_prefix = (source.get("BESSER_SMARTGEN_S3_PREFIX") or "smartgen").strip("/")
        if not s3_prefix or ".." in s3_prefix.split("/"):
            raise DurableStateConfigurationError("BESSER_SMARTGEN_S3_PREFIX is invalid")

        event_page_size = _positive_int(
            source,
            "BESSER_SMARTGEN_EVENT_PAGE_SIZE",
            200,
        )
        if event_page_size > 1000:
            raise DurableStateConfigurationError(
                "BESSER_SMARTGEN_EVENT_PAGE_SIZE cannot exceed 1000"
            )
        artifact_url_ttl = _positive_int(
            source,
            "BESSER_SMARTGEN_ARTIFACT_URL_TTL_SECONDS",
            300,
        )
        if artifact_url_ttl > 3600:
            raise DurableStateConfigurationError(
                "BESSER_SMARTGEN_ARTIFACT_URL_TTL_SECONDS cannot exceed 3600"
            )

        return cls(
            mode=mode,
            sqlite_path=sqlite_path,
            storage_dir=storage_dir,
            database_url=database_url,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_kms_key_id=(source.get("BESSER_SMARTGEN_S3_KMS_KEY_ID") or "").strip() or None,
            sqs_queue_url=sqs_queue_url,
            aws_region=(source.get("BESSER_SMARTGEN_AWS_REGION") or "").strip() or None,
            idempotency_ttl_seconds=_positive_int(
                source,
                "BESSER_SMARTGEN_IDEMPOTENCY_TTL_SECONDS",
                86400,
            ),
            lease_ttl_seconds=_positive_int(
                source,
                "BESSER_SMARTGEN_LEASE_TTL_SECONDS",
                60,
            ),
            max_concurrent_runs_per_owner=_positive_int(
                source,
                "BESSER_SMARTGEN_MAX_CONCURRENT_RUNS_PER_OWNER",
                2,
            ),
            max_starts_per_hour=_positive_int(
                source,
                "BESSER_SMARTGEN_MAX_STARTS_PER_HOUR",
                10,
            ),
            max_storage_bytes_per_owner=_positive_int(
                source,
                "BESSER_SMARTGEN_MAX_STORAGE_BYTES_PER_OWNER",
                1024 * 1024 * 1024,
            ),
            max_request_bytes=_positive_int(
                source,
                "BESSER_SMARTGEN_MAX_DURABLE_REQUEST_BYTES",
                5 * 1024 * 1024,
            ),
            event_page_size=event_page_size,
            artifact_url_ttl_seconds=artifact_url_ttl,
        )
