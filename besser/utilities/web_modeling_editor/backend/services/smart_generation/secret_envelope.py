"""Short-lived encryption envelopes for per-run BYOK credentials.

Production jobs may cross a durable queue before an isolated worker starts.
Provider keys must therefore never be written to PostgreSQL, SQS, or S3 in
plaintext.  This module keeps the encryption boundary small and testable:

* production uses AWS KMS with a run/owner encryption context;
* local development uses a process-local Fernet key, so ciphertext becomes
  unusable when the process exits;
* no implementation logs plaintext or ciphertext.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from cryptography.fernet import Fernet, InvalidToken


class SecretEnvelopeError(RuntimeError):
    """Raised when a BYOK envelope cannot be encrypted or decrypted."""


@dataclass(frozen=True)
class EncryptedSecret:
    """Serializable encrypted payload safe for durable job metadata."""

    algorithm: str
    ciphertext: str

    def to_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "ciphertext": self.ciphertext}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EncryptedSecret":
        algorithm = value.get("algorithm")
        ciphertext = value.get("ciphertext")
        if not isinstance(algorithm, str) or not isinstance(ciphertext, str):
            raise SecretEnvelopeError("Malformed encrypted secret envelope")
        return cls(algorithm=algorithm, ciphertext=ciphertext)


class SecretEnvelope(Protocol):
    def encrypt(self, plaintext: str, *, run_id: str, owner_id: str) -> EncryptedSecret:
        """Encrypt one credential for one owner-bound run."""

    def decrypt(self, secret: EncryptedSecret, *, run_id: str, owner_id: str) -> str:
        """Decrypt one credential for the exact same owner-bound run."""


def _context(run_id: str, owner_id: str) -> dict[str, str]:
    if not run_id or not owner_id:
        raise SecretEnvelopeError("run_id and owner_id are required for secret encryption")
    return {"besser:run_id": run_id, "besser:owner_id": owner_id}


class LocalSecretEnvelope:
    """Process-local encrypted envelopes for development and tests."""

    algorithm = "fernet-process-v1"

    def __init__(self, key: bytes | None = None) -> None:
        self._fernet = Fernet(key or Fernet.generate_key())

    def encrypt(self, plaintext: str, *, run_id: str, owner_id: str) -> EncryptedSecret:
        context = _context(run_id, owner_id)
        if not plaintext:
            raise SecretEnvelopeError("Cannot encrypt an empty provider key")
        bound = self._bind(plaintext, context)
        token = self._fernet.encrypt(bound)
        return EncryptedSecret(self.algorithm, token.decode("ascii"))

    def decrypt(self, secret: EncryptedSecret, *, run_id: str, owner_id: str) -> str:
        if secret.algorithm != self.algorithm:
            raise SecretEnvelopeError("Unsupported local secret envelope algorithm")
        context = _context(run_id, owner_id)
        try:
            bound = self._fernet.decrypt(secret.ciphertext.encode("ascii"))
        except (InvalidToken, ValueError) as exc:
            raise SecretEnvelopeError("Unable to decrypt provider key") from exc
        prefix = self._prefix(context)
        if not bound.startswith(prefix):
            raise SecretEnvelopeError("Provider key envelope owner/run mismatch")
        return bound[len(prefix) :].decode("utf-8")

    @staticmethod
    def _prefix(context: Mapping[str, str]) -> bytes:
        return (
            context["besser:owner_id"].encode("utf-8")
            + b"\x00"
            + context["besser:run_id"].encode("ascii")
            + b"\x00"
        )

    def _bind(self, plaintext: str, context: Mapping[str, str]) -> bytes:
        return self._prefix(context) + plaintext.encode("utf-8")


class KmsSecretEnvelope:
    """AWS KMS envelope implementation used by queued production jobs."""

    algorithm = "aws-kms-v1"

    def __init__(
        self,
        key_id: str,
        *,
        region_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not key_id:
            raise SecretEnvelopeError("BESSER_SMARTGEN_KMS_KEY_ID is required")
        self._key_id = key_id
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise SecretEnvelopeError(
                    "boto3 is required for production BYOK encryption"
                ) from exc
            client = boto3.client("kms", region_name=region_name)
        self._client = client

    def encrypt(self, plaintext: str, *, run_id: str, owner_id: str) -> EncryptedSecret:
        context = _context(run_id, owner_id)
        if not plaintext:
            raise SecretEnvelopeError("Cannot encrypt an empty provider key")
        try:
            response = self._client.encrypt(
                KeyId=self._key_id,
                Plaintext=plaintext.encode("utf-8"),
                EncryptionContext=context,
            )
            blob = response["CiphertextBlob"]
        except Exception as exc:
            raise SecretEnvelopeError("AWS KMS failed to encrypt provider key") from exc
        return EncryptedSecret(
            self.algorithm,
            base64.b64encode(blob).decode("ascii"),
        )

    def decrypt(self, secret: EncryptedSecret, *, run_id: str, owner_id: str) -> str:
        if secret.algorithm != self.algorithm:
            raise SecretEnvelopeError("Unsupported KMS secret envelope algorithm")
        context = _context(run_id, owner_id)
        try:
            blob = base64.b64decode(secret.ciphertext, validate=True)
            response = self._client.decrypt(
                KeyId=self._key_id,
                CiphertextBlob=blob,
                EncryptionContext=context,
            )
            plaintext = response["Plaintext"]
            return plaintext.decode("utf-8")
        except Exception as exc:
            raise SecretEnvelopeError("AWS KMS failed to decrypt provider key") from exc


_LOCAL_ENVELOPE = LocalSecretEnvelope()


def build_secret_envelope(mode: str | None = None) -> SecretEnvelope:
    """Build the configured envelope and fail closed in production mode."""

    resolved_mode = (
        mode or os.environ.get("BESSER_SMARTGEN_STATE_MODE", "local")
    ).strip().lower()
    if resolved_mode == "local":
        return _LOCAL_ENVELOPE
    if resolved_mode != "production":
        raise SecretEnvelopeError(
            "BESSER_SMARTGEN_STATE_MODE must be 'local' or 'production'"
        )
    return KmsSecretEnvelope(
        os.environ.get("BESSER_SMARTGEN_KMS_KEY_ID", ""),
        region_name=os.environ.get("BESSER_SMARTGEN_AWS_REGION") or None,
    )
