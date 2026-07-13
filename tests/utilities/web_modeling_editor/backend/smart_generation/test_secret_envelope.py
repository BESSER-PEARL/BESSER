from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from besser.utilities.web_modeling_editor.backend.services.smart_generation.secret_envelope import (
    EncryptedSecret,
    KmsSecretEnvelope,
    LocalSecretEnvelope,
    SecretEnvelopeError,
    build_secret_envelope,
)


def test_local_envelope_round_trip_is_bound_to_owner_and_run() -> None:
    envelope = LocalSecretEnvelope(Fernet.generate_key())
    encrypted = envelope.encrypt("sk-private", run_id="a" * 32, owner_id="github:42")

    assert encrypted.algorithm == "fernet-process-v1"
    assert "sk-private" not in encrypted.ciphertext
    assert envelope.decrypt(
        encrypted, run_id="a" * 32, owner_id="github:42"
    ) == "sk-private"

    with pytest.raises(SecretEnvelopeError):
        envelope.decrypt(encrypted, run_id="b" * 32, owner_id="github:42")
    with pytest.raises(SecretEnvelopeError):
        envelope.decrypt(encrypted, run_id="a" * 32, owner_id="github:99")


def test_encrypted_secret_validates_serialized_shape() -> None:
    secret = EncryptedSecret.from_dict(
        {"algorithm": "aws-kms-v1", "ciphertext": "YWJj"}
    )
    assert secret.to_dict() == {
        "algorithm": "aws-kms-v1",
        "ciphertext": "YWJj",
    }
    with pytest.raises(SecretEnvelopeError):
        EncryptedSecret.from_dict({"algorithm": "aws-kms-v1"})


class _FakeKms:
    def __init__(self) -> None:
        self.context = None

    def encrypt(self, **kwargs):
        self.context = kwargs["EncryptionContext"]
        return {"CiphertextBlob": b"ciphertext"}

    def decrypt(self, **kwargs):
        assert kwargs["EncryptionContext"] == self.context
        assert kwargs["CiphertextBlob"] == b"ciphertext"
        return {"Plaintext": b"sk-kms"}


def test_kms_envelope_uses_encryption_context() -> None:
    client = _FakeKms()
    envelope = KmsSecretEnvelope("alias/test", client=client)
    encrypted = envelope.encrypt("sk-kms", run_id="c" * 32, owner_id="github:7")

    assert client.context == {
        "besser:run_id": "c" * 32,
        "besser:owner_id": "github:7",
    }
    assert envelope.decrypt(
        encrypted, run_id="c" * 32, owner_id="github:7"
    ) == "sk-kms"


def test_production_envelope_fails_closed_without_kms_key(monkeypatch) -> None:
    monkeypatch.delenv("BESSER_SMARTGEN_KMS_KEY_ID", raising=False)
    with pytest.raises(SecretEnvelopeError):
        build_secret_envelope("production")


def test_unknown_state_mode_is_rejected() -> None:
    with pytest.raises(SecretEnvelopeError):
        build_secret_envelope("maybe")
