from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from besser.utilities.web_modeling_editor.backend.services.deployment.session_store import (
    SessionStore,
)


def test_external_session_key_is_not_written_beside_ciphertext(tmp_path) -> None:
    path = tmp_path / "sessions.enc"
    key = Fernet.generate_key()

    store = SessionStore(str(path), encryption_key=key)
    store.set("session", {"username": "alice"})

    assert path.exists()
    assert not path.with_suffix(".key").exists()
    reopened = SessionStore(str(path), encryption_key=key)
    assert reopened.get("session") == {"username": "alice"}


def test_production_session_store_fails_closed_without_external_key(
    tmp_path,
) -> None:
    with pytest.raises(RuntimeError, match="SESSION_STORE_FERNET_KEY"):
        SessionStore(
            str(tmp_path / "sessions.enc"),
            require_external_key=True,
        )


def test_local_session_store_keeps_backward_compatible_key_file(tmp_path) -> None:
    path = tmp_path / "sessions.enc"
    store = SessionStore(str(path))
    store.set("session", {"username": "local"})

    assert path.with_suffix(".key").exists()
    assert SessionStore(str(path)).get("session") == {"username": "local"}
