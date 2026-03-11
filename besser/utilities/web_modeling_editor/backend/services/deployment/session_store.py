"""
File-based session store with encryption for OAuth tokens.

Stores sessions in a JSON file with Fernet encryption.
Thread-safe with file locking. Falls back to in-memory storage
if the cryptography package is not installed.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False
    logger.warning(
        "cryptography package not installed. "
        "SessionStore will use unencrypted in-memory storage. "
        "Install cryptography for encrypted file-based sessions: "
        "pip install cryptography"
    )


class SessionStore:
    """
    File-based encrypted session store with automatic TTL expiration.

    When the ``cryptography`` package is available, sessions are persisted
    to an encrypted file on disk. Otherwise, an in-memory dict is used
    as a transparent fallback (matching the old behaviour).

    Args:
        store_path: Path to the encrypted session file. Defaults to the
            ``SESSION_STORE_PATH`` environment variable or
            ``<tempdir>/besser_sessions.enc``.
        ttl: Time-to-live for sessions in seconds. Expired entries are
            pruned on read and via :meth:`cleanup_expired`.
    """

    def __init__(self, store_path: str = None, ttl: int = 3600):
        self._lock = threading.Lock()
        self._ttl = ttl

        if _HAS_CRYPTOGRAPHY:
            import tempfile
            default_path = os.path.join(
                tempfile.gettempdir(), "besser_sessions.enc"
            )
            self._store_path = Path(
                store_path or os.environ.get("SESSION_STORE_PATH", default_path)
            )
            self._key = self._get_or_create_key()
            self._fernet = Fernet(self._key)
            self._memory: Optional[Dict[str, Any]] = None
            logger.info(
                "SessionStore using encrypted file backend at %s",
                self._store_path,
            )
        else:
            self._store_path = None
            self._fernet = None
            self._key = None
            self._memory: Dict[str, Any] = {}
            logger.info("SessionStore using in-memory fallback (no encryption)")

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def _get_or_create_key(self) -> bytes:
        """Load or generate a Fernet encryption key stored next to the session file."""
        key_path = self._store_path.with_suffix(".key")
        if key_path.exists():
            return key_path.read_bytes()
        key = Fernet.generate_key()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(key)
        try:
            os.chmod(str(key_path), 0o600)
        except OSError:
            # os.chmod may not fully work on Windows; log and continue
            logger.debug("Could not set permissions on %s (expected on Windows)", key_path)
        return key

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load and decrypt the session data from disk (or return in-memory dict)."""
        if self._memory is not None:
            return self._memory
        if not self._store_path.exists():
            return {}
        try:
            encrypted = self._store_path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            return json.loads(decrypted)
        except Exception:
            logger.warning("Failed to decrypt session store; starting fresh")
            return {}

    def _save(self, data: dict):
        """Encrypt and persist session data to disk (or update in-memory dict)."""
        if self._memory is not None:
            self._memory.clear()
            self._memory.update(data)
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        encrypted = self._fernet.encrypt(json.dumps(data).encode())
        self._store_path.write_bytes(encrypted)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session value by key.

        Returns ``None`` if the key does not exist or has expired.
        Expired entries are pruned automatically.
        """
        with self._lock:
            data = self._load()
            entry = data.get(key)
            if entry is None:
                return None
            if time.time() - entry.get("_created", 0) < self._ttl:
                return entry.get("value")
            # Entry expired -- remove it
            del data[key]
            self._save(data)
            return None

    def set(self, key: str, value: Dict[str, Any]):
        """Store a session value with the current timestamp."""
        with self._lock:
            data = self._load()
            data[key] = {"value": value, "_created": time.time()}
            self._save(data)

    def delete(self, key: str):
        """Remove a session entry. No-op if the key does not exist."""
        with self._lock:
            data = self._load()
            if key in data:
                del data[key]
                self._save(data)

    def cleanup_expired(self):
        """Remove all entries whose TTL has elapsed."""
        with self._lock:
            data = self._load()
            now = time.time()
            expired = [
                k for k, v in data.items()
                if now - v.get("_created", 0) >= self._ttl
            ]
            for k in expired:
                del data[k]
            if expired:
                self._save(data)
                logger.debug("Cleaned up %d expired sessions", len(expired))
