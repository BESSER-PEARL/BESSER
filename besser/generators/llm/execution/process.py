"""Allowlisted subprocess environments shared by tools and runtime probes."""

import os


# Environment variables that are safe to expose to LLM-invoked subprocesses.
# These are needed for basic tooling to work (PATH for binaries, HOME for
# tool caches, LANG/LC_* for locale-aware tools, TMPDIR for scratch space,
# SystemRoot/USERPROFILE on Windows). Everything else — including provider
# API keys, deployment credentials, SMTP passwords, OAuth secrets — is
# stripped so the LLM cannot `printenv` them into generated code.
_SAFE_ENV_ALLOWLIST: frozenset[str] = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL",
    "LANG", "LC_ALL", "LC_CTYPE",
    "TMPDIR", "TMP", "TEMP",
    # Windows
    "SystemRoot", "SYSTEMROOT", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    "COMSPEC", "PATHEXT", "ProgramFiles", "ProgramData",
    # Python (harmless, often needed by tools)
    "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
    # Node (harmless, often needed by tools)
    "NODE_PATH",
})

# Variable-name substrings that identify secret-like env vars. Even if
# a variable is accidentally in the allowlist, names matching these
# patterns are always stripped.
_SECRET_SUBSTRINGS: tuple[str, ...] = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "PRIVATE",
    "API_", "AUTH", "CERT", "SESSION",
)


def _safe_subprocess_env() -> dict[str, str]:
    """Return a minimal subprocess environment with secrets stripped.

    Never pass the full ``os.environ`` to an LLM-invoked subprocess —
    that would leak provider API keys, OAuth secrets, SMTP credentials,
    and any other server-side configuration. This helper constructs a
    new environment by copying only allowlisted variables and
    deliberately drops anything whose name contains a secret-like
    substring, even if it's in the allowlist.
    """
    safe: dict[str, str] = {}
    for name, value in os.environ.items():
        if name not in _SAFE_ENV_ALLOWLIST:
            continue
        upper = name.upper()
        if any(substr in upper for substr in _SECRET_SUBSTRINGS):
            continue
        safe[name] = value
    # Ensure PATH exists even if the parent somehow didn't have it.
    safe.setdefault("PATH", os.defpath)
    # Suppress .pyc writes — same behaviour as the old `env={**os.environ,...}`.
    safe["PYTHONDONTWRITEBYTECODE"] = "1"
    return safe
