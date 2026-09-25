"""Subprocess environment and artefact locations shared by tools and probes."""

import os


# Workspace subdirectory holding the untruncated output of shell commands.
# Head+tail truncation drops the middle of a failing `tsc` / `npm run build`
# log, which is where its errors are; the full log is spilled here so
# search_in_files / read_file can still reach them. Run-internal: excluded
# from packaging, the push, the scaffold inventory and the recipe manifest.
COMMAND_OUTPUT_DIR = ".besser_command_output"


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
    # Case-folded: Windows env var names are case-insensitive and os.environ
    # iterates them upper-cased (PROGRAMFILES, not "ProgramFiles"). Nothing
    # beyond the allowlist's own entries is admitted.
    allowed = {name.upper() for name in _SAFE_ENV_ALLOWLIST}
    safe: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper not in allowed:
            continue
        if any(substr in upper for substr in _SECRET_SUBSTRINGS):
            continue
        safe[name] = value
    # Ensure PATH exists even if the parent somehow didn't have it.
    safe.setdefault("PATH", os.defpath)
    # Suppress .pyc writes.
    safe["PYTHONDONTWRITEBYTECODE"] = "1"
    return safe
