"""A minimal environment for child processes, with secrets stripped.

Shared by every place that starts a subprocess on code or data a user (or an
LLM acting for one) controls: the spec-driven agent's tools and probes, and the
Knowledge Graph consistency check's validator process. None of them may hand
the server's own environment (provider API keys, deployment credentials, OAuth
secrets) to the child.
"""

import os

# Environment variables that are safe to expose to LLM-invoked subprocesses.
# These are needed for basic tooling to work (PATH for binaries, HOME for
# tool caches, LANG/LC_* for locale-aware tools, TMPDIR for scratch space,
# SystemRoot/USERPROFILE on Windows). Everything else — including provider
# API keys, deployment credentials, SMTP passwords, OAuth secrets — is
# stripped so the LLM cannot `printenv` them into generated code.
SAFE_ENV_ALLOWLIST: frozenset[str] = frozenset({
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
SECRET_SUBSTRINGS: tuple[str, ...] = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "PRIVATE",
    "API_", "AUTH", "CERT", "SESSION",
)

# Trust-store locations. Behind a TLS-inspecting proxy pip / npm / curl fail
# certificate verification without them, which surfaced as a false
# "Dependency conflict" blocker. They name a CA bundle (public certificates),
# not a credential, so they bypass the "CERT" deny substring - but only when
# the value is an existing path, never inline content.
CA_BUNDLE_ENV: frozenset[str] = frozenset({
    "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT", "NPM_CONFIG_CAFILE",
})


def safe_subprocess_env() -> dict[str, str]:
    """Return a minimal subprocess environment with secrets stripped.

    Never pass the full ``os.environ`` to a subprocess that runs user- or
    LLM-controlled code —
    that would leak provider API keys, OAuth secrets, SMTP credentials,
    and any other server-side configuration. This helper constructs a
    new environment by copying only allowlisted variables and
    deliberately drops anything whose name contains a secret-like
    substring, even if it's in the allowlist.
    """
    # Case-folded: Windows env var names are case-insensitive and os.environ
    # iterates them upper-cased (PROGRAMFILES, not "ProgramFiles"). Nothing
    # beyond the allowlist's own entries is admitted.
    allowed = {name.upper() for name in SAFE_ENV_ALLOWLIST}
    safe: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper in CA_BUNDLE_ENV:
            if os.path.exists(value):
                safe[name] = value
            continue
        if upper not in allowed:
            continue
        if any(substr in upper for substr in SECRET_SUBSTRINGS):
            continue
        safe[name] = value
    # Ensure PATH exists even if the parent somehow didn't have it.
    safe.setdefault("PATH", os.defpath)
    # Suppress .pyc writes.
    safe["PYTHONDONTWRITEBYTECODE"] = "1"
    return safe
