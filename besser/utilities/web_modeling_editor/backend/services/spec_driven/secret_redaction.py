"""Credential redaction for generated artifacts and streamed run events.

The user's BYOK credential is never intentionally passed to the model, but a
generated project can still contain a credential copied from imported content
or invented/pasted in an instruction.  Treat the artifact and SSE serializers
as the final trust boundary: remove populated ``.env`` files and replace
credential-shaped text everywhere else with a stable placeholder.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable


REDACTED_SECRET = "[REDACTED]"
_MAX_TEXT_FILE_BYTES = 2 * 1024 * 1024

# .env-family files whose content is safe to publish when it only contains
# placeholders. They are still scanned and redacted if a real token appears.
_ENV_SAFE_SUFFIXES = (".example", ".sample", ".template", ".dist")

_SECRET_ENV_ASSIGNMENT_RE = re.compile(
    r"(?im)^(\s*[A-Za-z0-9_]*"
    r"(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE[_-]?KEY|ACCESS[_-]?KEY)"
    r"[A-Za-z0-9_]*\s*=\s*)([^\r\n]+)$"
)

# Provider tokens with sufficiently specific prefixes to avoid redacting
# ordinary prose or generated identifiers.
_SECRET_VALUE_TOKEN_RE = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{12,}"
    r"|sk-[A-Za-z0-9_\-]{16,}"
    r"|gh[posru]_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,})"
)

_ENV_PLACEHOLDER_VALUES = {
    "",
    REDACTED_SECRET.lower(),
    "changeme",
    "change-me",
    "your-key-here",
    "your_api_key",
    "xxx",
    "xxxx",
    "todo",
    "...",
    "none",
    "null",
}


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().strip('"').strip("'").strip()
    lowered = normalized.lower()
    return (
        lowered in _ENV_PLACEHOLDER_VALUES
        or normalized.startswith("${")
        or normalized.startswith("<")
        or "your" in lowered
        or "placeholder" in lowered
        or "example" in lowered
    )


def redact_text(value: str) -> tuple[str, int]:
    """Return ``value`` with credential material replaced and a match count."""
    findings = 0

    def _assignment_replacement(match: re.Match[str]) -> str:
        nonlocal findings
        if _is_placeholder(match.group(2)):
            return match.group(0)
        findings += 1
        return match.group(1) + REDACTED_SECRET

    redacted = _SECRET_ENV_ASSIGNMENT_RE.sub(_assignment_replacement, value)
    redacted, token_count = _SECRET_VALUE_TOKEN_RE.subn(REDACTED_SECRET, redacted)
    return redacted, findings + token_count


def redact_data(value: Any) -> tuple[Any, int]:
    """Recursively redact strings in JSON-like event or recipe data."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        output = []
        findings = 0
        for item in value:
            safe_item, item_findings = redact_data(item)
            output.append(safe_item)
            findings += item_findings
        return output, findings
    if isinstance(value, tuple):
        safe_list, findings = redact_data(list(value))
        return tuple(safe_list), findings
    if isinstance(value, dict):
        output = {}
        findings = 0
        for key, item in value.items():
            safe_item, item_findings = redact_data(item)
            output[key] = safe_item
            findings += item_findings
        return output, findings
    return value, 0


def looks_like_secret_env(content: str) -> bool:
    """Whether a non-template ``.env`` body carries a populated credential."""
    _redacted, findings = redact_text(content)
    return findings > 0


@dataclass(frozen=True)
class SecretScrubResult:
    removed_files: tuple[str, ...] = ()
    redacted_files: tuple[str, ...] = ()
    findings: int = 0


def scrub_secret_files(
    workdir: str,
    *,
    excluded_names: Iterable[str] = (),
) -> SecretScrubResult:
    """Remove populated ``.env`` files and redact secrets in other text files.

    Files are modified in place before they are registered for download or
    copied to GitHub. Symlinks and files outside ``workdir`` are never followed.
    Unreadable, binary, and files larger than 2 MiB are left untouched.
    """
    root_path = os.path.realpath(workdir)
    excluded = tuple(excluded_names)
    removed: list[str] = []
    redacted_files: list[str] = []
    findings = 0

    for root, dirs, files in os.walk(root_path, followlinks=False):
        dirs[:] = [
            name
            for name in dirs
            if not os.path.islink(os.path.join(root, name))
            and not any(fnmatch.fnmatch(name, pattern) for pattern in excluded)
        ]
        for name in files:
            if any(fnmatch.fnmatch(name, pattern) for pattern in excluded):
                continue
            full_path = os.path.join(root, name)
            if os.path.islink(full_path):
                continue
            try:
                if os.path.commonpath((root_path, os.path.realpath(full_path))) != root_path:
                    continue
                if os.path.getsize(full_path) > _MAX_TEXT_FILE_BYTES:
                    continue
                with open(full_path, "rb") as handle:
                    raw = handle.read()
                if b"\x00" in raw:
                    continue
                content = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError, ValueError):
                continue

            safe_content, file_findings = redact_text(content)
            if not file_findings:
                continue

            relative = os.path.relpath(full_path, root_path).replace("\\", "/")
            lowered = name.lower()
            is_env = lowered.startswith(".env")
            is_safe_template = any(lowered.endswith(suffix) for suffix in _ENV_SAFE_SUFFIXES)
            if is_env and not is_safe_template:
                try:
                    os.remove(full_path)
                except OSError:
                    continue
                removed.append(relative)
            else:
                try:
                    with open(full_path, "w", encoding="utf-8", newline="") as handle:
                        handle.write(safe_content)
                except OSError:
                    continue
                redacted_files.append(relative)
            findings += file_findings

    return SecretScrubResult(
        removed_files=tuple(removed),
        redacted_files=tuple(redacted_files),
        findings=findings,
    )
