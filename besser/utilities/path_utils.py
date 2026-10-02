"""Filesystem path helpers shared by generators and backend services."""

import re

_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def normalize_relative_path(path: str) -> str:
    """Normalize a user-supplied path into a POSIX path relative to some base directory.

    Backslashes are converted to ``/``, a leading drive letter (``C:``) and
    leading slashes are stripped, and empty or ``.`` segments are dropped.

    Args:
        path (str): The path to normalize (absolute or relative, POSIX or Windows style).

    Returns:
        str: A non-empty POSIX relative path such as ``"data/files"``.

    Raises:
        ValueError: If the normalized path is empty or contains a ``..`` segment
            (which could escape the base directory).
    """
    candidate = (path or "").strip().replace("\\", "/")
    candidate = _DRIVE_PREFIX.sub("", candidate, count=1).lstrip("/")
    parts = [part for part in candidate.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise ValueError(f"Path {path!r} must not contain '..' segments.")
    if not parts:
        raise ValueError(f"Path {path!r} does not contain any relative path segment.")
    return "/".join(parts)
