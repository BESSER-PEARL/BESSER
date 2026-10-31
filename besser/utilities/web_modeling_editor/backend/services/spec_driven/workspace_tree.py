"""A run's own record files, read and written without following links.

A run workspace is written by model-authored code, so a symlink in it (e.g.
``ln -s /proc/self/environ leak.txt``) must never be read through. Link-free
traversal and copying of the whole tree live in
``besser.spec_driven_agent.execution.workspace_fs``.
"""

import json
import os
import stat

from besser.spec_driven_agent.execution.workspace_fs import write_atomic_plain


# A run's own records (recipe, GitHub source) are small; anything larger is not one.
MAX_RUN_RECORD_BYTES = 16 * 1024 * 1024


def read_run_json(path: str, max_bytes: int = MAX_RUN_RECORD_BYTES):
    """The JSON in a run's own record file, or None when it is missing, unreadable or not a plain file.

    The run's sandboxed code can replace the file with a link (to a backend
    file, or to ``/dev/zero``), so links are never followed and the size is capped.
    """
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    with os.fdopen(fd, "rb") as fh:
        if os.fstat(fh.fileno()).st_size > max_bytes:
            return None
        try:
            return json.loads(fh.read(max_bytes + 1))
        except ValueError:
            return None


def write_run_json(path: str, value) -> None:
    """Write a run's own record file by replacing it, so a link planted at ``path`` is replaced, not written through."""
    write_atomic_plain(path, json.dumps(value))
