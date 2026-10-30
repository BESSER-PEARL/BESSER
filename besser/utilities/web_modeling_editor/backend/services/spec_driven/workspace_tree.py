"""Link-free copying of run workspaces.

A run workspace is written by model-authored code, so a symlink in it (e.g.
``ln -s /proc/self/environ leak.txt``) must never be read through into a
download zip, a modify seed, or a GitHub push. Only regular files and real
directories are kept; symlinks, devices, sockets and FIFOs are dropped.
The run's own record files are read and written without following links.
"""

import contextlib
import json
import os
import shutil
import stat
import tempfile
from typing import Iterable


def is_plain_entry(path: str) -> bool:
    """True for a regular file or a real directory, without following links."""
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) or stat.S_ISDIR(mode)


def copytree_no_links(src: str, dst: str, ignore_patterns: Iterable[str] = ()) -> None:
    """``shutil.copytree`` into an existing ``dst`` that skips every non-plain entry.

    ``ignore_patterns`` are ``shutil.ignore_patterns`` globs matched on base names.
    """
    by_pattern = shutil.ignore_patterns(*ignore_patterns)

    def _ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set(by_pattern(directory, names))
        ignored.update(
            name for name in names
            if not is_plain_entry(os.path.join(directory, name))
        )
        return ignored

    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=_ignore)


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
    fd, tmp = tempfile.mkstemp(prefix=".besser_tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
