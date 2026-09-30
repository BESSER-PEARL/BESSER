"""Link-free copying of run workspaces.

A run workspace is written by model-authored code, so a symlink in it (e.g.
``ln -s /proc/self/environ leak.txt``) must never be read through into a
download zip, a modify seed, or a GitHub push. Only regular files and real
directories are kept; symlinks, devices, sockets and FIFOs are dropped.
"""

import os
import shutil
import stat
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
