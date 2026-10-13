"""Link-free traversal, reads and writes of run workspaces.

The workspace is written by model-authored commands, but the harness reads it
unsandboxed: a planted ``leak.txt -> /proc/self/environ`` read by a walk, a
snapshot copy or a prompt section puts the worker's keys in front of the
model, and a write to a planted ``.besser_trace.jsonl -> <outside file>``
lands outside the run. Only regular files and real directories are visited,
read or copied; symlinks, devices, sockets and FIFOs are skipped, and a write
replaces a link rather than following it. Same semantics as the backend's
``services/spec_driven/workspace_tree.py``.
"""

import os
import shutil
import stat
import tempfile
from typing import Iterable, Iterator


def is_plain_entry(path: str) -> bool:
    """True for a regular file or a real directory, without following links."""
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) or stat.S_ISDIR(mode)


def _refuse_link_parents(path: str, root: "str | None") -> str:
    """``path`` made absolute; ``OSError`` when it is outside ``root`` or a
    directory between ``root`` and it is a link."""
    path = os.path.abspath(path)
    if root is not None:
        root = os.path.abspath(root)
        try:
            rel = os.path.relpath(path, root)
        except ValueError:  # another drive
            raise PermissionError(f"outside the workspace: {path}") from None
        if rel == os.pardir or rel.startswith(os.pardir + os.sep):
            raise PermissionError(f"outside the workspace: {path}")
        current = root
        for part in rel.split(os.sep)[:-1]:
            current = os.path.join(current, part)
            if os.path.islink(current):
                raise PermissionError(f"reached through a link: {path}")
    return path


def _refuse_unless_plain_file(path: str, root: "str | None") -> None:
    """Raise ``OSError`` unless ``path`` is a regular file reached without a
    link: the file itself, and every directory between ``root`` and it."""
    path = _refuse_link_parents(path, root)
    if not stat.S_ISREG(os.lstat(path).st_mode):  # FileNotFoundError when absent
        raise PermissionError(f"not a regular file (link or special): {path}")


def is_plain_file(path, root: "str | None" = None) -> bool:
    """``os.path.isfile`` that is False for anything :func:`open_plain` refuses."""
    try:
        _refuse_unless_plain_file(os.fspath(path), None if root is None else os.fspath(root))
        return True
    except OSError:
        return False


def open_plain(path, mode: str = "r", *, root: "str | None" = None, **kwargs):
    """``open()`` for a workspace file that refuses a link or special file.

    Raises ``OSError`` (``PermissionError``) instead of following it, so call
    sites that already treat an unreadable file as absent keep working.
    """
    _refuse_unless_plain_file(os.fspath(path), None if root is None else os.fspath(root))
    return open(path, mode, **kwargs)


def read_plain_text(path, root: "str | None" = None, **kwargs) -> "str | None":
    """The file's text, or ``None`` when it is absent, a link, special, or
    undecodable. ``kwargs`` go to ``open`` (encoding defaults to utf-8)."""
    kwargs.setdefault("encoding", "utf-8")
    try:
        with open_plain(path, "r", root=root, **kwargs) as handle:
            return handle.read()
    except (OSError, UnicodeError):
        return None


def open_plain_write(path, mode: str = "w", *, root: "str | None" = None, **kwargs):
    """``open(path, mode)`` for writing (``w``/``a``/``x``, text or binary)
    that never writes through a link.

    A link planted at ``path`` is removed and a real file created in its
    place; a directory or special file there, a link parent or a path outside
    ``root`` raises ``OSError``. ``O_NOFOLLOW`` closes the check/open race
    where the platform has it.
    """
    path = _refuse_link_parents(os.fspath(path), None if root is None else os.fspath(root))
    try:
        existing = os.lstat(path).st_mode
    except FileNotFoundError:
        existing = None
    if existing is not None and not stat.S_ISREG(existing):
        if not stat.S_ISLNK(existing):
            raise PermissionError(f"not a regular file: {path}")
        os.unlink(path)
    flags = (os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_BINARY", 0))  # newlines stay Python's job
    if "a" in mode:
        flags |= os.O_APPEND
    elif "x" in mode:
        flags |= os.O_EXCL
    else:
        flags |= os.O_TRUNC
    fd = os.open(path, flags, 0o666)
    try:
        return os.fdopen(fd, mode.replace("x", "w"), **kwargs)
    except Exception:
        os.close(fd)
        raise


def write_atomic_plain(path, text: str, *, root: "str | None" = None,
                       encoding: str = "utf-8") -> None:
    """Replace ``path`` with ``text`` via a fresh temp file and a rename.

    ``mkstemp`` creates the temp file with ``O_EXCL``, and the rename
    replaces a link at ``path`` rather than writing through it.
    """
    path = _refuse_link_parents(os.fspath(path), None if root is None else os.fspath(root))
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".besser_tmp_")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def walk_plain(top: str) -> Iterator[tuple[str, list[str], list[str]]]:
    """``os.walk(top)`` without links or special files.

    ``dirs`` is os.walk's own list, so in-place pruning (``dirs[:] = ...``)
    still steers the walk.
    """
    for root, dirs, files in os.walk(top):
        dirs[:] = [d for d in dirs if is_plain_entry(os.path.join(root, d))]
        yield root, dirs, [f for f in files if is_plain_entry(os.path.join(root, f))]


def copytree_plain(src: str, dst: str, ignore_patterns: Iterable[str] = (),
                   dirs_exist_ok: bool = False) -> None:
    """``shutil.copytree`` that drops every non-plain entry instead of copying
    what a link points at. ``ignore_patterns`` are base-name globs."""
    by_pattern = shutil.ignore_patterns(*ignore_patterns)

    def _ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set(by_pattern(directory, names))
        ignored.update(n for n in names if not is_plain_entry(os.path.join(directory, n)))
        return ignored

    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=dirs_exist_ok)
