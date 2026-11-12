"""Link-free traversal, reads and writes of run workspaces.

The workspace is written by model-authored commands, but the harness reads it
unsandboxed: a planted ``leak.txt -> /proc/self/environ`` read by a walk, a
snapshot copy or a prompt section puts the worker's keys in front of the
model, and a write to a planted ``.besser_trace.jsonl -> <outside file>``
lands outside the run. Only regular files and real directories are visited,
read or copied; symlinks, devices, sockets and FIFOs are skipped, and a write
replaces a link rather than following it.
"""

import os
import posixpath
import shutil
import stat
import tempfile
from typing import Iterable, Iterator

# An NTFS junction: a directory to os.lstat and os.path.islink, though it leads elsewhere like a link.
_IO_REPARSE_TAG_MOUNT_POINT = 0xA0000003


def is_link(path: str) -> bool:
    """True for a symlink or an NTFS junction, without following it."""
    try:
        info = os.lstat(path)
    except (OSError, ValueError):
        return False
    return stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", 0) == _IO_REPARSE_TAG_MOUNT_POINT


def is_plain_entry(path: str) -> bool:
    """True for a regular file or a real directory, without following links."""
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return False
    return (stat.S_ISREG(mode) or stat.S_ISDIR(mode)) and not is_link(path)


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
            if is_link(current):
                raise PermissionError(f"reached through a link: {path}")
    return path


# The open never follows a link at the leaf; the type is then checked on the
# open descriptor.
_NO_LINK_NO_WAIT = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
# O_PATH: a directory handle that needs no read permission (Linux).
_DIR_HANDLE = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
# Where the platform can, each directory below the root is opened relative to
# its parent's descriptor without following links, so a parent swapped for a
# link after a check is refused rather than followed (no window between the
# check and the open). Elsewhere (Windows) the static checks are all there is.
_BENEATH = bool(getattr(os, "O_NOFOLLOW", 0)) and os.open in os.supports_dir_fd


def _close(fd: "int | None") -> None:
    if fd is not None:
        os.close(fd)


def _open_dirs(root: str, parts: list[str], create: bool = False) -> int:
    """A descriptor of ``root/parts...``, entering each part without
    following a link (``OSError`` on one); ``create`` makes missing parts."""
    fd = os.open(os.path.abspath(root), _DIR_HANDLE & ~getattr(os, "O_NOFOLLOW", 0))
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, _DIR_HANDLE, dir_fd=fd)
            os.close(fd)
            fd = child
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_parent_dir(path: str, root: "str | None") -> "tuple[int | None, str]":
    """``(dir_fd, name)`` to reach the absolute ``path`` with ``dir_fd=``
    calls: the parent opened link-free beneath ``root``, or ``(None, path)``
    without a root or on a platform without ``dir_fd``."""
    if root is None or not _BENEATH:
        return None, path
    *dirs, name = os.path.relpath(path, os.path.abspath(root)).split(os.sep)
    return _open_dirs(root, dirs), name


def _fdopen_regular(fd: int, path: str, mode: str, kwargs: dict, truncate: bool = False):
    """``os.fdopen(fd)`` once ``fd`` is known to be a regular file, emptied
    first when ``truncate`` (never a FIFO or device that slipped in)."""
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PermissionError(f"not a regular file (link or special): {path}")
        if truncate:
            os.ftruncate(fd, 0)
        return os.fdopen(fd, mode, **kwargs)
    except BaseException:
        os.close(fd)
        raise


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
    except (OSError, ValueError):  # ValueError: a NUL byte in the path
        return False


def open_plain(path, mode: str = "r", *, root: "str | None" = None, **kwargs):
    """``open()`` for reading a workspace file that refuses a link or special file.

    Raises ``OSError`` (``PermissionError``) instead of following it, so call
    sites that already treat an unreadable file as absent keep working. The
    open itself never follows a link (see :func:`_open_parent_dir`) and does
    not block on a FIFO swapped in after the check.
    """
    if "r" not in mode or "+" in mode:
        raise ValueError(f"open_plain only reads; use open_plain_write for mode {mode!r}")
    path = os.fspath(path)
    root = None if root is None else os.fspath(root)
    _refuse_unless_plain_file(path, root)
    parent, name = _open_parent_dir(os.path.abspath(path), root)
    try:
        fd = os.open(name, os.O_RDONLY | _NO_LINK_NO_WAIT, dir_fd=parent)
    finally:
        _close(parent)
    return _fdopen_regular(fd, path, mode, kwargs)


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
    ``root`` raises ``OSError``. The open never follows a link and does not
    block on a FIFO, and the file is truncated only once it is known to be a
    regular file.
    """
    path = _refuse_link_parents(os.fspath(path), None if root is None else os.fspath(root))
    parent, name = _open_parent_dir(path, None if root is None else os.fspath(root))
    try:
        try:
            existing = os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing):
            if not stat.S_ISLNK(existing):
                raise PermissionError(f"not a regular file: {path}")
            os.unlink(name, dir_fd=parent)
        flags = os.O_WRONLY | os.O_CREAT | _NO_LINK_NO_WAIT
        if "a" in mode:
            flags |= os.O_APPEND
        elif "x" in mode:
            flags |= os.O_EXCL
        fd = os.open(name, flags, 0o666, dir_fd=parent)
    finally:
        _close(parent)
    return _fdopen_regular(fd, path, mode.replace("x", "w"), kwargs, truncate="w" in mode)


def remove_plain(path, *, root: "str | None" = None) -> None:
    """``os.remove`` that never reaches the file through a link parent.

    A link at ``path`` is removed itself, as ``os.remove`` does.
    """
    path = _refuse_link_parents(os.fspath(path), None if root is None else os.fspath(root))
    parent, name = _open_parent_dir(path, None if root is None else os.fspath(root))
    try:
        os.unlink(name, dir_fd=parent)
    finally:
        _close(parent)


def makedirs_plain(path, *, root: str) -> None:
    """``os.makedirs(path, exist_ok=True)`` that never creates or enters a
    directory through a link between ``root`` and ``path``."""
    path = _refuse_link_parents(os.fspath(path), os.fspath(root))
    if not _BENEATH:
        os.makedirs(path, exist_ok=True)
        return
    _close(_open_dirs(os.fspath(root), os.path.relpath(path, os.path.abspath(root)).split(os.sep), create=True))


def write_atomic_plain_bytes(path, content: bytes, *, root: "str | None" = None,
                             mode: "int | None" = None) -> None:
    """Replace ``path`` with ``content`` via a fresh temp file and a rename.

    ``mkstemp`` creates the temp file with ``O_EXCL``, and the rename replaces a
    link at ``path`` rather than writing through it. ``mode`` sets the new file's
    permissions (the temp file's are 0600).
    """
    path = _refuse_link_parents(os.fspath(path), None if root is None else os.fspath(root))
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".besser_tmp_")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def is_plain_relpath(rel: str) -> bool:
    """A plain workspace-relative path (no ``..``, ``\\``, drive, root or NUL)."""
    return (isinstance(rel, str) and rel == posixpath.normpath(rel) and not rel.startswith(("/", "../"))
            and not any(c in rel for c in ("\\", ":", "\x00")))


def is_text(content: bytes) -> bool:
    """UTF-8 without NUL bytes."""
    if b"\x00" in content:
        return False
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def write_atomic_plain(path, text: str, *, root: "str | None" = None,
                       encoding: str = "utf-8", newline: "str | None" = None) -> None:
    """:func:`write_atomic_plain_bytes` for text. ``newline`` is ``open``'s:
    ``""`` writes the line endings of ``text`` unchanged."""
    if newline != "":
        text = text.replace("\n", newline or os.linesep)
    write_atomic_plain_bytes(path, text.encode(encoding), root=root)


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
