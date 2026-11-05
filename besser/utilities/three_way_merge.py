"""Three-way merge of file trees, for regenerating output that someone has edited.

Runs on libgit2 through ``pygit2``, an optional dependency (``besser[regen]``,
also in ``besser[agents]``). Check :func:`merge_available` before merging.
"""

from __future__ import annotations

import tempfile
from typing import NamedTuple

from besser.utilities.workspace_fs import is_text

try:
    import pygit2
    from pygit2.enums import FileMode, MergeFlag
except ImportError:
    pygit2 = None


class TreeMerge(NamedTuple):
    """A merge of ``{path: content}`` trees."""

    clean: dict[str, bytes]
    #: Path -> the file with conflict markers. ``None`` when markers cannot show the
    #: conflict: one side deleted the file, or a side is not UTF-8 text.
    conflicts: dict[str, bytes | None]


def merge_available() -> bool:
    """Whether ``pygit2`` is installed."""
    return pygit2 is not None


def merge_trees(base: dict[str, bytes], ours: dict[str, bytes], theirs: dict[str, bytes]) -> TreeMerge:
    """Merge three ``{path: content}`` trees.

    Renames are not guessed: libgit2 misses a class rename once the file was
    edited, and a guess could pair a removed class's file with an added one.
    A caller that knows a file moved passes it under its new path in ``base``
    and ``ours``.
    """
    # TemporaryDirectory, as libgit2's objects are read-only: on Windows rmtree cannot remove them.
    with tempfile.TemporaryDirectory(prefix="besser_merge_", ignore_cleanup_errors=True) as work:
        repo = pygit2.init_repository(work, bare=True)

        def tree(files: dict[str, bytes]):
            index = pygit2.Index()
            for path, content in files.items():
                index.add(pygit2.IndexEntry(path, repo.create_blob(content), FileMode.BLOB))
            return index.write_tree(repo)

        index = repo.merge_trees(tree(base), tree(ours), tree(theirs), flags=MergeFlag(0))
        conflicts = {}
        for ancestor, mine, their in index.conflicts or ():
            path = (mine or their or ancestor).path
            conflicts[path] = _with_markers(repo, ancestor, mine, their) if mine and their else None
        merged = TreeMerge({e.path: repo[e.id].data for e in index if e.path not in conflicts}, conflicts)
        repo.free()
        return merged


def _with_markers(repo, ancestor, ours, theirs) -> bytes | None:
    # Checked here because libgit2 looks for NUL in the first 8000 bytes only, and pygit2 reads
    # the result as a C string: a later NUL would silently cut the file short.
    if not all(is_text(repo[side.id].data) for side in (ancestor, ours, theirs) if side):
        return None
    return repo.merge_file_from_index(ancestor, ours, theirs).contents.encode("utf-8")


def lf(content: bytes) -> bytes:
    """``content`` with LF line endings: on Windows a generator writes CRLF, and an editor may convert."""
    return content.replace(b"\r\n", b"\n")


def same_text(a: bytes | None, b: bytes | None) -> bool:
    """The same text, line endings aside."""
    return a is not None and b is not None and lf(a) == lf(b)


def with_endings_of(current: bytes, merged: bytes) -> bytes:
    """``merged`` (LF) with CRLF endings when every line of ``current`` ends with CRLF."""
    lines = current.count(b"\n")
    return merged.replace(b"\n", b"\r\n") if lines and current.count(b"\r\n") == lines else merged
