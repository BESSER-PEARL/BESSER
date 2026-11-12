"""Three-way merge of file trees, for regenerating output that someone has edited.

A merge where, for every file, a side is unchanged needs nothing more. A real merge
(both sides changed one file) runs on libgit2 through ``pygit2``, an optional dependency
(``besser[regen]``, also in ``besser[agents]``); without it, it raises :class:`MergeUnavailableError`.
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


class MergeUnavailableError(RuntimeError):
    """Some files changed on both sides, and merging them needs ``pygit2``, which is not installed."""

    def __init__(self, files: list[str]):
        super().__init__(f"merging {', '.join(files)} needs pygit2, which is not installed")
        #: The files changed on both sides, sorted.
        self.files = files


def merge_available() -> bool:
    """Whether ``pygit2`` is installed."""
    return pygit2 is not None


def merge_trees(base: dict[str, bytes], ours: dict[str, bytes], theirs: dict[str, bytes]) -> TreeMerge:
    """Merge three ``{path: content}`` trees.

    Renames are not guessed: libgit2 misses a class rename once the file was
    edited, and a guess could pair a removed class's file with an added one.
    A caller that knows a file moved passes it under its new path in ``base``
    and ``ours``.

    Raises:
        MergeUnavailableError: A file changed on both sides and ``pygit2`` is not installed.
    """
    both_changed = _changed_on_both_sides(base, ours, theirs)
    if not both_changed:
        return TreeMerge(_one_sided_merge(base, ours, theirs), {})
    if pygit2 is None:
        raise MergeUnavailableError(both_changed)
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


def _changed_on_both_sides(base: dict[str, bytes], ours: dict[str, bytes], theirs: dict[str, bytes]) -> list[str]:
    """The paths where neither side is the base and the sides differ (a missing file is ``None``)."""
    return sorted(path for path in set(base) | set(ours) | set(theirs)
                  if base.get(path) != ours.get(path) != theirs.get(path) != base.get(path))


def _one_sided_merge(base: dict[str, bytes], ours: dict[str, bytes], theirs: dict[str, bytes]) -> dict[str, bytes]:
    """The merge when every path changed on one side at most: the changed side, a deletion included."""
    merged = {}
    for path in set(base) | set(ours) | set(theirs):
        content = theirs.get(path) if ours.get(path) == base.get(path) else ours.get(path)
        if content is not None:
            merged[path] = content
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
