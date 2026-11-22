"""Three-way merge of two versions of a project's ``diagrams.json``.

Git merges text line by line, so two people who each add a class to the same
model collide: both additions land at the end of the same JSON object (and of
the same B-UML source). The model is data, so it is merged as data instead:

- objects merge key by key, recursively;
- a list of objects that all carry a unique ``id`` (diagram elements in the
  React Flow format, diagrams of one type) merges item by item, as do lists of
  unique scalars (an Apollon class's attribute ids);
- a value both sides changed differently is a conflict, except layout,
  timestamps and what each person had open, where this side's value is kept
  (two people moving the same box is not worth a pull request) and each person's
  own assistant conversation, and who added an
  element and the project's own id and creation date, where the repository's
  value is kept: it was there first. Opening a repository gives the local project
  a new id and date, so those two differ for everyone who opened it.

The format is not interpreted beyond that, so the merge holds for every
diagram type and for both editor engines.
"""

from typing import Any

# Keys that only place or date things, or say what someone had open, and the saver's own assistant
# conversation (its turns have no ids to merge by); when both sides changed one, ours is kept.
OURS_WINS_KEYS = frozenset({
    "bounds", "position", "measured", "width", "height", "path",
    "exportedAt", "lastUpdate", "updatedAt", "lastModified",
    "editorVersion", "currentDiagramType", "currentDiagramIndices",
    "assistantConversation",
})
# Who added an element; the commit already in the repository was first.
THEIRS_WINS_KEYS = frozenset({"addedBy", "addedAt"})
# The project's own id and date, in an export envelope or a bare project: opening a repo mints new ones.
# Exact paths: a diagram's or an element's id identifies it, so those stay strict.
PROJECT_IDENTITY_PATHS = frozenset({"project.id", "project.createdAt", "id", "createdAt"})


class MergeConflict(Exception):
    """Both sides changed the same value differently; ``path`` says where."""

    def __init__(self, path: str):
        super().__init__(path)
        self.path = path


_MISSING = object()


def merge_values(base: Any, ours: Any, theirs: Any, path: str = "") -> Any:
    """The three-way merge of ``ours`` and ``theirs`` from ``base``; raises ``MergeConflict``.

    A missing value (``_MISSING``) is a key one side deleted or never had.
    """
    if ours == theirs:
        return ours
    if ours == base:
        return theirs
    if theirs == base:
        return ours
    key = path.rsplit(".", 1)[-1]
    if key in OURS_WINS_KEYS and ours is not _MISSING:
        return ours
    if (key in THEIRS_WINS_KEYS or path in PROJECT_IDENTITY_PATHS) and theirs is not _MISSING:
        return theirs
    if isinstance(ours, dict) and isinstance(theirs, dict):
        return _merge_dicts(base if isinstance(base, dict) else {}, ours, theirs, path)
    if isinstance(ours, list) and isinstance(theirs, list):
        merged = _merge_lists(base if isinstance(base, list) else [], ours, theirs, path)
        if merged is not None:
            return merged
    raise MergeConflict(path or "(root)")


def _merge_dicts(base: dict, ours: dict, theirs: dict, path: str) -> dict:
    merged = {}
    for key in [*theirs, *(k for k in ours if k not in theirs)]:
        value = merge_values(base.get(key, _MISSING), ours.get(key, _MISSING), theirs.get(key, _MISSING),
                             f"{path}.{key}" if path else str(key))
        if value is not _MISSING:
            merged[key] = value
    return merged


def _keyed(items: list) -> dict | None:
    """``id -> item`` when every item is an object with a unique string ``id``, else None."""
    if not all(isinstance(item, dict) and isinstance(item.get("id"), str) for item in items):
        return None
    keyed = {item["id"]: item for item in items}
    return keyed if len(keyed) == len(items) else None


def _merge_lists(base: list, ours: list, theirs: list, path: str) -> list | None:
    """Item-wise merge of id-keyed objects or unique scalars; None when the list is neither."""
    keyed = [_keyed(items) for items in (base, ours, theirs)]
    if all(k is not None for k in keyed):
        base_k, ours_k, theirs_k = keyed
        merged = _merge_dicts(base_k, ours_k, theirs_k, path)
        return list(merged.values())
    if all(_unique_scalars(items) for items in (base, ours, theirs)):
        removed = (set(base) - set(ours)) | (set(base) - set(theirs))
        result = [item for item in theirs if item not in removed]
        result += [item for item in ours if item not in removed and item not in result]
        return result
    return None


def _unique_scalars(items: list) -> bool:
    return all(isinstance(item, (str, int, float, bool)) for item in items) and len(set(items)) == len(items)


def merge_documents(base: dict | None, ours: dict, theirs: dict | None) -> dict:
    """Merge whole documents; a document new on both sides merges from an empty one."""
    if theirs is None and base is not None:
        raise MergeConflict("(deleted on the other side)")
    return merge_values(base or {}, ours, theirs or {})
