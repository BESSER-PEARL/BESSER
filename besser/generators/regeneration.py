"""Bring a generated folder up to a changed model without losing edits.

``generator.regenerate()`` runs the generator in a scratch folder and merges the
result into the output folder three ways: what it generated last time (the
record file in the folder), the folder as it is now, and the new output. An edit
is either kept or flagged, never overwritten.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from besser.utilities.three_way_merge import lf, merge_available, merge_trees, same_text, with_endings_of
from besser.utilities.workspace_fs import (
    is_plain_entry,
    is_plain_file,
    open_plain,
    open_plain_write,
    read_plain_text,
    write_atomic_plain,
)

if TYPE_CHECKING:
    from besser.generators.generator_interface import GeneratorInterface

logger = logging.getLogger(__name__)

RECORD_FILENAME = ".besser_generated.json"
_RECORD_FORMAT = 1
_CONFLICT_MARKER = b"<<<<<<< "
_NOT_OUTPUT = frozenset({".git", "__pycache__", "node_modules"})
# A Spec-Driven app keeps its own base and line authorship. A merge here would update neither: the
# next modify run would read these lines as a person's edits.
_SPEC_DRIVEN_RECORDS = (".besser_generator_output.json", ".besser_attribution.json")


class RegenerationError(Exception):
    """The folder cannot be regenerated safely; nothing in it was changed."""


@dataclass
class RegenerationReport:
    """What a regeneration did. Paths are relative to the output folder, with ``/``."""

    #: Took the new output: new files, and files nobody edited.
    written: list[str] = field(default_factory=list)
    #: Edited files that now also hold the model's changes. A clean merge can still
    #: break, for instance an edit using an attribute the model removed: run your tests.
    merged: list[str] = field(default_factory=list)
    #: Hold conflict markers to resolve; the next run refuses until they are gone.
    conflicted: list[str] = field(default_factory=list)
    #: Generated files the model no longer produces, removed as nobody edited them.
    deleted: list[str] = field(default_factory=list)
    #: Edited files the model no longer produces, left in place.
    kept_removed_from_model: list[str] = field(default_factory=list)
    #: Files someone deleted and the model changed, left deleted.
    not_restored: list[str] = field(default_factory=list)
    #: Edited files that clash with the model but are not UTF-8 text, left as they were.
    not_merged: list[str] = field(default_factory=list)
    #: The folder was recorded as this model's output, and no file changed.
    adopted: bool = False


def regenerate_output(generator: GeneratorInterface, adopt: bool = False) -> RegenerationReport:
    """See :meth:`GeneratorInterface.regenerate`."""
    _require_support(generator)
    folder = generator.build_generation_dir()
    record = _load_record(folder)
    _require_usable_state(folder, record, adopt)

    output = _generate_in_scratch(generator)
    new = {rel: content for rel, content in output.items() if _is_text(content)}
    base = record["files"] if record else {}
    current = _read_files(folder, set(base) | set(new))
    new = generator._keep_generated_values(base or current, new)

    report = RegenerationReport(adopted=adopt)
    if not adopt:
        _apply(folder, base, current, new, report)
        for rel, content in output.items():
            if rel not in new and not os.path.lexists(os.path.join(folder, *rel.split("/"))):
                _write(folder, rel, content)  # binary output: written once, never merged
    _save_record(folder, new, report.conflicted)
    _log(folder, report)
    return report


def _require_support(generator: GeneratorInterface) -> None:
    if not generator.regenerable:
        raise RegenerationError(f"{type(generator).__name__} cannot regenerate: its output differs from run to run "
                                "for one model, so a merge could not tell a model change from noise.")
    if not merge_available():
        raise RegenerationError('Regenerating needs pygit2: pip install "besser[regen]"')


def _require_usable_state(folder: str, record: dict | None, adopt: bool) -> None:
    for name in _SPEC_DRIVEN_RECORDS:
        if os.path.lexists(os.path.join(folder, name)):
            raise RegenerationError(f"{folder} is a Spec-Driven Agent app (it has {name}): "
                                    "change its model through a modify run, which keeps its authorship record.")
    if adopt and record:
        raise RegenerationError(f"{folder} already has {RECORD_FILENAME}: there is nothing to adopt.")
    if not adopt and record is None and os.listdir(folder):
        raise RegenerationError(
            f"{folder} holds files but no {RECORD_FILENAME}, so edits cannot be told from generated code. Run "
            "regenerate(adopt=True) once on a generator holding the model those files came from.")
    for rel in (record or {}).get("conflicted", []):
        path = os.path.join(folder, *rel.split("/"))
        if is_plain_file(path, root=folder):
            with open_plain(path, "rb", root=folder) as handle:
                if _CONFLICT_MARKER in handle.read():
                    raise RegenerationError(f"Resolve the conflict markers in {rel} first.")


def _generate_in_scratch(generator: GeneratorInterface) -> dict[str, bytes]:
    """Run the generator elsewhere: its ``generate()`` may overwrite or delete the output folder."""
    scratch = tempfile.mkdtemp(prefix="besser_regenerate_")
    output_dir = generator.output_dir
    try:
        generator.output_dir = scratch
        generator.generate()
        return _read_tree(scratch)
    finally:
        generator.output_dir = output_dir
        shutil.rmtree(scratch, ignore_errors=True)


def _read_tree(top: str) -> dict[str, bytes]:
    files = {}
    for root, dirs, names in os.walk(top):
        dirs[:] = [d for d in dirs if d not in _NOT_OUTPUT]
        for name in names:
            path = os.path.join(root, name)
            with open(path, "rb") as handle:
                files[os.path.relpath(path, top).replace(os.sep, "/")] = handle.read()
    return files


def _read_files(folder: str, paths: set[str]) -> dict[str, bytes]:
    files = {}
    for rel in paths:
        path = os.path.join(folder, *rel.split("/"))
        if is_plain_file(path, root=folder):
            with open_plain(path, "rb", root=folder) as handle:
                files[rel] = handle.read()
    return files


def _is_text(content: bytes) -> bool:
    if b"\0" in content:
        return False
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _apply(folder: str, base: dict[str, bytes], current: dict[str, bytes], new: dict[str, bytes],
           report: RegenerationReport) -> None:
    merged, conflicts = merge_trees({rel: lf(c) for rel, c in base.items()},
                                    {rel: lf(c) for rel, c in current.items()},
                                    {rel: lf(c) for rel, c in new.items()})
    for rel in sorted(set(base) | set(current) | set(new)):
        mine = current.get(rel)
        if rel in conflicts:
            _apply_conflict(folder, rel, conflicts[rel], mine, new, report)
        elif rel not in merged:
            if mine is not None:
                os.remove(os.path.join(folder, *rel.split("/")))
                report.deleted.append(rel)
        elif mine is None or lf(mine) != merged[rel]:
            _write(folder, rel, with_endings_of(mine if mine is not None else new[rel], merged[rel]))
            untouched = mine is None or same_text(mine, base.get(rel))
            (report.written if untouched else report.merged).append(rel)


def _apply_conflict(folder: str, rel: str, content: bytes | None, mine: bytes | None, new: dict[str, bytes],
                    report: RegenerationReport) -> None:
    if content is not None:
        _write(folder, rel, with_endings_of(mine, content))
        report.conflicted.append(rel)
    elif mine is None:
        report.not_restored.append(rel)
    elif rel in new:
        report.not_merged.append(rel)
    else:
        report.kept_removed_from_model.append(rel)


def _write(folder: str, rel: str, content: bytes) -> None:
    path = os.path.join(folder, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open_plain_write(path, "wb", root=folder) as handle:
        handle.write(content)


def _load_record(folder: str) -> dict | None:
    """The record as ``{"files": {path: bytes}, "conflicted": [path]}``, or ``None`` if the folder has none."""
    path = os.path.join(folder, RECORD_FILENAME)
    if not os.path.lexists(path):
        return None
    if not is_plain_entry(path):
        raise RegenerationError(f"{path} is not a regular file, so it was not used as the record.")
    try:
        data = json.loads(read_plain_text(path, root=folder, encoding="utf-8") or "")
        files, conflicted = data["files"], data.get("conflicted", [])
        if data["format"] != _RECORD_FORMAT or not isinstance(files, dict) or not isinstance(conflicted, list):
            raise ValueError("unexpected content")
        if not all(isinstance(rel, str) and _inside(rel) for rel in [*files, *conflicted]):
            raise ValueError("a path outside the folder")
        return {"files": {rel: text.encode("utf-8") for rel, text in files.items()}, "conflicted": conflicted}
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise RegenerationError(f"The record {path} cannot be used ({type(exc).__name__}: {exc}). Restore it from "
                                "version control, or remove it and adopt the folder.") from exc


_DRIVE = re.compile(r"^[A-Za-z]:")


def _inside(rel: str) -> bool:
    """A path that stays inside the folder, whichever platform reads it."""
    normal = posixpath.normpath(rel)
    return not (normal.startswith(("/", "../")) or normal == ".." or "\\" in rel or _DRIVE.match(rel))


def _save_record(folder: str, new: dict[str, bytes], conflicted: list[str]) -> None:
    # Last, so a crash above leaves the previous base and the next run merges again.
    record = {"format": _RECORD_FORMAT, "files": {rel: lf(c).decode("utf-8") for rel, c in new.items()},
              "conflicted": conflicted}
    text = json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False)
    write_atomic_plain(os.path.join(folder, RECORD_FILENAME), text, root=folder, newline="")


def _log(folder: str, report: RegenerationReport) -> None:
    if report.merged:
        logger.warning("Merged the model's changes into edited files in %s: %s. A clean merge can still break, "
                       "so run your tests.", folder, ", ".join(report.merged))
    if report.conflicted:
        logger.warning("Conflicts to resolve in %s: %s", folder, ", ".join(report.conflicted))
    for paths, what in ((report.kept_removed_from_model, "edited, and the model no longer produces them: kept"),
                        (report.not_restored, "deleted, and the model changed them: left deleted"),
                        (report.not_merged, "clash with the model but are not UTF-8 text: left as they were")):
        if paths:
            logger.warning("Files in %s %s: %s", folder, what, ", ".join(paths))
