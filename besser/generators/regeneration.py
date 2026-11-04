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
import stat
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from besser.utilities.three_way_merge import lf, merge_available, merge_trees, same_text, with_endings_of
from besser.utilities.workspace_fs import (
    is_plain_entry,
    is_plain_file,
    is_text,
    open_plain,
    read_plain_text,
    write_atomic_plain,
    write_atomic_plain_bytes,
)

if TYPE_CHECKING:
    from besser.generators.generator_interface import GeneratorInterface

logger = logging.getLogger(__name__)

RECORD_FILENAME = ".besser_generated.json"
_RECORD_FORMAT = 1
_CONFLICT_START = re.compile(rb"^<<<<<<< .*$", re.MULTILINE)
_CONFLICT_END = re.compile(rb"^>>>>>>> .*$", re.MULTILINE)
_DRIVE = re.compile(r"^[A-Za-z]:")
_NOT_OUTPUT = frozenset({".git", "__pycache__", "node_modules"})
_DEFAULT_MODE = 0o644
# A Spec-Driven app keeps its own base and line authorship. A merge here would update neither: the
# next modify run would read these lines as a person's edits.
_SPEC_DRIVEN_RECORDS = (".besser_generator_output.json", ".besser_attribution.json")


class RegenerationError(Exception):
    """The folder cannot be regenerated safely. Raised before any generated file is written."""


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
    #: Files that clash with the model but cannot be merged as text, left as they were.
    not_merged: list[str] = field(default_factory=list)
    #: The folder was recorded as this model's output, and no file changed.
    adopted: bool = False
    #: On adopting: files that differ from this model's output. These read as edits from
    #: now on, so check that this is the model the folder came from.
    differs_from_model: list[str] = field(default_factory=list)


def regenerate_output(generator: GeneratorInterface, adopt: bool = False) -> RegenerationReport:
    """See :meth:`GeneratorInterface.regenerate`."""
    _require_support(generator)
    folder = generator.output_dir
    os.makedirs(folder, exist_ok=True)
    name = type(generator).__name__
    record = _load_record(folder, name)
    _require_usable_state(folder, record, adopt)

    output, modes = _generate_in_scratch(generator)
    new = {rel: content for rel, content in output.items() if is_text(content)}
    base = dict(record["files"]) if record else {}
    current = _read_files(folder, set(base) | set(new))
    new = generator._keep_generated_values(base or current, new)

    report = RegenerationReport(adopted=adopt)
    if adopt:
        report.differs_from_model = sorted(rel for rel in new if not same_text(current.get(rel), new[rel]))
    else:
        if record is None:
            _require_no_generated_files(folder, output)
            # A crash from here on resumes with the next run, rather than needing an adoption.
            _save_record(folder, {}, [], name)
        _follow_case_renames(folder, base, current, new)
        _require_writable_paths(folder, output)
        _apply(folder, base, current, new, modes, report, name)
        _write_binaries(folder, output, new, modes, report)
    _save_record(folder, new, report.conflicted, name)
    _log(folder, report)
    return report


def _require_support(generator: GeneratorInterface) -> None:
    if not generator.regenerable:
        raise RegenerationError(f"{type(generator).__name__} cannot regenerate: its output differs from run to run "
                                "for one model, so a merge could not tell a model change from noise.")
    refusal = generator._regeneration_refusal()
    if refusal:
        raise RegenerationError(f"{type(generator).__name__} cannot regenerate: {refusal}")
    if generator.output_dir is None:
        raise RegenerationError("Regenerating needs an explicit output_dir: "
                                "it says which folder's record and edits to use.")
    if not merge_available():
        raise RegenerationError('Regenerating needs pygit2: pip install "besser[regen]"')


def _require_usable_state(folder: str, record: dict | None, adopt: bool) -> None:
    for name in _SPEC_DRIVEN_RECORDS:
        if os.path.lexists(os.path.join(folder, name)):
            raise RegenerationError(f"{folder} is a Spec-Driven Agent app (it has {name}): "
                                    "change its model through a modify run, which keeps its authorship record.")
    if adopt and record:
        raise RegenerationError(f"{folder} already has {RECORD_FILENAME}: there is nothing to adopt.")
    for rel in (record or {}).get("conflicted", []):
        path = os.path.join(folder, *rel.split("/"))
        if is_plain_file(path, root=folder):
            with open_plain(path, "rb", root=folder) as handle:
                if _CONFLICT_START.search(handle.read()):
                    raise RegenerationError(f"Resolve the conflict markers in {rel} first.")


def _require_no_generated_files(folder: str, output: dict[str, bytes]) -> None:
    """Without a record, edits cannot be told from generated code in a file the generator also produces."""
    existing = sorted(rel for rel in output if os.path.lexists(os.path.join(folder, *rel.split("/"))))
    if existing:
        raise RegenerationError(
            f"{folder} holds generated files ({', '.join(existing[:3])}{', ...' if len(existing) > 3 else ''}) but no "
            f"{RECORD_FILENAME}, so edits cannot be told from generated code. Run regenerate(adopt=True) once on a "
            "generator holding the model those files came from.")


def _require_writable_paths(folder: str, output: dict[str, bytes]) -> None:
    """Fail before the first write, not halfway through: a folder or link where a file goes, or the reverse."""
    for rel in sorted(output):
        parts = rel.split("/")
        current = folder
        for part in parts[:-1]:
            current = os.path.join(current, part)
            if os.path.islink(current) or (os.path.lexists(current) and not os.path.isdir(current)):
                raise RegenerationError(f"Cannot write {rel}: {current} is not a plain folder.")
        path = os.path.join(current, parts[-1])
        if os.path.islink(path) or os.path.isdir(path):
            raise RegenerationError(f"Cannot write {rel}: a folder or link is already at that path.")


def _generate_in_scratch(generator: GeneratorInterface) -> tuple[dict[str, bytes], dict[str, int]]:
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


def _read_tree(top: str) -> tuple[dict[str, bytes], dict[str, int]]:
    files, modes = {}, {}
    for root, dirs, names in os.walk(top):
        dirs[:] = [d for d in dirs if d not in _NOT_OUTPUT]
        for name in names:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, top).replace(os.sep, "/")
            modes[rel] = stat.S_IMODE(os.stat(path).st_mode)
            with open(path, "rb") as handle:
                files[rel] = handle.read()
    return files, modes


def _read_files(folder: str, paths: set[str]) -> dict[str, bytes]:
    files = {}
    for rel in paths:
        path = os.path.join(folder, *rel.split("/"))
        if is_plain_file(path, root=folder):
            with open_plain(path, "rb", root=folder) as handle:
                files[rel] = handle.read()
    return files


def _follow_case_renames(folder: str, base: dict[str, bytes], current: dict[str, bytes], new: dict[str, bytes]) -> None:
    """A file the model renamed in case only (``book.java`` to ``Book.java``) moves, keeping its edits.

    On a case-insensitive disk both names are one file, which a merge would read as
    one deleted and one added. ``base`` and ``current`` are updated in place.
    """
    by_case = {rel.casefold(): rel for rel in new}
    if len(by_case) != len(new):
        raise RegenerationError("The generator's output has paths that differ only in case.")
    for old in [rel for rel in base if rel not in new and rel.casefold() in by_case]:
        target = by_case[old.casefold()]
        base[target] = base.pop(old)
        if old not in current:
            continue
        old_path = os.path.join(folder, *old.split("/"))
        target_path = os.path.join(folder, *target.split("/"))
        if target in current and not os.path.samefile(old_path, target_path):
            continue  # a case-sensitive disk holding both: two files, and the old one is no longer generated
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.rename(old_path, target_path)
        current[target] = current.pop(old)


def _apply(folder: str, base: dict[str, bytes], current: dict[str, bytes], new: dict[str, bytes],
           modes: dict[str, int], report: RegenerationReport, name: str) -> None:
    try:
        merged, conflicts = merge_trees({rel: lf(c) for rel, c in base.items()},
                                        {rel: lf(c) for rel, c in current.items()},
                                        {rel: lf(c) for rel, c in new.items()})
    except Exception as exc:  # libgit2's own errors have no shared base class worth naming here
        raise RegenerationError(f"The merge failed ({type(exc).__name__}: {exc}); no file was changed.") from exc
    planned = sorted(rel for rel, content in conflicts.items() if content is not None)
    if planned:
        # Before any marker is written, so a crash below cannot merge markers into markers on the next run.
        _save_record(folder, base, planned, name)
    for rel in sorted(set(base) | set(current) | set(new)):
        mine = current.get(rel)
        if rel in conflicts:
            _apply_conflict(folder, rel, conflicts[rel], mine, new, report)
        elif rel not in merged:
            if mine is not None:
                os.remove(os.path.join(folder, *rel.split("/")))
                report.deleted.append(rel)
        elif mine is None or lf(mine) != merged[rel]:
            content = with_endings_of(mine if mine is not None else new[rel], merged[rel])
            _write(folder, rel, content, modes.get(rel, _DEFAULT_MODE))
            untouched = mine is None or same_text(mine, base.get(rel))
            (report.written if untouched else report.merged).append(rel)


def _apply_conflict(folder: str, rel: str, content: bytes | None, mine: bytes | None, new: dict[str, bytes],
                    report: RegenerationReport) -> None:
    if content is not None:
        labelled = _CONFLICT_START.sub(b"<<<<<<< your edit", content)
        labelled = _CONFLICT_END.sub(b">>>>>>> generated from your model", labelled)
        _write(folder, rel, with_endings_of(mine, labelled), _DEFAULT_MODE)
        report.conflicted.append(rel)
    elif mine is None:
        report.not_restored.append(rel)
    elif rel in new:
        report.not_merged.append(rel)
    else:
        report.kept_removed_from_model.append(rel)


def _write_binaries(folder: str, output: dict[str, bytes], new: dict[str, bytes], modes: dict[str, int],
                    report: RegenerationReport) -> None:
    """Binary output is written once and never merged; one that changed since is left as it is, and reported."""
    for rel, content in sorted(output.items()):
        if rel in new:
            continue
        path = os.path.join(folder, *rel.split("/"))
        if not os.path.lexists(path):
            _write(folder, rel, content, modes.get(rel, _DEFAULT_MODE))
            report.written.append(rel)
        else:
            with open_plain(path, "rb", root=folder) as handle:
                if handle.read() != content:
                    report.not_merged.append(rel)


def _write(folder: str, rel: str, content: bytes, mode: int) -> None:
    """Replace the file in one step, keeping its mode: a full disk must not leave half an edit behind."""
    path = os.path.join(folder, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        mode = stat.S_IMODE(os.stat(path).st_mode)
    write_atomic_plain_bytes(path, content, root=folder, mode=mode)


def _load_record(folder: str, generator_name: str) -> dict | None:
    """The record as ``{"files": {path: bytes}, "conflicted": [path]}``, or ``None`` if the folder has none."""
    path = os.path.join(folder, RECORD_FILENAME)
    if not os.path.lexists(path):
        return None
    if not is_plain_entry(path):
        raise RegenerationError(f"{path} is not a regular file, so it was not used as the record.")
    try:
        data = json.loads(read_plain_text(path, root=folder, encoding="utf-8") or "")
        files, conflicted = data["files"], data["conflicted"]
        if data["format"] != _RECORD_FORMAT or not isinstance(files, dict) or not isinstance(conflicted, list):
            raise ValueError("unexpected content")
        if not all(isinstance(rel, str) and _inside(rel) for rel in [*files, *conflicted]):
            raise ValueError("a path outside the folder")
        record = {"files": {rel: text.encode("utf-8") for rel, text in files.items()}, "conflicted": conflicted}
        recorded_by = data["generator"]
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise RegenerationError(f"The record {path} cannot be used ({type(exc).__name__}: {exc}). Restore it from "
                                "version control, or remove it and adopt the folder.") from exc
    if recorded_by != generator_name:
        raise RegenerationError(f"{folder} holds the output of {recorded_by}. Merging {generator_name} into it would "
                                "delete those files: give it its own output_dir.")
    return record


def _inside(rel: str) -> bool:
    """A path that stays inside the folder, whichever platform reads it."""
    normal = posixpath.normpath(rel)
    return not (normal.startswith(("/", "../")) or normal == ".." or "\\" in rel or _DRIVE.match(rel)
                or ".git" in (part.casefold() for part in normal.split("/")))


def _save_record(folder: str, files: dict[str, bytes], conflicted: list[str], generator_name: str) -> None:
    record = {"format": _RECORD_FORMAT, "generator": generator_name, "conflicted": conflicted,
              "files": {rel: lf(content).decode("utf-8") for rel, content in files.items()}}
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
                        (report.not_merged, "clash with the model but cannot be merged as text: left as they were"),
                        (report.differs_from_model, "differ from this model's output, and read as edits from now on")):
        if paths:
            logger.warning("Files in %s %s: %s", folder, what, ", ".join(paths))
