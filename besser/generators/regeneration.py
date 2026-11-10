"""Bring a generated folder up to a changed model without losing edits.

``generator.regenerate()`` runs the generator in a scratch folder and merges the
result into the output folder three ways: what it generated last time (the
record file in the folder), the folder as it is now, and the new output. An edit
is either kept or flagged, never overwritten.

``record_output()`` generates and writes that record without merging, so a folder
it wrote regenerates from the start; it needs no pygit2. ``preview_output()`` and
``preview_regeneration()`` say what the two would change, and change nothing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from besser.utilities.provenance import ATTRIBUTION_FILENAME, GENERATOR_OUTPUT_FILENAME
from besser.utilities.three_way_merge import lf, merge_available, merge_trees, same_text, with_endings_of
from besser.utilities.workspace_fs import (
    is_link,
    is_plain_entry,
    is_plain_file,
    is_plain_relpath,
    is_text,
    open_plain,
    read_plain_text,
    walk_plain,
    write_atomic_plain,
    write_atomic_plain_bytes,
)

if TYPE_CHECKING:
    from besser.generators.generator_interface import GeneratorInterface

logger = logging.getLogger(__name__)

RECORD_FILENAME = ".besser_generated.json"
_RECORD_FORMAT = 1
_YOUR_EDIT = b"<<<<<<< your edit"
_GENERATED = b">>>>>>> generated from your model"
_UNRESOLVED = re.compile(rb"^" + re.escape(_YOUR_EDIT) + rb"\r?$", re.MULTILINE)
_PATHS_NAMED = 3  # in the refusal of a folder without a record
_NOT_OUTPUT = frozenset({".git", "__pycache__", "node_modules"})
_DEFAULT_MODE = 0o644
# Scratch folders: the generator's output, and the copy a preview merges into. The editor
# backend's hourly cleanup removes stale ``besser_*`` folders.
_SCRATCH_PREFIX = "besser_regenerate_"
# What a preview says of each file it would write.
NEW, CHANGED, UNCHANGED = "new", "changed", "unchanged"
# A Spec-Driven app keeps its own base and line authorship. A merge here would update neither: the
# next modify run would read these lines as a person's edits.
_SPEC_DRIVEN_RECORDS = (GENERATOR_OUTPUT_FILENAME, ATTRIBUTION_FILENAME)


class RegenerationError(Exception):
    """The folder cannot be regenerated safely. Raised before any generated file is written."""

    def _naming(self, copy: str, folder: str) -> RegenerationError:
        """The same refusal, naming ``folder`` where it names ``copy`` (the scratch copy a preview merges into)."""
        return type(self)(str(self).replace(copy, folder))


class UnrecordedOutputError(RegenerationError):
    """The folder holds generated files and no record of them: it has to be adopted first.

    Attributes:
        folder: The output folder.
        files: The generated files already in it, relative to it with ``/``, sorted.
    """

    def __init__(self, folder: str, files: list[str]):
        self.folder, self.files = folder, list(files)
        super().__init__(f"{folder} holds generated files ({self.named_files()}) but no {RECORD_FILENAME}, so edits "
                         "cannot be told from generated code. Run regenerate(adopt=True) once on a generator holding "
                         "the model those files came from.")

    def named_files(self) -> str:
        """The first few of ``files``, comma-separated, with ``...`` when there are more."""
        more = ", ..." if len(self.files) > _PATHS_NAMED else ""
        return ", ".join(self.files[:_PATHS_NAMED]) + more

    def _naming(self, copy: str, folder: str) -> UnrecordedOutputError:
        return type(self)(self.folder.replace(copy, folder), self.files)


class ConflictsPendingError(RegenerationError):
    """Files still hold the conflict markers the last regeneration wrote.

    Attributes:
        folder: The output folder.
        files: The files with markers, relative to it with ``/``, sorted.
    """

    def __init__(self, folder: str, files: list[str]):
        self.folder, self.files = folder, list(files)
        super().__init__(f"Resolve the conflict markers in {', '.join(self.files)} first.")

    def _naming(self, copy: str, folder: str) -> ConflictsPendingError:
        return type(self)(self.folder.replace(copy, folder), self.files)


class AlreadyRecordedError(RegenerationError):
    """``adopt`` on a folder that already has its record.

    Attributes:
        folder: The output folder.
    """

    def __init__(self, folder: str):
        self.folder = folder
        super().__init__(f"{folder} already has {RECORD_FILENAME}: there is nothing to adopt.")

    def _naming(self, copy: str, folder: str) -> AlreadyRecordedError:
        return type(self)(self.folder.replace(copy, folder))


class NotRecordableError(RegenerationError):
    """The generator, with its options, or the folder cannot take a record; ``generate()`` still can."""


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
    #: Files the model no longer produces and that may hold your work, left in place: edited
    #: text, any binary file, and an old name when both case variants of a rename exist.
    kept_removed_from_model: list[str] = field(default_factory=list)
    #: Files someone deleted and the model changed, left deleted.
    not_restored: list[str] = field(default_factory=list)
    #: Files that clash with the model but cannot be merged as text, left as they were.
    not_merged: list[str] = field(default_factory=list)
    #: On adopting: files that differ from this model's output. These read as edits from
    #: now on, so check that this is the model the folder came from.
    differs_from_model: list[str] = field(default_factory=list)


def regenerate_output(generator: GeneratorInterface, adopt: bool = False) -> RegenerationReport:
    """See :meth:`GeneratorInterface.regenerate`."""
    _require_support(generator)
    if not merge_available():
        raise RegenerationError('Regenerating needs pygit2: pip install "besser[regen]"')
    folder = generator.output_dir
    _make_folder(folder)
    name = type(generator).__name__
    record = _load_record(folder, name)
    _require_usable_state(folder, record, adopt)

    output, modes = generate_in_scratch(generator)
    new = {rel: content for rel, content in output.items() if is_text(content)}
    base = dict(record["files"]) if record else {}
    current = _read_files(folder, set(base) | set(new))
    new = generator._keep_generated_values(base or current, new)
    binaries = sorted(set(output) - set(new))

    def save(files: dict[str, bytes], conflicted: list[str]) -> None:
        _save_record(folder, generator._without_secrets(files), binaries, conflicted, name)

    report = RegenerationReport()
    if adopt:
        report.differs_from_model = sorted(rel for rel in new if not same_text(current.get(rel), new[rel]))
    else:
        # First, as a case rename already moves a file: through a planted link it would leave the folder.
        _require_writable_paths(folder, output)
        if record is None:
            _require_no_generated_files(folder, output)
            # A crash from here on resumes with the next run, rather than needing an adoption.
            save({}, [])
        report.kept_removed_from_model.extend(_follow_case_renames(folder, base, current, new))
        _apply(folder, base, current, new, modes, report, save)
        _write_binaries(folder, output, new, modes, report)
        report.kept_removed_from_model.extend(
            rel for rel in (record or {}).get("binaries", []) if rel not in output
            and is_plain_file(os.path.join(folder, *rel.split("/")), root=folder))
    save(new, report.conflicted)
    _log(folder, report)
    return report


def record_output(generator: GeneratorInterface) -> list[str]:
    """Generate into the output folder, overwriting, and record the output as a first ``regenerate()`` does.

    ``regenerate()`` then merges into the folder without adopting it. Recording needs no
    pygit2, only a merge does. Files the generator does not produce are left alone, and the
    record is the same bytes for the same output.

    Returns:
        The files written, sorted, relative to the output folder with ``/``.

    Raises:
        NotRecordableError: The generator or the folder cannot take a record; the generator has not run.
        RegenerationError: A folder or link is where a file goes; nothing was written.
    """
    _require_support(generator)
    folder = generator.output_dir
    _require_not_spec_driven(folder)
    output, modes = generate_in_scratch(generator)
    _require_writable_paths(folder, output)
    _make_folder(folder)
    for rel, content in sorted(output.items()):
        _write(folder, rel, content, modes.get(rel, _DEFAULT_MODE))
    write_atomic_plain(os.path.join(folder, RECORD_FILENAME), _first_record(generator, output),
                       root=folder, newline="")
    return sorted(output)


def preview_output(generator: GeneratorInterface, record: bool = False) -> dict[str, str]:
    """What generating into the output folder would write, without writing it: path -> ``NEW``, ``CHANGED``
    or ``UNCHANGED``, sorted. With ``record``, as :func:`record_output` would, record included.

    The generator runs in a scratch folder, so it must not act outside its output: one that would
    (a backend that pushes an image) is refused.

    Raises:
        NotRecordableError: With ``record``, the generator or the folder cannot take a record.
        RegenerationError: The generator would act outside its output, or, with ``record``,
            a folder or link is where a file goes.
    """
    refusal = generator._regeneration_refusal()
    if refusal:
        raise RegenerationError(f"{type(generator).__name__} cannot preview its output: {refusal}")
    folder = generator.output_dir
    if record:
        _require_support(generator)
        _require_not_spec_driven(folder)
    output, _ = generate_in_scratch(generator)
    planned = dict(output)
    if record:
        _require_writable_paths(folder, output)
        planned[RECORD_FILENAME] = _first_record(generator, output).encode("utf-8")
    return {rel: _status(folder, rel, content) for rel, content in sorted(planned.items())}


def preview_regeneration(generator: GeneratorInterface, adopt: bool = False) -> RegenerationReport:
    """What ``regenerate(adopt)`` would do, done on a scratch copy of the output folder, which is left as it is.

    Raises:
        RegenerationError: As ``regenerate()``, naming the output folder.
    """
    _require_support(generator)
    folder = generator.output_dir
    if os.path.lexists(folder) and not os.path.isdir(folder):
        raise RegenerationError(f"{folder} is not a folder.")
    scratch = tempfile.mkdtemp(prefix=_SCRATCH_PREFIX)
    copy = os.path.join(scratch, "output")
    try:
        if os.path.isdir(folder):
            # Links are copied as links, so the copy is refused where the folder would be.
            shutil.copytree(folder, copy, symlinks=True, ignore=shutil.ignore_patterns(*_NOT_OUTPUT))
        generator.output_dir = copy
        try:
            return regenerate_output(generator, adopt)
        except RegenerationError as exc:
            raise exc._naming(copy, folder) from exc
    finally:
        generator.output_dir = folder
        shutil.rmtree(scratch, ignore_errors=True)


def _status(folder: str, rel: str, content: bytes) -> str:
    path = os.path.join(folder, *rel.split("/"))
    if not os.path.lexists(path):
        return NEW
    if is_plain_file(path, root=folder):
        with open_plain(path, "rb", root=folder) as handle:
            if handle.read() == content:
                return UNCHANGED
    return CHANGED


def _make_folder(folder: str) -> None:
    """Create the output folder if missing; a file at its path, or at a parent's, is a refusal."""
    if os.path.lexists(folder) and not os.path.isdir(folder):
        raise RegenerationError(f"{folder} is not a folder.")
    try:
        os.makedirs(folder, exist_ok=True)
    except (FileExistsError, NotADirectoryError) as exc:
        raise RegenerationError(f"Cannot create the folder {folder}: a file is where one of its folders goes.") from exc


def _require_support(generator: GeneratorInterface) -> None:
    """What a record needs: a stable output and an explicit folder. A merge also needs pygit2."""
    if not generator.regenerable:
        raise NotRecordableError(f"{type(generator).__name__} cannot regenerate: its output differs from run to run "
                                "for one model, so a merge could not tell a model change from noise.")
    refusal = generator._regeneration_refusal()
    if refusal:
        raise NotRecordableError(f"{type(generator).__name__} cannot regenerate: {refusal}")
    if generator.output_dir is None:
        raise NotRecordableError("Regenerating needs an explicit output_dir: "
                                 "it says which folder's record and edits to use.")


def _require_not_spec_driven(folder: str) -> None:
    for name in _SPEC_DRIVEN_RECORDS:
        if os.path.lexists(os.path.join(folder, name)):
            raise NotRecordableError(f"{folder} is a Spec-Driven Agent app (it has {name}): "
                                     "change its model through a modify run, which keeps its authorship record.")


def _require_usable_state(folder: str, record: dict | None, adopt: bool) -> None:
    _require_not_spec_driven(folder)
    if adopt and record:
        raise AlreadyRecordedError(folder)
    pending = []
    for rel in (record or {}).get("conflicted", []):
        path = os.path.join(folder, *rel.split("/"))
        if is_plain_file(path, root=folder):
            with open_plain(path, "rb", root=folder) as handle:
                if _UNRESOLVED.search(handle.read()):
                    pending.append(rel)
    if pending:
        raise ConflictsPendingError(folder, sorted(pending))


def _require_no_generated_files(folder: str, output: dict[str, bytes]) -> None:
    """Without a record, edits cannot be told from generated code in a file the generator also produces."""
    existing = sorted(rel for rel in output if os.path.lexists(os.path.join(folder, *rel.split("/"))))
    if existing:
        raise UnrecordedOutputError(folder, existing)


def _require_writable_paths(folder: str, output: dict[str, bytes]) -> None:
    """Fail before the first write, not halfway through: a folder or link where a file goes, or the reverse."""
    for rel in sorted(output):
        parts = rel.split("/")
        current = folder
        for part in parts[:-1]:
            current = os.path.join(current, part)
            if is_link(current) or (os.path.lexists(current) and not os.path.isdir(current)):
                raise RegenerationError(f"Cannot write {rel}: {current} is not a plain folder.")
        path = os.path.join(current, parts[-1])
        if is_link(path) or os.path.isdir(path):
            raise RegenerationError(f"Cannot write {rel}: a folder or link is already at that path.")


def generate_in_scratch(generator: GeneratorInterface) -> tuple[dict[str, bytes], dict[str, int]]:
    """Run the generator elsewhere, as its ``generate()`` may overwrite or delete the output folder.

    Returns:
        Its output and each file's mode, by path relative to the folder with ``/``.
    """
    scratch = tempfile.mkdtemp(prefix=_SCRATCH_PREFIX)
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
    for root, dirs, names in walk_plain(top):
        dirs[:] = [d for d in dirs if d not in _NOT_OUTPUT]
        for name in names:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, top).replace(os.sep, "/")
            modes[rel] = stat.S_IMODE(os.stat(path).st_mode)
            with open_plain(path, "rb", root=top) as handle:
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


def _follow_case_renames(folder: str, base: dict[str, bytes], current: dict[str, bytes],
                         new: dict[str, bytes]) -> list[str]:
    """A file the model renamed in case only (``book.java`` to ``Book.java``) moves, keeping its edits.

    On a case-insensitive disk both names are one file, which a merge would read as
    one deleted and one added. ``base`` and ``current`` are updated in place. Returns
    the old names kept because a case-sensitive disk holds both: no longer generated.
    """
    kept = []
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
            kept.append(old)
            continue
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.rename(old_path, target_path)
        current[target] = current.pop(old)
    return kept


def _apply(folder: str, base: dict[str, bytes], current: dict[str, bytes], new: dict[str, bytes],
           modes: dict[str, int], report: RegenerationReport, save) -> None:
    try:
        merged, conflicts = merge_trees({rel: lf(c) for rel, c in base.items()},
                                        {rel: lf(c) for rel, c in current.items()},
                                        {rel: lf(c) for rel, c in new.items()})
    except Exception as exc:  # libgit2's own errors have no shared base class worth naming here
        raise RegenerationError(f"The merge failed ({type(exc).__name__}: {exc}); your edits are untouched.") from exc
    planned = sorted(rel for rel, content in conflicts.items() if content is not None)
    if planned:
        # Before any marker is written, so a crash below cannot merge markers into markers on the next run.
        save(base, planned)
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
        _write(folder, rel, with_endings_of(mine, _labelled(content, rel)), _DEFAULT_MODE)
        report.conflicted.append(rel)
    elif mine is None:
        report.not_restored.append(rel)
    elif rel in new:
        report.not_merged.append(rel)
    else:
        report.kept_removed_from_model.append(rel)


def _labelled(content: bytes, rel: str) -> bytes:
    """libgit2's markers say which side is which. They name the file, or nothing when that side is
    empty; any other line starting with marker characters is the user's own."""
    label = rb"(?: " + re.escape(rel.encode("utf-8")) + rb")?$"
    content = re.sub(rb"^<<<<<<<" + label, _YOUR_EDIT, content, flags=re.MULTILINE)
    return re.sub(rb"^>>>>>>>" + label, _GENERATED, content, flags=re.MULTILINE)


def _write_binaries(folder: str, output: dict[str, bytes], new: dict[str, bytes], modes: dict[str, int],
                    report: RegenerationReport) -> None:
    """Binary output is written once, and never merged or removed; one that changed since is left as it is, and reported."""
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
    """The record as ``{"files": {path: bytes}, "binaries": [path], "conflicted": [path]}``, or ``None`` if there is none."""
    path = os.path.join(folder, RECORD_FILENAME)
    if not os.path.lexists(path):
        return None
    if not is_plain_entry(path):
        raise RegenerationError(f"{path} is not a regular file, so it was not used as the record.")
    try:
        data = json.loads(read_plain_text(path, root=folder, encoding="utf-8") or "")
        files, binaries, conflicted = data["files"], data["binaries"], data["conflicted"]
        if (data["format"] != _RECORD_FORMAT or not isinstance(files, dict)
                or not isinstance(binaries, list) or not isinstance(conflicted, list)):
            raise ValueError("unexpected content")
        if not all(_inside(rel) for rel in [*files, *binaries, *conflicted]):
            raise ValueError("a path outside the folder, or in .git")
        record = {"files": {rel: text.encode("utf-8") for rel, text in files.items()},
                  "binaries": binaries, "conflicted": conflicted}
        recorded_by = data["generator"]
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise RegenerationError(f"The record {path} cannot be used ({type(exc).__name__}: {exc}). Restore it from "
                                "version control, or remove it and adopt the folder.") from exc
    if recorded_by != generator_name:
        raise RegenerationError(f"{folder} holds the output of {recorded_by}. Merging {generator_name} into it would "
                                "delete those files: give it its own output_dir.")
    return record


def _inside(rel: str) -> bool:
    """A plain path inside the folder, whichever platform reads it, and not git's own."""
    return is_plain_relpath(rel) and ".git" not in (part.casefold() for part in rel.split("/"))


def _save_record(folder: str, files: dict[str, bytes], binaries: list[str], conflicted: list[str],
                 generator_name: str) -> None:
    text = _record_text(files, binaries, conflicted, generator_name)
    write_atomic_plain(os.path.join(folder, RECORD_FILENAME), text, root=folder, newline="")


def _record_text(files: dict[str, bytes], binaries: list[str], conflicted: list[str], generator_name: str) -> str:
    record = {"format": _RECORD_FORMAT, "generator": generator_name, "binaries": binaries, "conflicted": conflicted,
              "files": {rel: lf(content).decode("utf-8") for rel, content in files.items()}}
    return json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False)


def _first_record(generator: GeneratorInterface, output: dict[str, bytes]) -> str:
    """The record a first ``regenerate()`` writes of ``output``: its text files, and the names of the rest."""
    text = {rel: content for rel, content in output.items() if is_text(content)}
    return _record_text(generator._without_secrets(text), sorted(set(output) - set(text)), [],
                        type(generator).__name__)


def _log(folder: str, report: RegenerationReport) -> None:
    if report.merged:
        logger.warning("Merged the model's changes into edited files in %s: %s. A clean merge can still break, "
                       "so run your tests.", folder, ", ".join(report.merged))
    if report.conflicted:
        logger.warning("Conflicts to resolve in %s: %s", folder, ", ".join(report.conflicted))
    for paths, what in ((report.kept_removed_from_model, "the model no longer produces, and that may hold your work: kept"),
                        (report.not_restored, "deleted, and the model changed them: left deleted"),
                        (report.not_merged, "clash with the model but cannot be merged as text: left as they were"),
                        (report.differs_from_model, "differ from this model's output, and read as edits from now on")):
        if paths:
            logger.warning("Files in %s %s: %s", folder, what, ", ".join(paths))
