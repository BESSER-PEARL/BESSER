"""Regenerate the generated files nobody edited, before a modify run's LLM loop (#659).

A modify run used to leave every model change to the LLM, including the files
the generator wrote and nobody touched since. Those are now regenerated from the
new model: a file whose lines are all still the generator's, unchanged, is
replaced; a file the new model adds is written; a file it no longer produces is
deleted. "Unchanged" comes from the line attribution (``.besser_attribution.json``),
which tracks generator, AI and human lines across runs and GitHub round trips.

An edited file is kept as it is. The agent gets, per file, where the new
generator output differs from the generator's lines still in the file, so it
applies the same change by hand. Blocks where the file holds only someone
else's lines are that person's choice and are left out.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.pipeline.constants import _APPLICATION_OUTPUT_DIRS
from besser.spec_driven_agent.state.agent_trace import (
    KIND_GENERATOR,
    AttributionState,
    AttributionStateError,
    iter_attributable_files,
    line_hashes,
    read_text_lines,
    record_generated_files,
)
from besser.utilities.provenance import get_besser_version
from besser.utilities.workspace_fs import open_plain_write, read_plain_text, walk_plain

logger = logging.getLogger(__name__)

# The generators whose output was measured: on 44 FastAPI runs, 70% of the generated
# files were untouched and regenerating them kept every app working (#659). The
# web app's React output has no determinism test yet, so it is not regenerated.
REGENERABLE_GENERATORS = frozenset({"generate_fastapi_backend"})

# Lines of one file's change shown to the agent; past that it gets the count.
_MAX_NOTE_LINES = 24

# Stands for "changed since attribution was recorded" when aligning a file's owners.
_CHANGED = {"kind": "changed"}


@dataclass
class FileUpdate:
    """An edited generated file the new model changes: what to apply by hand."""

    path: str
    added: list[tuple[str, str]] = field(default_factory=list)  # (line, generated line before it)
    removed: list[str] = field(default_factory=list)
    no_longer_generated: bool = False


@dataclass
class Regeneration:
    """What regenerating one generator's untouched files changed."""

    generator: str
    replaced: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    # Untouched and no longer generated, but another file still refers to it.
    kept_in_use: list[str] = field(default_factory=list)
    # Generated before, deleted since, still produced by the new model: not brought back.
    not_restored: list[str] = field(default_factory=list)
    # Not text attribution can read (binary, too large, a link): left alone.
    not_checked: list[str] = field(default_factory=list)
    to_update: list[FileUpdate] = field(default_factory=list)

    @property
    def written(self) -> list[str]:
        return self.replaced + self.added

    def note(self) -> str:
        """The agent's view of the regeneration, for Phase 2."""
        lines = []
        done = [f"{label} {_paths(paths)}" for label, paths in (
            ("replaced", self.replaced), ("added", self.added), ("deleted", self.removed)) if paths]
        if done:
            lines.append(
                "BESSER regenerated the generated files nobody had edited from the new model: "
                + "; ".join(done) + ". They already follow the model change: do not edit them for it.")
        if self.to_update:
            lines.append(
                "These generated files were edited earlier, so they were kept as they are. Apply the "
                "model change to them yourself (leave out a line that was removed on purpose):")
            lines.extend(_render_update(update) for update in self.to_update)
        if self.kept_in_use:
            lines.append(
                "Still on disk although the new model no longer produces them, because another file "
                f"refers to them: {_paths(self.kept_in_use)}. Update that file, then delete these.")
        if self.not_restored:
            lines.append(
                f"Deleted earlier and not brought back, though the new model still produces them: "
                f"{_paths(self.not_restored)}.")
        if self.not_checked:
            lines.append(f"Not checked (binary, too large or a link): {_paths(self.not_checked)}.")
        return "\n".join(lines)


def regenerate_untouched(
    workspace: str, generator: str, domain_model
) -> tuple[Regeneration | None, str]:
    """Regenerate ``generator``'s untouched files in ``workspace`` from ``domain_model``.

    Returns the regeneration, or None and the reason it did not happen.
    """
    if generator not in REGENERABLE_GENERATORS:
        return None, f"{generator} output is not regenerated"
    try:
        state = AttributionState.load(workspace)
    except AttributionStateError as exc:
        return None, f"the line attribution cannot be trusted ({exc})"
    if state is None:
        return None, "the app has no line attribution, so edited files cannot be told apart"
    out_dir = _APPLICATION_OUTPUT_DIRS[generator]
    new_files, error = _generate(generator, domain_model, out_dir)
    if new_files is None:
        return None, error

    # Only this generator's own lines: packaging steps (the secret scrub, the
    # frontend repair) are recorded as generator contributors too.
    owner = {i for i, c in enumerate(state.contributors)
             if c.get("kind") == KIND_GENERATOR and c.get("generator") == out_dir}
    old_generated = {rel for rel, (_, owners) in state.files.items()
                     if rel.startswith(out_dir + "/") and owner.intersection(owners)}
    previous = _previous_files(workspace)

    result = Regeneration(generator=generator)
    gone_untouched: list[str] = []
    try:
        for rel in sorted(set(new_files) | old_generated):
            _regenerate_file(workspace, rel, new_files.get(rel), state, owner, previous,
                             result, gone_untouched)
        for rel in gone_untouched:
            if _referenced(rel, workspace, skip=set(gone_untouched)):
                result.kept_in_use.append(rel)
            else:
                os.remove(os.path.join(workspace, rel))
                result.removed.append(rel)
    finally:
        if result.written or result.removed:
            record_generated_files(workspace, result.written + result.removed, out_dir, get_besser_version())
    if result.not_checked:
        logger.warning("Regeneration left files it cannot read alone: %s", ", ".join(result.not_checked))
    return result, ""


def _regenerate_file(workspace, rel, new_bytes, state, owner, previous, result, gone_untouched) -> None:
    current = read_text_lines(workspace, rel)
    new = new_bytes.decode("utf-8").splitlines() if new_bytes is not None else None
    if current is None:
        if os.path.lexists(os.path.join(workspace, rel)):
            result.not_checked.append(rel)
        elif new_bytes is not None:
            if rel in previous:
                result.not_restored.append(rel)
            elif _write(workspace, rel, new_bytes):
                result.added.append(rel)
            else:
                result.not_checked.append(rel)
        return
    if new == current:
        return
    current_owners = _owners_now(state, rel, current)
    if current_owners and all(o in owner for o in current_owners):
        if new is None:
            gone_untouched.append(rel)
        elif _write(workspace, rel, new_bytes):
            result.replaced.append(rel)
        else:
            result.not_checked.append(rel)
        return
    update = (FileUpdate(path=rel, no_longer_generated=True) if new is None
              else _file_update(rel, current, current_owners, owner, new))
    if update.no_longer_generated or update.added or update.removed:
        result.to_update.append(update)


def _owners_now(state: AttributionState, rel: str, current: list[str]) -> list[int]:
    """The contributor of each current line: a line changed since attribution was
    recorded gets ``_CHANGED`` (harness runs never record the AI's edits)."""
    view = AttributionState()
    view.contributors = list(state.contributors)
    if rel in state.files:
        view.files[rel] = state.files[rel]
    view.reconcile_file(rel, current, _CHANGED)
    return view.files[rel][1]


def _file_update(rel: str, current: list[str], owners: list[int], owner: set[int], new: list[str]) -> FileUpdate:
    """Where the new generator output differs from the generator's lines in an edited file.

    The file is aligned with the new output line by line (repeated template lines
    stay in place). A differing block counts only if the file side holds generator
    lines, or holds nothing: a block of someone else's lines is their choice.
    """
    update = FileUpdate(path=rel)
    matcher = difflib.SequenceMatcher(None, line_hashes(current), line_hashes(new), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        ours = [line for line, o in zip(current[i1:i2], owners[i1:i2]) if o in owner and line.strip()]
        if i2 > i1 and not ours:
            continue
        update.removed.extend(ours)
        anchor = next((line for line in reversed(new[:j1]) if line.strip()), "")
        update.added.extend((line, anchor) for line in new[j1:j2] if line.strip())
    return update


def _previous_files(workspace: str) -> set[str]:
    """The files the previous run delivered, from its recipe."""
    text = read_plain_text(os.path.join(workspace, ".besser_recipe.json"), root=workspace)
    if text is None:
        logger.warning("No seed recipe: a deleted generated file may be regenerated")
        return set()
    try:
        entries = json.loads(text).get("output_files") or []
    except (ValueError, AttributeError):
        logger.warning("Seed recipe unreadable: a deleted generated file may be regenerated")
        return set()
    return {e["path"] for e in entries if isinstance(e, dict) and isinstance(e.get("path"), str)}


def _generate(generator: str, domain_model, out_dir: str) -> tuple[dict[str, bytes] | None, str]:
    """Run the generator into a scratch folder, exactly as Phase 1 does: each text file's
    bytes by workspace path."""
    work = tempfile.mkdtemp(prefix="besser_llm_regen_")
    try:
        reply = json.loads(ToolExecutor(work, domain_model=domain_model).execute(generator, {}))
        if reply.get("error"):
            return None, f"the generator failed: {str(reply['error'])[:200]}"
        files: dict[str, bytes] = {}
        for root, dirs, names in walk_plain(os.path.join(work, out_dir)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in names:
                rel = os.path.relpath(os.path.join(root, name), work).replace(os.sep, "/")
                if read_text_lines(work, rel) is not None:
                    # Bytes: the generator mixes CRLF and LF files on Windows; keep each as written.
                    with open(os.path.join(work, rel), "rb") as handle:
                        files[rel] = handle.read()
        return files, ""
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _write(workspace: str, rel: str, content: bytes) -> bool:
    """Write a generated file; False when its folder would leave the workspace (a link)."""
    path = os.path.join(workspace, rel)
    parent = os.path.dirname(path)
    root = os.path.realpath(workspace)
    if os.path.commonpath([root, os.path.realpath(parent)]) != root:
        logger.warning("Regeneration did not write %s: its folder leads outside the app", rel)
        return False
    os.makedirs(parent, exist_ok=True)
    with open_plain_write(path, "wb", root=workspace) as handle:
        handle.write(content)
    return True


def _referenced(rel: str, workspace: str, skip: set[str]) -> bool:
    """Whether another file of the app names ``rel``'s module."""
    pattern = re.compile(rf"\b{re.escape(os.path.splitext(os.path.basename(rel))[0])}\b")
    for other in iter_attributable_files(workspace, ("node_modules", "__pycache__")):
        if other == rel or other in skip:
            continue
        lines = read_text_lines(workspace, other) or []
        if any(pattern.search(line) for line in lines):
            return True
    return False


def _paths(paths: list[str]) -> str:
    return ", ".join(f"`{p}`" for p in paths)


def _render_update(update: FileUpdate) -> str:
    if update.no_longer_generated:
        return (f"- `{update.path}`: the new model no longer produces this file. Move what it still "
                "needs elsewhere, then delete it.")
    size = len(update.added) + len(update.removed)
    if size > _MAX_NOTE_LINES:
        return (f"- `{update.path}`: {len(update.added)} lines added and {len(update.removed)} removed "
                "in the generator's version; apply the model change above to this file.")
    parts = [f"- `{update.path}`:"]
    anchor = None
    for line, after in update.added:
        if after != anchor:
            parts.append(f"  after `{after.strip()}` add:" if after else "  at the top add:")
            anchor = after
        parts.append(f"    {line}")
    if update.removed:
        parts.append("  the generator no longer writes:")
        parts.extend(f"    {line}" for line in update.removed)
    return "\n".join(parts)
