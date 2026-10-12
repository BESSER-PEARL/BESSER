"""Line-level authorship of a run's output, emitted as an Agent Trace record.

A Spec-Driven Agent run delivers code that is partly deterministic scaffold and
partly written by the LLM. This module tracks, for every line of every text
file in the workspace, who wrote it, and turns that into an
`Agent Trace <https://agent-trace.dev/>`_ record (``.agent-trace/<run id>.json``)
that ships with the download and the GitHub push.

Attribution is kept in ``.besser_attribution.json`` at the workspace root: one
short hash per line plus the contributor of that line. It is updated at three
points, and never by tracking individual edits:

* a BESSER generator writes files: those files are attributed to the generator
  as a whole (:func:`record_generated_files`);
* a run is seeded from a previous run or a GitHub import: lines that no longer
  match the stored state were edited by a person in between
  (:func:`seed_attribution`);
* the run is packaged: lines that differ from the stored state were written or
  modified by the LLM (:func:`reconcile_workspace`).

Each update diffs the file's current lines against its stored lines, so a line
keeps its contributor however often the file was edited, as long as the line
itself is unchanged. The state file holds only hashes and contributor labels,
never file content, so it can ride along with the GitHub push (a later
continue-from-GitHub run reads it back to detect human edits).
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

ATTRIBUTION_FILENAME = ".besser_attribution.json"
TRACE_DIRNAME = ".agent-trace"
AGENT_TRACE_VERSION = "0.1.0"
AGENT_TRACE_MIME_TYPE = "application/vnd.agent-trace.record+json"
TOOL_NAME = "BESSER Spec-Driven Agent"
METADATA_KEY = "org.besser-pearl"

_STATE_VERSION = 1
_HASH_LEN = 12
# Files above this size, and files that are not UTF-8 text, are not attributed:
# a line-level trace of a minified bundle or a binary says nothing useful.
_MAX_FILE_BYTES = 2 * 1024 * 1024
# Never attributed: the trace itself, the provenance file, run internals.
_ALWAYS_SKIPPED_DIRS = {TRACE_DIRNAME, ".git"}
_ALWAYS_SKIPPED_FILES = {"BESSER_GENERATION.md"}

# Contributor kinds, recorded in the state file (``kind``) and mapped onto the
# Agent Trace contributor types when the record is built.
KIND_GENERATOR = "generator"
KIND_AI = "ai"
KIND_HUMAN = "human"
KIND_UNKNOWN = "unknown"


def generator_contributor(name: str, besser_version: Optional[str] = None) -> dict:
    """A deterministic BESSER generator (``unknown`` in the Agent Trace record)."""
    contributor = {"kind": KIND_GENERATOR, "generator": str(name)}
    if besser_version:
        contributor["besser_version"] = str(besser_version)
    return contributor


def ai_contributor(model_id: Optional[str], run_id: Optional[str] = None) -> dict:
    """The LLM of one run."""
    contributor: dict[str, Any] = {"kind": KIND_AI}
    if model_id:
        contributor["model_id"] = str(model_id)[:250]
    if run_id:
        contributor["run_id"] = str(run_id)
    return contributor


def human_contributor() -> dict:
    """A person who edited the code between two runs."""
    return {"kind": KIND_HUMAN}


def unknown_contributor() -> dict:
    """Code that was already there, with no attribution on record."""
    return {"kind": KIND_UNKNOWN}


def normalize_model_id(provider: Optional[str], model: Optional[str]) -> Optional[str]:
    """``provider/model-name`` (the models.dev convention Agent Trace uses).

    A model id that already names its provider (``openai/gpt-4o``, an
    OpenRouter-style id) is kept as is.
    """
    model = (model or "").strip()
    if not model:
        return None
    if "/" in model:
        return model
    provider = (provider or "").strip().lower()
    if provider in ("", "free", "sponsored", "pia", "local"):
        return model
    return f"{provider}/{model}"


def _line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()[:_HASH_LEN]


def _read_lines(path: str) -> Optional[list[str]]:
    """The file's lines without line endings, or None when it is not attributable."""
    try:
        if os.path.islink(path) or not os.path.isfile(path):
            return None
        if os.path.getsize(path) > _MAX_FILE_BYTES:
            return None
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text.splitlines()


def _content_hash(lines: list[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


class AttributionState:
    """Per-line contributors for the files of one workspace."""

    def __init__(self) -> None:
        self.contributors: list[dict] = []
        # rel path -> (line hashes, contributor index per line)
        self.files: dict[str, tuple[list[str], list[int]]] = {}

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, workspace: str) -> Optional["AttributionState"]:
        """The workspace's stored state, or None when absent or unreadable."""
        path = os.path.join(workspace, ATTRIBUTION_FILENAME)
        if os.path.islink(path) or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            state = cls()
            contributors = data["contributors"]
            if not isinstance(contributors, list) or not all(
                isinstance(c, dict) and c.get("kind") for c in contributors
            ):
                return None
            state.contributors = contributors
            for rel, entry in data["files"].items():
                hashes_blob = entry["hashes"]
                hashes = [
                    hashes_blob[i:i + _HASH_LEN]
                    for i in range(0, len(hashes_blob), _HASH_LEN)
                ]
                owners: list[int] = []
                for index, count in entry["runs"]:
                    if not 0 <= int(index) < len(contributors):
                        return None
                    owners.extend([int(index)] * int(count))
                if len(owners) != len(hashes):
                    return None
                state.files[str(rel)] = (hashes, owners)
            return state
        except Exception:
            logger.debug("Unreadable attribution state at %s", path, exc_info=True)
            return None

    def save(self, workspace: str) -> None:
        files: dict[str, dict] = {}
        for rel in sorted(self.files):
            hashes, owners = self.files[rel]
            files[rel] = {"hashes": "".join(hashes), "runs": _run_lengths(owners)}
        used = sorted({i for _, owners in self.files.values() for i in owners})
        # Drop contributors no line refers to any more, renumbering the runs.
        remap = {old: new for new, old in enumerate(used)}
        for entry in files.values():
            entry["runs"] = [[remap[i], n] for i, n in entry["runs"]]
        payload = {
            "version": _STATE_VERSION,
            "contributors": [self.contributors[i] for i in used],
            "files": files,
        }
        path = os.path.join(workspace, ATTRIBUTION_FILENAME)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, path)

    # -- updates -----------------------------------------------------------

    def contributor_index(self, contributor: dict) -> int:
        for index, existing in enumerate(self.contributors):
            if existing == contributor:
                return index
        self.contributors.append(dict(contributor))
        return len(self.contributors) - 1

    def set_file(self, rel: str, lines: list[str], contributor: dict) -> None:
        """Attribute every line of ``rel`` to ``contributor``."""
        index = self.contributor_index(contributor)
        self.files[rel] = ([_line_hash(line) for line in lines], [index] * len(lines))

    def reconcile_file(self, rel: str, lines: list[str], contributor: dict) -> int:
        """Keep the contributor of unchanged lines; give the rest to ``contributor``.

        Returns the number of lines newly attributed to ``contributor``.
        """
        new_hashes = [_line_hash(line) for line in lines]
        old = self.files.get(rel)
        index = self.contributor_index(contributor)
        if old is None:
            self.files[rel] = (new_hashes, [index] * len(new_hashes))
            return len(new_hashes)
        old_hashes, old_owners = old
        if old_hashes == new_hashes:
            return 0
        owners = [index] * len(new_hashes)
        kept = 0
        matcher = difflib.SequenceMatcher(None, old_hashes, new_hashes, autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                owners[block.b + offset] = old_owners[block.a + offset]
            kept += block.size
        self.files[rel] = (new_hashes, owners)
        return len(new_hashes) - kept


def _run_lengths(owners: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for owner in owners:
        if runs and runs[-1][0] == owner:
            runs[-1][1] += 1
        else:
            runs.append([owner, 1])
    return runs


def _skipped_file(name: str, file_globs: list[str]) -> bool:
    return (
        name.startswith(".besser_")
        or name in _ALWAYS_SKIPPED_FILES
        or any(fnmatch.fnmatch(name, pattern) for pattern in file_globs)
    )


def iter_attributable_files(
    workspace: str, excluded: Optional[Iterable[str]] = None
) -> Iterable[str]:
    """Workspace-relative paths (``/``-separated) of the files a trace covers.

    ``excluded`` holds directory names and file globs, the same set the
    packaging walk uses; internals, the trace directory and the provenance
    file are always skipped.
    """
    excluded_set = set(excluded or ()) | _ALWAYS_SKIPPED_DIRS
    file_globs = [p for p in excluded_set if "*" in p or "?" in p]
    for root, dirs, names in os.walk(workspace):
        dirs[:] = sorted(
            d for d in dirs
            if d not in excluded_set
            and not d.startswith(".besser_")
            and not os.path.islink(os.path.join(root, d))
        )
        for name in sorted(names):
            if _skipped_file(name, file_globs):
                continue
            yield os.path.relpath(os.path.join(root, name), workspace).replace(os.sep, "/")


def _load_or_new(workspace: str) -> AttributionState:
    return AttributionState.load(workspace) or AttributionState()


def record_generated_files(
    workspace: str,
    rel_paths: Iterable[str],
    generator: str,
    besser_version: Optional[str] = None,
) -> None:
    """Attribute the files a BESSER generator just wrote to that generator.

    Best-effort: attribution never breaks a generator call.
    """
    try:
        state = _load_or_new(workspace)
        contributor = generator_contributor(generator, besser_version)
        for rel in rel_paths:
            rel = str(rel).replace("\\", "/")
            if rel.split("/", 1)[0] in _ALWAYS_SKIPPED_DIRS:
                continue
            lines = _read_lines(os.path.join(workspace, rel))
            if lines is None:
                state.files.pop(rel, None)
            else:
                state.set_file(rel, lines, contributor)
        state.save(workspace)
    except Exception:
        logger.debug("Recording generator attribution failed", exc_info=True)


def reconcile_workspace(
    workspace: str,
    contributor: dict,
    excluded: Optional[Iterable[str]] = None,
    paths: Optional[Iterable[str]] = None,
) -> AttributionState:
    """Give every changed or new line in the workspace to ``contributor``.

    With ``paths``, only those files are reconciled. Files that are gone (or
    no longer attributable) are dropped from the state. Saves and returns the
    state.
    """
    state = _load_or_new(workspace)
    if paths is None:
        current = set(iter_attributable_files(workspace, excluded))
        for rel in list(state.files):
            if rel not in current:
                del state.files[rel]
        targets: Iterable[str] = sorted(current)
    else:
        targets = sorted({str(p).replace("\\", "/") for p in paths})
    for rel in targets:
        lines = _read_lines(os.path.join(workspace, rel))
        if lines is None:
            state.files.pop(rel, None)
            continue
        state.reconcile_file(rel, lines, contributor)
    state.save(workspace)
    return state


def seed_attribution(workspace: str, excluded: Optional[Iterable[str]] = None) -> None:
    """Bring a freshly seeded workspace's attribution up to date.

    The seed carries the previous run's state when it has one (a run seeded
    from an earlier run, or a GitHub repo a run was pushed to): any line that
    no longer matches it was edited by a person in between. A seed without a
    state (a repo pushed before attribution existed) has no record of who
    wrote anything, so its lines are ``unknown``.
    """
    try:
        has_state = AttributionState.load(workspace) is not None
        reconcile_workspace(
            workspace,
            human_contributor() if has_state else unknown_contributor(),
            excluded,
        )
    except Exception:
        logger.debug("Seeding attribution failed", exc_info=True)


# ---------------------------------------------------------------------
# Agent Trace record
# ---------------------------------------------------------------------


def _trace_contributor(contributor: dict) -> dict:
    kind = contributor.get("kind")
    if kind == KIND_AI:
        out = {"type": "ai"}
        if contributor.get("model_id"):
            out["model_id"] = contributor["model_id"]
        return out
    if kind == KIND_HUMAN:
        return {"type": "human"}
    return {"type": "unknown"}


def _trace_related(contributor: dict) -> list[dict]:
    related = []
    if contributor.get("run_id"):
        related.append({"type": "session", "url": f"urn:besser:run:{contributor['run_id']}"})
    if contributor.get("kind") == KIND_GENERATOR:
        ref = contributor.get("generator", "unknown")
        if contributor.get("besser_version"):
            ref = f"{ref}@{contributor['besser_version']}"
        related.append({"type": "generator", "url": f"urn:besser:generator:{ref}"})
    return related


def build_trace_record(
    workspace: str,
    state: AttributionState,
    *,
    run_id: str,
    besser_version: Optional[str] = None,
    base_generator: Optional[str] = None,
    model_id: Optional[str] = None,
    record_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """The Agent Trace record for the workspace's current files.

    Ranges are maximal runs of lines with the same contributor, grouped into
    one conversation per contributor, and refer to the files as they are now.
    BESSER-specific detail (generator names, run ids, per-kind line counts)
    goes in ``metadata["org.besser-pearl"]``.
    """
    files: list[dict] = []
    totals = {KIND_GENERATOR: 0, KIND_AI: 0, KIND_HUMAN: 0, KIND_UNKNOWN: 0}
    for rel in sorted(state.files):
        lines = _read_lines(os.path.join(workspace, rel))
        hashes, owners = state.files[rel]
        if lines is None or len(lines) != len(owners) or not owners:
            continue
        ranges_by_owner: dict[int, list[dict]] = {}
        start = 0
        for i in range(1, len(owners) + 1):
            if i == len(owners) or owners[i] != owners[start]:
                ranges_by_owner.setdefault(owners[start], []).append({
                    "start_line": start + 1,
                    "end_line": i,
                    "content_hash": _content_hash(lines[start:i]),
                })
                start = i
        conversations = []
        for owner in sorted(ranges_by_owner, key=lambda o: ranges_by_owner[o][0]["start_line"]):
            contributor = state.contributors[owner]
            conversation: dict[str, Any] = {
                "contributor": _trace_contributor(contributor),
                "ranges": ranges_by_owner[owner],
            }
            related = _trace_related(contributor)
            if related:
                conversation["related"] = related
            conversations.append(conversation)
        for owner in owners:
            kind = state.contributors[owner].get("kind", KIND_UNKNOWN)
            totals[kind if kind in totals else KIND_UNKNOWN] += 1
        files.append({"path": rel, "conversations": conversations})

    tool: dict[str, str] = {"name": TOOL_NAME}
    if besser_version:
        tool["version"] = besser_version
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "lines": totals,
    }
    if base_generator:
        metadata["base_generator"] = base_generator
    if model_id:
        metadata["model_id"] = model_id
    return {
        "version": AGENT_TRACE_VERSION,
        "id": record_id or str(uuid.uuid4()),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": tool,
        "files": files,
        "metadata": {METADATA_KEY: metadata},
    }


def write_trace_record(workspace: str, record: dict, run_id: str) -> str:
    """Write ``.agent-trace/<run id>.json``; returns its path."""
    directory = os.path.join(workspace, TRACE_DIRNAME)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return path


def ai_share(state: AttributionState, run_id: Optional[str] = None, max_files: int = 200) -> dict:
    """AI-written share of the final lines, overall and per file.

    ``aiLines`` counts lines attributed to any run's LLM; ``aiLinesThisRun``
    only those of ``run_id``. ``files`` lists files with AI lines first (most
    AI lines first), capped at ``max_files``.
    """
    ai_indexes = {i for i, c in enumerate(state.contributors) if c.get("kind") == KIND_AI}
    this_run = {
        i for i in ai_indexes
        if run_id is not None and state.contributors[i].get("run_id") == run_id
    }
    per_file = []
    total_lines = total_ai = total_this_run = 0
    for rel in sorted(state.files):
        owners = state.files[rel][1]
        ai_lines = sum(1 for o in owners if o in ai_indexes)
        this_lines = sum(1 for o in owners if o in this_run)
        total_lines += len(owners)
        total_ai += ai_lines
        total_this_run += this_lines
        per_file.append({
            "path": rel,
            "aiLines": ai_lines,
            "totalLines": len(owners),
            "aiPct": round(100.0 * ai_lines / len(owners), 1) if owners else 0.0,
        })
    per_file.sort(key=lambda f: (-f["aiLines"], f["path"]))
    return {
        "aiLines": total_ai,
        "aiLinesThisRun": total_this_run,
        "totalLines": total_lines,
        "aiPct": round(100.0 * total_ai / total_lines, 1) if total_lines else 0.0,
        "files": per_file[:max_files],
        "fileCount": len(per_file),
    }
