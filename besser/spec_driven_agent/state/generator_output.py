"""The generator's output as it wrote it, kept for the three-way merge of a later modify run (#659).

A modify run whose model changed regenerates the app's generated files. A file
someone edited since is merged: base = the output stored here, ours = the file
now, theirs = the generator's output for the new model. The store is written
when a regenerable generator writes the app (Phase 1) and after each
regeneration, in ``.besser_generator_output.json`` at the workspace root. It
rides with the seed of a modify run, the ZIP and the GitHub push, so a
continue-from-GitHub run merges too.

At packaging the editor backend attaches the class diagram the output was
generated from (:func:`attach_model`). The next run generates that diagram
again: with the same BESSER, to check the stored files (a store edited by hand
is not used); after an upgrade, as the merge base of edited files, so they do
not take the older generator's differences for edits.

It holds the generated code itself, which the app already contains. The
packaging scrub reads it like any other file. Every read and write goes
through :mod:`~besser.utilities.workspace_fs`, and an imported store is
checked before use: it comes from a repository anyone with write access can edit.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from besser.spec_driven_agent.state.agent_trace import is_plain_relpath, is_text
from besser.utilities.workspace_fs import open_plain, write_atomic_plain

logger = logging.getLogger(__name__)

GENERATOR_OUTPUT_FILENAME = ".besser_generator_output.json"
_FORMAT = 1
# The packaging scrub skips files over 2 MiB; a bigger store would ship unscrubbed.
MAX_STORED_BYTES = 1_500_000
# The version reaches the agent's note: only a version-shaped string is kept.
_VERSION = re.compile(r"[0-9A-Za-z.+-]{1,40}")


class GeneratorOutputError(Exception):
    """The stored generator output is malformed or does not belong to this generator."""


@dataclass(frozen=True)
class GeneratorOutput:
    """One generator's files as it wrote them, by workspace path."""

    out_dir: str
    besser_version: Optional[str]
    files: dict[str, bytes]
    # The class diagram (editor JSON) the files were generated from, once attached.
    model: Optional[dict] = None
    # Files a run left behind the model (to the agent): they keep an older base than ``model`` gives.
    pending: frozenset[str] = frozenset()


def store_generator_output(
    workspace: str, out_dir: str, files: dict[str, bytes], besser_version: Optional[str],
    pending: Iterable[str] = (),
) -> bool:
    """Store ``files`` (``out_dir``'s whole output) as the base of the next merge.

    A store that cannot be written or would be too large is removed, so a
    later merge never uses an older run's output as its base. Returns whether
    the output was stored.
    """
    try:
        payload = json.dumps({
            "format": _FORMAT,
            "besser_version": besser_version,
            "out_dir": out_dir,
            "files": {rel: files[rel].decode("utf-8") for rel in sorted(files)},
            "pending": sorted(pending),
        }, sort_keys=True)
        if len(payload.encode("utf-8")) > MAX_STORED_BYTES:
            raise ValueError(f"{len(files)} files exceed {MAX_STORED_BYTES} bytes")
        write_atomic_plain(os.path.join(workspace, GENERATOR_OUTPUT_FILENAME), payload, root=workspace)
        return True
    except (OSError, ValueError) as exc:
        logger.warning("The generator output was not stored, so the next model change is not merged: %s", exc)
        discard_generator_output(workspace)
        return False


def store_written_files(workspace: str, out_dir: str, rel_paths: Iterable[str], besser_version: Optional[str]) -> None:
    """Store the text files a generator call just wrote, read back from the workspace."""
    files = {}
    for rel in rel_paths:
        rel = str(rel).replace("\\", "/")
        try:
            with open_plain(os.path.join(workspace, rel), "rb", root=workspace) as handle:
                content = handle.read()
        except OSError as exc:
            logger.warning("The generator output was not stored: %s is unreadable (%s)", rel, exc)
            discard_generator_output(workspace)
            return
        if is_text(content):
            files[rel] = content
    store_generator_output(workspace, out_dir, files, besser_version)


def load_generator_output(workspace: str, out_dir: str) -> Optional[GeneratorOutput]:
    """The stored output of ``out_dir``'s generator; None when the app has none.

    Raises :class:`GeneratorOutputError` for a store that is too large,
    malformed, belongs to another generator, or names a path outside ``out_dir``.
    """
    data = _read_store(workspace)
    if data is None:
        return None
    if data.get("format") != _FORMAT:
        raise GeneratorOutputError("not a stored generator output")
    if data.get("out_dir") != out_dir:
        raise GeneratorOutputError(f"it holds {str(data.get('out_dir'))[:40]!r}, not {out_dir!r}")
    files = data.get("files")
    if not isinstance(files, dict) or not all(isinstance(v, str) for v in files.values()):
        raise GeneratorOutputError("its file list is malformed")
    outside = [rel for rel in files if not (is_plain_relpath(rel) and rel.startswith(out_dir + "/"))]
    if outside:
        raise GeneratorOutputError(f"it names paths outside {out_dir}/: {outside[0][:200]!r}")
    pending = data.get("pending", [])
    if not isinstance(pending, list) or not all(isinstance(rel, str) for rel in pending):
        raise GeneratorOutputError("its pending list is malformed")
    version, model = data.get("besser_version"), data.get("model")
    return GeneratorOutput(out_dir, version if isinstance(version, str) and _VERSION.fullmatch(version) else None,
                           {rel: content.encode("utf-8") for rel, content in files.items()},
                           model if isinstance(model, dict) else None, frozenset(pending))


def attach_model(workspace: str, model: dict) -> None:
    """Record the class diagram the stored output was generated from.

    Only a store written in this run, which carries no model yet, gets one: an
    older store keeps the model of the run that wrote it. No store, no-op.
    """
    try:
        data = _read_store(workspace)
        if data is None or data.get("model") is not None:
            return
        data["model"] = model
        payload = json.dumps(data, sort_keys=True)
        if len(payload.encode("utf-8")) > MAX_STORED_BYTES:
            raise ValueError(f"with the model it exceeds {MAX_STORED_BYTES} bytes")
        write_atomic_plain(os.path.join(workspace, GENERATOR_OUTPUT_FILENAME), payload, root=workspace)
    except (GeneratorOutputError, OSError, ValueError, TypeError) as exc:
        logger.warning("The model was not stored with the generator output, so the next run cannot "
                       "check the stored files or rebuild the merge base: %s", exc)


def stored_model(workspace: str) -> Optional[dict]:
    """The class diagram attached to the stored output, or None (no store, none attached, or unreadable)."""
    try:
        data = _read_store(workspace)
    except GeneratorOutputError as exc:
        logger.warning("The stored generator output is unreadable: %s", exc)
        return None
    model = data.get("model") if data is not None else None
    return model if isinstance(model, dict) else None


def discard_generator_output(workspace: str) -> None:
    """Remove the stored output, so no later merge uses it as its base."""
    try:
        os.remove(os.path.join(workspace, GENERATOR_OUTPUT_FILENAME))
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not remove a stale %s", GENERATOR_OUTPUT_FILENAME, exc_info=True)


def _read_store(workspace: str) -> Optional[dict]:
    """The store's JSON object, or None when there is none. Raises :class:`GeneratorOutputError`
    for a link, a store over ``MAX_STORED_BYTES`` (it may come from an imported repo) or bad JSON."""
    path = os.path.join(workspace, GENERATOR_OUTPUT_FILENAME)
    if not os.path.lexists(path):
        return None
    try:
        with open_plain(path, "rb", root=workspace) as handle:
            size = os.fstat(handle.fileno()).st_size
            if size > MAX_STORED_BYTES:
                raise GeneratorOutputError(f"it is {size} bytes, over {MAX_STORED_BYTES}")
            data = json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise GeneratorOutputError(f"unreadable ({exc})") from exc
    if not isinstance(data, dict):
        raise GeneratorOutputError("not a stored generator output")
    return data
