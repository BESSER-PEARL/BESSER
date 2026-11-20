"""The lab's team memory: a small list of reviewed notes kept outside the working copies.

A note is a team decision not yet written down, a mistake the agent made and corrected, or a convention
(how the code works) waiting to be moved into the docs it names. Notes live in one JSON file,
``<state_dir>/memory/notes.json``; only notes a lab user approved are stored. The build step proposes
notes with ``propose_lab_note`` (they wait on the run card for a person); the agent reads stored notes
with ``read_lab_note``. Memory is off when the state folder is inside the workspace root, where
sandboxed code could plant a note that every later run would read.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Literal, Optional, Sequence

from besser.spec_driven_agent.lab.contracts import REPO_ORDER, ChangeKind, ExtraTool, NoteDraft

logger = logging.getLogger(__name__)

NoteKind = Literal["decision", "mistake", "convention"]
_NOTE_KINDS: tuple[str, ...] = ("decision", "mistake", "convention")

#: A list a person can review on one page; past it, a note has to go (or move to the docs) first.
MAX_NOTES = 60
#: A note is a paragraph; a longer text belongs in a CLAUDE.md or a skill.
MAX_BODY_CHARS = 1_200
MAX_TITLE_CHARS = 100
#: The notes index in the system prompt; with titles of 100 characters it stays within the knowledge section.
MAX_INDEX_LINES = 25
#: The build proposes what it learned at the end; more than two per run is noise for the approver.
MAX_PROPOSALS_PER_RUN = 2
#: A note no change has read for this long is flagged on the lab page; a person decides whether to keep it.
STALE_AFTER_DAYS = 90

_FILE_VERSION = 1
_SECONDS_PER_DAY = 86_400


class LabMemoryError(Exception):
    """A memory call that cannot be done; the message is written for the user."""


class LabMemoryUnavailable(LabMemoryError):
    """The memory is off (state folder inside the workspace root) or its file cannot be read."""


class LabMemoryFull(LabMemoryError):
    """The memory holds :data:`MAX_NOTES` notes."""


class LabNoteNotFound(LabMemoryError):
    """No note has that id."""


@dataclass
class LabNote:
    """An approved team note."""

    id: str
    #: a :data:`NoteKind`
    kind: str
    title: str
    body: str
    #: repository keys it applies to; empty: all
    repos: list[str]
    #: :class:`ChangeKind` values it applies to; empty: every kind
    change_kinds: list[str]
    #: the CLAUDE.md or skill it belongs in when it states how the code works
    docs_target: str = ""
    #: the change it was proposed in; "" for a note written by hand
    source_change: str = ""
    source_url: str = ""
    approved_by: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    #: 0 when never read
    last_read_at: float = 0.0
    reads: int = 0

    def is_stale(self, now: Optional[float] = None) -> bool:
        """No change has read it (or it was written) within :data:`STALE_AFTER_DAYS`."""
        now = time.time() if now is None else now
        return now - max(self.last_read_at, self.created_at) > STALE_AFTER_DAYS * _SECONDS_PER_DAY


_NOTE_FIELDS = {f.name for f in fields(LabNote)}


def _draft_problems(draft: NoteDraft) -> list[str]:
    """Every reason ``draft`` cannot be stored, written for the person or the model that wrote it."""
    problems = []
    if draft.kind not in _NOTE_KINDS:
        problems.append(f"kind must be one of {', '.join(_NOTE_KINDS)}, not {draft.kind!r}")
    title = draft.title.strip() if isinstance(draft.title, str) else ""
    if not title:
        problems.append("title is empty")
    elif "\n" in title:
        problems.append("title must be one line")
    elif len(title) > MAX_TITLE_CHARS:
        problems.append(f"title is {len(title):,} characters; at most {MAX_TITLE_CHARS:,}")
    body = draft.body.strip() if isinstance(draft.body, str) else ""
    if not body:
        problems.append("body is empty")
    elif len(body) > MAX_BODY_CHARS:
        problems.append(f"body is {len(body):,} characters; at most {MAX_BODY_CHARS:,}")
    unknown_repos = [repo for repo in draft.repos if repo not in REPO_ORDER]
    if unknown_repos:
        problems.append(f"unknown repositories {', '.join(map(str, unknown_repos))} "
                        f"(known: {', '.join(REPO_ORDER)})")
    kinds = {kind.value for kind in ChangeKind}
    unknown_kinds = [kind for kind in draft.change_kinds if kind not in kinds]
    if unknown_kinds:
        problems.append(f"unknown change kinds {', '.join(map(str, unknown_kinds))}")
    target = draft.docs_target.strip()
    if draft.kind == "convention" and not target:
        problems.append("a convention needs a docs_target: the CLAUDE.md or skill it belongs in")
    parts = target.replace("\\", "/").split("/")
    if target and (os.path.isabs(target) or target.startswith(("/", "\\")) or ".." in parts):
        problems.append(f"docs_target must be a path inside a repository, not {target!r}")
    return problems


def _check_draft(draft: NoteDraft) -> None:
    problems = _draft_problems(draft)
    if problems:
        raise LabMemoryError("The note was not stored: " + "; ".join(problems) + ".")


def _within(path: str, folder: str) -> bool:
    path, folder = (os.path.normcase(os.path.realpath(p)) for p in (path, folder))
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


class LabMemory:
    """The team's notes in ``<state_dir>/memory/notes.json``. One instance per state folder.

    Every call refuses with :class:`LabMemoryUnavailable` while :attr:`enabled` is False, and when
    the file cannot be read (it is then left as it is, for a person to fix or delete).
    """

    def __init__(self, state_dir: str, workspace_root: str):
        self.path = os.path.join(os.path.abspath(state_dir), "memory", "notes.json")
        #: False when the state folder is inside the workspace root; :attr:`reason` says why
        self.enabled: bool = not _within(state_dir, workspace_root)
        self.reason: str = "" if self.enabled else (
            f"Team memory is off: the state folder {os.path.abspath(state_dir)} is inside the workspace root "
            f"{os.path.abspath(workspace_root)}, where code the agent runs could write a note. Set "
            "BESSER_LAB_STATE_DIR to a folder outside it.")
        if not self.enabled:
            logger.warning("BESSER Lab: %s", self.reason)
        self._lock = threading.Lock()

    # -- file ----------------------------------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        """``{"version", "next_id", "notes": [LabNote]}``; an empty memory when the file does not exist yet."""
        if not self.enabled:
            raise LabMemoryUnavailable(self.reason)
        if not os.path.exists(self.path):
            return {"version": _FILE_VERSION, "next_id": 1, "notes": []}
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or not isinstance(data.get("notes"), list):
                raise ValueError("not an object with a notes list")
            next_id = data.get("next_id")
            if not isinstance(next_id, int) or next_id < 1:
                raise ValueError("next_id is not a positive number")
            notes = [LabNote(**{key: value for key, value in note.items() if key in _NOTE_FIELDS})
                     for note in data["notes"]]
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            message = (f"The team notes in {self.path} cannot be read ({exc}); fix or delete that file. "
                       "Nothing was written to it.")
            logger.warning("BESSER Lab: %s", message)
            raise LabMemoryUnavailable(message) from None
        return {"version": _FILE_VERSION, "next_id": next_id, "notes": notes}

    def _save(self, data: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {"version": _FILE_VERSION, "next_id": data["next_id"], "notes": [asdict(n) for n in data["notes"]]}
        with open(self.path + ".tmp", "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(self.path + ".tmp", self.path)

    @staticmethod
    def _find(data: dict[str, Any], note_id: str) -> LabNote:
        for note in data["notes"]:
            if note.id == note_id:
                return note
        raise LabNoteNotFound(f"No team note {note_id!r}.")

    # -- notes ---------------------------------------------------------------------------------------------

    def notes(self) -> list[LabNote]:
        """Every note, oldest first."""
        with self._lock:
            return list(self._load()["notes"])

    def get(self, note_id: str) -> LabNote:
        """The note ``note_id``; :class:`LabNoteNotFound` when there is none."""
        with self._lock:
            return self._find(self._load(), note_id)

    def add(self, draft: NoteDraft, approved_by: str, *, source_change: str = "") -> LabNote:
        """Store ``draft`` as approved by ``approved_by``; :class:`LabMemoryFull` past :data:`MAX_NOTES`."""
        _check_draft(draft)
        with self._lock:
            data = self._load()
            if len(data["notes"]) >= MAX_NOTES:
                raise LabMemoryFull(f"The team memory holds {MAX_NOTES} notes, its maximum: delete one, or move "
                                    "one into the docs it belongs in, first.")
            now = time.time()
            note = LabNote(id=f"m{data['next_id']}", kind=draft.kind, title=draft.title.strip(),
                           body=draft.body.strip(), repos=list(draft.repos), change_kinds=list(draft.change_kinds),
                           docs_target=draft.docs_target.strip(), source_change=source_change,
                           approved_by=approved_by, created_at=now, updated_at=now)
            data["notes"].append(note)
            data["next_id"] += 1
            self._save(data)
            return note

    def update(self, note_id: str, draft: NoteDraft) -> LabNote:
        """Replace the text and scope of note ``note_id`` with ``draft``'s; who approved it is kept."""
        _check_draft(draft)
        with self._lock:
            data = self._load()
            note = self._find(data, note_id)
            note.kind, note.title, note.body = draft.kind, draft.title.strip(), draft.body.strip()
            note.repos, note.change_kinds = list(draft.repos), list(draft.change_kinds)
            note.docs_target = draft.docs_target.strip()
            note.updated_at = time.time()
            self._save(data)
            return note

    def delete(self, note_id: str) -> None:
        """Remove note ``note_id``; its id is never given again."""
        with self._lock:
            data = self._load()
            data["notes"].remove(self._find(data, note_id))
            self._save(data)

    def mark_read(self, note_id: str) -> None:
        """Count a read of note ``note_id`` by the agent (what the stale flag follows)."""
        with self._lock:
            data = self._load()
            note = self._find(data, note_id)
            note.reads += 1
            note.last_read_at = time.time()
            self._save(data)

    def index_text(self, repos: Sequence[str], kind: Optional[ChangeKind | str]) -> str:
        """One line per note that applies to ``repos`` and ``kind``, newest first, at most :data:`MAX_INDEX_LINES`.

        A note applies when its repositories are empty or share one with ``repos``, and its change
        kinds are empty or hold ``kind`` (any kind when ``kind`` is None). "" when none applies.
        """
        wanted_kind = None if kind is None else ChangeKind(kind).value
        with self._lock:
            notes = self._load()["notes"]
        matching = [note for note in notes
                    if (not note.repos or set(note.repos) & set(repos))
                    and (wanted_kind is None or not note.change_kinds or wanted_kind in note.change_kinds)]
        matching.sort(key=lambda note: (note.updated_at, int(note.id[1:]) if note.id[1:].isdigit() else 0),
                      reverse=True)
        shown = matching if len(matching) <= MAX_INDEX_LINES else matching[:MAX_INDEX_LINES - 1]
        lines = [f"- {note.id} [{note.kind}] ({', '.join(note.repos) or 'all repositories'}; "
                 f"{', '.join(note.change_kinds) or 'every kind'}) {note.title}" for note in shown]
        if len(shown) < len(matching):
            lines.append(f"({len(matching) - len(shown)} older notes not listed)")
        return "\n".join(lines)


# -- tools ---------------------------------------------------------------------------------------------------

_READ_SCHEMA: dict[str, Any] = {
    "name": "read_lab_note",
    "description": "Read a team note listed in the Knowledge section of your instructions, by its id (for example "
                   "m7). Notes are decisions and corrections the team approved; read one when its title bears on "
                   "your task.",
    "input_schema": {
        "type": "object",
        "required": ["id"],
        "properties": {"id": {"type": "string", "description": "The note's id, as listed (m<number>)."}},
    },
}

_PROPOSE_SCHEMA: dict[str, Any] = {
    "name": "propose_lab_note",
    "description": f"Propose a team note for a lab user to approve, edit or dismiss (at most "
                   f"{MAX_PROPOSALS_PER_RUN} per run). Propose a review decision the team made, or a mistake you "
                   "made and corrected, that later changes should know. A rule of the codebase (a file to touch, a "
                   "command, a convention) is a 'convention' and names the docs_target it belongs in.",
    "input_schema": {
        "type": "object",
        "required": ["kind", "title", "body", "why"],
        "properties": {
            "kind": {"type": "string", "enum": list(_NOTE_KINDS)},
            "title": {"type": "string", "description": f"One line, at most {MAX_TITLE_CHARS} characters."},
            "body": {"type": "string", "description": f"At most {MAX_BODY_CHARS:,} characters."},
            "repos": {"type": "array", "items": {"type": "string", "enum": list(REPO_ORDER)},
                      "description": "Repositories it applies to; empty: all."},
            "change_kinds": {"type": "array", "items": {"type": "string", "enum": [k.value for k in ChangeKind]},
                             "description": "Kinds of change it applies to; empty: every kind."},
            "docs_target": {"type": "string",
                            "description": "The CLAUDE.md or skill it belongs in (a repository path); required for "
                                           "a convention."},
            "why": {"type": "string", "description": "One sentence for the approver: why the team should keep it."},
        },
    },
}


def _format_note(note: LabNote) -> str:
    scope = f"{', '.join(note.repos) or 'all repositories'}; {', '.join(note.change_kinds) or 'every kind'}"
    lines = [f"Team note {note.id} [{note.kind}] ({scope}), approved by {note.approved_by or 'a lab user'}:",
             note.title, "", note.body]
    if note.docs_target:
        lines += ["", f"It belongs in {note.docs_target}: a change to that file may move it there."]
    return "\n".join(lines)


def read_lab_note_tool(memory: LabMemory) -> ExtraTool:
    """The ``read_lab_note`` tool: the full text of an approved note, counted as a read."""
    def handle(arguments: dict[str, Any]) -> str:
        note_id = arguments.get("id")
        if not isinstance(note_id, str) or not note_id.strip():
            raise ValueError("Give the note's id, for example m7.")
        try:
            note = memory.get(note_id.strip())
            memory.mark_read(note.id)
        except LabMemoryError as exc:
            raise ValueError(str(exc)) from None
        return _format_note(note)

    return ExtraTool(schema=_READ_SCHEMA, handler=handle, writes=False)


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("must be a list of strings")
    return tuple(value)


def propose_lab_note_tool(on_proposal: Callable[[NoteDraft], None]) -> ExtraTool:
    """The ``propose_lab_note`` tool of one run: checks the draft and passes it to ``on_proposal``.

    It stores nothing (a person approves first). At most :data:`MAX_PROPOSALS_PER_RUN` proposals
    per tool, so make one per run; an invalid draft raises ``ValueError`` listing every problem and
    does not count.
    """
    accepted: list[NoteDraft] = []

    def handle(arguments: dict[str, Any]) -> str:
        if len(accepted) >= MAX_PROPOSALS_PER_RUN:
            raise ValueError(f"Not proposed: this run already proposed {MAX_PROPOSALS_PER_RUN} notes, the most a "
                             "run may.")
        problems = []
        lists = {}
        for name in ("repos", "change_kinds"):
            try:
                lists[name] = _strings(arguments.get(name))
            except ValueError as exc:
                problems.append(f"{name} {exc}")
                lists[name] = ()
        text = {name: arguments.get(name, "") for name in ("kind", "title", "body", "docs_target", "why")}
        for name, value in text.items():
            if not isinstance(value, str):
                problems.append(f"{name} must be a string")
                text[name] = ""
        draft = NoteDraft(kind=text["kind"], title=text["title"].strip(), body=text["body"].strip(),
                          repos=lists["repos"], change_kinds=lists["change_kinds"],
                          docs_target=text["docs_target"].strip(), why=text["why"].strip())
        problems += _draft_problems(draft)
        if problems:
            raise ValueError("The note was not proposed. Fix these and call propose_lab_note again:\n- "
                             + "\n- ".join(problems))
        accepted.append(draft)
        on_proposal(draft)
        return "Proposed. A lab user approves, edits or dismisses it on the run card; nothing is stored before that."

    return ExtraTool(schema=_PROPOSE_SCHEMA, handler=handle, writes=False)

