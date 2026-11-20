"""The lab's team memory: approved notes in one JSON file outside the workspace, and the agent's two tools."""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from besser.spec_driven_agent.lab.contracts import ChangeKind, NoteDraft
from besser.spec_driven_agent.lab.memory import (
    MAX_BODY_CHARS,
    MAX_INDEX_LINES,
    MAX_NOTES,
    MAX_PROPOSALS_PER_RUN,
    MAX_TITLE_CHARS,
    STALE_AFTER_DAYS,
    LabMemory,
    LabMemoryError,
    LabMemoryFull,
    LabMemoryUnavailable,
    LabNote,
    LabNoteNotFound,
    propose_lab_note_tool,
    read_lab_note_tool,
)


def _draft(title: str = "Generators write via _write_traced", **fields) -> NoteDraft:
    values = {"kind": "decision", "title": title, "body": "Review asked for it on PR 812.", "repos": ("besser",),
              "change_kinds": ("generator",)}
    values.update(fields)
    return NoteDraft(**values)


@pytest.fixture
def memory(tmp_path) -> LabMemory:
    return LabMemory(str(tmp_path / "work-state"), str(tmp_path / "work"))


def _notes_file(tmp_path) -> str:
    return str(tmp_path / "work-state" / "memory" / "notes.json")


def test_constants_are_the_designed_ones():
    assert (MAX_NOTES, MAX_BODY_CHARS, MAX_TITLE_CHARS, MAX_INDEX_LINES, MAX_PROPOSALS_PER_RUN, STALE_AFTER_DAYS) == (
        60, 1_200, 100, 25, 2, 90)


def test_add_get_update_delete_round_trip_through_the_file(memory, tmp_path):
    assert memory.enabled and memory.reason == ""
    assert memory.notes() == []

    note = memory.add(_draft(docs_target="besser/.claude/skills/besser-dev/SKILL.md"), "alice",
                      source_change="3f2a9c1b7e04")

    assert note.id == "m1" and note.approved_by == "alice" and note.source_change == "3f2a9c1b7e04"
    assert note.repos == ["besser"] and note.change_kinds == ["generator"]
    assert note.created_at == note.updated_at > 0 and note.reads == 0
    reopened = LabMemory(str(tmp_path / "work-state"), str(tmp_path / "work"))
    assert reopened.get("m1") == note
    with open(_notes_file(tmp_path), encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["version"] == 1 and stored["next_id"] == 2 and stored["notes"][0]["title"] == note.title

    updated = memory.update("m1", _draft("Generators write via self._write_traced", kind="convention",
                                         docs_target="besser/CLAUDE.md", repos=()))
    assert updated.title == "Generators write via self._write_traced" and updated.kind == "convention"
    assert updated.repos == [] and updated.approved_by == "alice" and updated.created_at == note.created_at
    assert reopened.get("m1") == updated

    second = memory.add(_draft("Another"), "bob")
    assert second.id == "m2"
    memory.delete("m1")
    assert [n.id for n in reopened.notes()] == ["m2"]
    with pytest.raises(LabNoteNotFound):
        memory.get("m1")
    with pytest.raises(LabNoteNotFound):
        memory.delete("m1")
    # Ids are never reused after a delete.
    assert memory.add(_draft("Third"), "bob").id == "m3"


def test_the_61st_note_is_refused(memory):
    for n in range(MAX_NOTES):
        memory.add(_draft(f"Note {n}"), "alice")
    with pytest.raises(LabMemoryFull, match="60"):
        memory.add(_draft("One too many"), "alice")
    assert len(memory.notes()) == MAX_NOTES


@pytest.mark.parametrize("fields, problem", [
    ({"body": "x" * (MAX_BODY_CHARS + 1)}, "1,200"),
    ({"body": "  "}, "body"),
    ({"title": "t" * (MAX_TITLE_CHARS + 1)}, "100"),
    ({"title": "two\nlines"}, "one line"),
    ({"kind": "rumour"}, "kind"),
    ({"kind": "convention"}, "docs_target"),
    ({"repos": ("besser", "elsewhere")}, "elsewhere"),
    ({"change_kinds": ("nonsense",)}, "nonsense"),
    ({"docs_target": "../outside.md"}, "docs_target"),
    ({"docs_target": "/etc/passwd"}, "docs_target"),
])
def test_an_invalid_note_is_refused_and_nothing_is_written(memory, tmp_path, fields, problem):
    with pytest.raises(LabMemoryError, match=problem):
        memory.add(_draft(**fields), "alice")
    assert not os.path.exists(_notes_file(tmp_path))


def test_a_body_of_exactly_the_cap_is_kept(memory):
    assert len(memory.add(_draft(body="b" * MAX_BODY_CHARS), "alice").body) == MAX_BODY_CHARS


def test_a_state_folder_inside_the_workspace_root_turns_memory_off(tmp_path):
    root = tmp_path / "work"
    memory = LabMemory(str(root / ".lab"), str(root))

    assert memory.enabled is False
    assert "inside the workspace root" in memory.reason and "BESSER_LAB_STATE_DIR" in memory.reason
    calls = [memory.notes, lambda: memory.get("m1"), lambda: memory.add(_draft(), "alice"),
             lambda: memory.update("m1", _draft()), lambda: memory.delete("m1"), lambda: memory.mark_read("m1"),
             lambda: memory.index_text(["besser"], ChangeKind.GENERATOR)]
    for call in calls:
        with pytest.raises(LabMemoryUnavailable, match="inside the workspace root"):
            call()
    assert not (root / ".lab").exists()


def test_the_index_is_filtered_by_repository_and_kind(memory):
    memory.add(_draft("Besser generator note"), "alice")
    memory.add(_draft("Editor note", repos=("frontend",), change_kinds=()), "alice")
    memory.add(_draft("Everywhere note", repos=(), change_kinds=()), "alice")
    memory.add(_draft("Besser metamodel note", change_kinds=("metamodel",)), "alice")

    text = memory.index_text(["besser"], ChangeKind.GENERATOR)

    assert "Besser generator note" in text and "Everywhere note" in text
    assert "Editor note" not in text and "Besser metamodel note" not in text
    assert "m1 [decision] (besser; generator) Besser generator note" in text
    assert "m3 [decision] (all repositories; every kind) Everywhere note" in text
    any_kind = memory.index_text(["besser"], None)
    assert "Besser metamodel note" in any_kind and "Editor note" not in any_kind
    assert memory.index_text(["modeling_agent"], ChangeKind.AGENT) == "- m3 [decision] (all repositories; every kind) Everywhere note"


def test_the_index_is_capped_at_25_lines_newest_first(memory):
    for n in range(40):
        memory.add(_draft(f"Note {n:02d}"), "alice")
    lines = memory.index_text(["besser"], "generator").splitlines()
    assert len(lines) == MAX_INDEX_LINES
    assert "Note 39" in lines[0]
    assert lines[-1] == "(16 older notes not listed)"


def test_an_empty_memory_has_an_empty_index(memory):
    assert memory.index_text(["besser"], None) == ""


def test_concurrent_adds_from_two_threads_keep_both(memory):
    errors = []

    def add_many(prefix: str) -> None:
        try:
            for n in range(15):
                memory.add(_draft(f"{prefix} {n}"), prefix)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=add_many, args=(name,)) for name in ("alice", "bob")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    notes = memory.notes()
    assert len(notes) == 30 and len({n.id for n in notes}) == 30
    assert sum(n.approved_by == "alice" for n in notes) == 15


@pytest.mark.parametrize("content", ["{not json", '{"version": 1, "notes": "nope"}', "[]",
                                     '{"version": 1, "next_id": 2, "notes": [{"id": "m1"}]}'])
def test_a_corrupt_notes_file_is_reported_and_never_overwritten(memory, tmp_path, content):
    path = _notes_file(tmp_path)
    os.makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)

    for call in (memory.notes, lambda: memory.add(_draft(), "alice"), lambda: memory.index_text(["besser"], None)):
        with pytest.raises(LabMemoryUnavailable, match="notes.json"):
            call()

    with open(path, encoding="utf-8") as handle:
        assert handle.read() == content


def test_mark_read_counts_reads_and_the_stale_flag_follows_them(memory):
    note = memory.add(_draft(), "alice")
    later = note.created_at + (STALE_AFTER_DAYS + 1) * 86_400
    assert not note.is_stale(note.created_at + 86_400)
    assert note.is_stale(later)

    memory.mark_read("m1")
    read = memory.get("m1")
    assert read.reads == 1 and read.last_read_at >= note.created_at
    assert not read.is_stale(read.last_read_at + STALE_AFTER_DAYS * 86_400 - 1)
    assert read.updated_at == note.updated_at  # a read is not an edit
    with pytest.raises(LabNoteNotFound):
        memory.mark_read("m9")


def test_lab_note_kind_values():
    assert LabNote(id="m1", kind="mistake", title="t", body="b", repos=[], change_kinds=[]).docs_target == ""


# -- tools ---------------------------------------------------------------------------------------------------


def test_read_lab_note_returns_the_note_and_counts_the_read(memory):
    memory.add(_draft(docs_target="besser/CLAUDE.md"), "alice")
    tool = read_lab_note_tool(memory)

    assert tool.writes is False
    assert tool.schema["name"] == "read_lab_note"
    assert tool.schema["input_schema"]["required"] == ["id"]
    text = tool.handler({"id": "m1"})

    assert "Generators write via _write_traced" in text and "Review asked for it on PR 812." in text
    assert "approved by alice" in text and "besser/CLAUDE.md" in text
    assert memory.get("m1").reads == 1
    with pytest.raises(ValueError, match="m7"):
        tool.handler({"id": "m7"})
    with pytest.raises(ValueError):
        tool.handler({})


def test_read_lab_note_on_a_disabled_memory_says_why(tmp_path):
    tool = read_lab_note_tool(LabMemory(str(tmp_path / "w" / "state"), str(tmp_path / "w")))
    with pytest.raises(ValueError, match="inside the workspace root"):
        tool.handler({"id": "m1"})


def test_propose_lab_note_passes_drafts_on_and_stops_at_two_per_run():
    proposals = []
    tool = propose_lab_note_tool(proposals.append)
    arguments = {"kind": "mistake", "title": "Ran pytest unscoped", "body": "Scope it to tests/.",
                 "repos": ["besser"], "change_kinds": [], "why": "It collected the docs examples."}

    assert tool.writes is False and tool.schema["name"] == "propose_lab_note"
    assert set(tool.schema["input_schema"]["required"]) == {"kind", "title", "body", "why"}
    assert "approve" in tool.handler(arguments)
    tool.handler({**arguments, "title": "Second"})
    with pytest.raises(ValueError, match="2"):
        tool.handler({**arguments, "title": "Third"})

    assert proposals == [
        NoteDraft("mistake", "Ran pytest unscoped", "Scope it to tests/.", ("besser",), (), "",
                  "It collected the docs examples."),
        NoteDraft("mistake", "Second", "Scope it to tests/.", ("besser",), (), "", "It collected the docs examples."),
    ]
    # A new run gets a new tool and a new allowance.
    propose_lab_note_tool(proposals.append).handler({**arguments, "title": "Next run"})
    assert len(proposals) == 3


def test_an_invalid_proposal_lists_its_problems_and_does_not_count():
    proposals = []
    tool = propose_lab_note_tool(proposals.append)
    with pytest.raises(ValueError) as caught:
        tool.handler({"kind": "convention", "title": "", "body": "x" * (MAX_BODY_CHARS + 1), "why": "w"})
    message = str(caught.value)
    assert "title" in message and "1,200" in message and "docs_target" in message
    assert proposals == []
    for n in range(MAX_PROPOSALS_PER_RUN):
        tool.handler({"kind": "decision", "title": f"ok {n}", "body": "b", "why": "w"})
    assert len(proposals) == MAX_PROPOSALS_PER_RUN


def test_notes_are_listed_oldest_first(memory):
    first = memory.add(_draft("First"), "alice")
    time.sleep(0.01)
    memory.add(_draft("Second"), "alice")
    assert [n.title for n in memory.notes()] == ["First", "Second"]
    assert memory.notes()[0] == first
