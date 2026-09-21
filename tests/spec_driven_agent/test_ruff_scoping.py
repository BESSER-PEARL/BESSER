"""Phase 3 must never truncate a ruff blocker away to make room for noise.

``_collect_ruff_issues`` used to report ruff's first 20 output lines. Ruff
sorts by path, so on a real workspace those 20 were always the same
alphabetically-first scaffold files — across seven live runs not one reported
line was in a file the agent had edited, and a genuine F821 late in the
alphabet never reached the fix loop at all. The truncation note was also
emitted as a ``ruff:`` finding and classified ``warning``, so the count the
loop gates on included a line that is not a defect.
"""

import os
import shutil

import pytest

from besser.spec_driven_agent.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.validation.issues import _classify_issue


pytestmark = pytest.mark.skipif(shutil.which("ruff") is None,
                                reason="ruff is not installed on this host")

# Unused imports (F401) are the scaffold noise; an undefined name (F821) is
# the blocker. Both are in ruff's default rule set.
NOISE = "import os\nimport sys\nimport json\n"
BLOCKER = "def handler():\n    return undefined_symbol()\n"


def _orchestrator(model, tmp_path, edited=()):
    class Client:
        model = "mock-model"

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    orchestrator = LLMOrchestrator(
        llm_client=Client(), domain_model=model, output_dir=str(tmp_path),
        enable_checkpointing=False,
    )
    orchestrator.tool_calls_log = [
        {"tool": "modify_file", "input": {"path": path}, "success": True}
        for path in edited
    ]
    return orchestrator


def _write(tmp_path, name, body):
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def test_a_blocker_survives_a_workspace_full_of_style_noise(simple_library_book_model, tmp_path):
    """The F821 sorts last by path and is far past the 20-line cap."""
    for index in range(40):
        _write(tmp_path, f"aaa_scaffold_{index:03d}.py", NOISE)
    _write(tmp_path, "zzz_router.py", BLOCKER)

    issues = _orchestrator(simple_library_book_model, tmp_path)._collect_ruff_issues()

    blockers = [i for i in issues if "F821" in i]
    assert blockers, f"F821 truncated away; reported instead: {issues[:3]}"
    assert "zzz_router.py" in blockers[0]
    # It is reported first, not buried under forty unused imports.
    assert issues[0] == blockers[0]


def test_edited_files_outrank_untouched_scaffold(simple_library_book_model, tmp_path):
    for index in range(40):
        _write(tmp_path, f"aaa_scaffold_{index:03d}.py", NOISE)
    _write(tmp_path, "zzz_edited.py", NOISE)

    issues = _orchestrator(simple_library_book_model, tmp_path, edited=["zzz_edited.py"])._collect_ruff_issues()

    assert "zzz_edited.py" in issues[0], issues[:3]


def test_the_truncation_note_is_not_counted_as_a_defect(simple_library_book_model, tmp_path):
    for index in range(40):
        _write(tmp_path, f"aaa_scaffold_{index:03d}.py", NOISE)

    issues = _orchestrator(simple_library_book_model, tmp_path)._collect_ruff_issues()

    notes = [i for i in issues if i.startswith("ruff: (+")]
    assert len(notes) == 1, issues[-3:]
    assert _classify_issue(notes[0]).severity == "style"
    # Every other reported line is a real finding with a rule code.
    assert all(_classify_issue(i).severity in ("style", "warning", "blocker")
               for i in issues)


def test_the_reported_path_is_resolved_against_the_workspace(simple_library_book_model, tmp_path):
    """A relative tool-call path must match ruff's absolute output path."""
    _write(tmp_path, "backend/routers/booking.py", NOISE)
    orchestrator = _orchestrator(simple_library_book_model, tmp_path, edited=["backend/routers/booking.py"])

    edited = orchestrator._llm_edited_paths()

    assert edited == {os.path.normcase(os.path.normpath(
        str(tmp_path / "backend" / "routers" / "booking.py")))}
