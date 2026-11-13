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

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.validation.issues import _classify_issue


pytestmark = pytest.mark.skipif(shutil.which("ruff") is None,
                                reason="ruff is not installed on this host")

# Phase 3 selects only runtime-defect rules, so the non-blocker noise is a
# literal comparison with ``is`` (F632); an undefined name (F821) is the blocker.
NOISE = 'a = 1 is "x"\nb = 1 is "y"\nc = 1 is "z"\n'
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
    # It is reported first, not buried under forty noisy files.
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


@pytest.mark.parametrize("note", [
    "tsc [frontend]: (+12 more errors truncated)",
    "cargo [.]: (+3 more errors truncated)",
    "kotlinc [app/src/main/kotlin]: (+3 more errors truncated)",
])
def test_every_toolchain_truncation_note_is_a_count_not_a_defect(note):
    """The compilers' notes (toolchain.py) are the same count as ruff's."""
    assert _classify_issue(note).severity == "style"
    # A compiler error line keeps its blocker severity.
    assert _classify_issue(note.split(": ")[0] + ": src/a.ts(1,1): error TS2304").severity == "blocker"


def test_the_reported_path_is_resolved_against_the_workspace(simple_library_book_model, tmp_path):
    """A relative tool-call path must match ruff's absolute output path."""
    _write(tmp_path, "backend/routers/booking.py", NOISE)
    orchestrator = _orchestrator(simple_library_book_model, tmp_path, edited=["backend/routers/booking.py"])

    edited = orchestrator._llm_edited_paths()

    assert edited == {os.path.normcase(os.path.normpath(
        str(tmp_path / "backend" / "routers" / "booking.py")))}


# ------------------------------------------------- both halves of the match

@pytest.mark.parametrize("cwd_is", ["an_ancestor_of_the_workspace", "elsewhere"])
def test_edited_files_outrank_scaffold_from_any_cwd(
    simple_library_book_model, tmp_path, monkeypatch, cwd_is
):
    """The ranking compares ruff's paths against _llm_edited_paths' absolute ones.

    Ruff prints its paths RELATIVE to the cwd whenever the target sits
    underneath it, and absolute otherwise, so the spelling depended on where
    the server process happened to be started. From a cwd above the workspace
    every comparison failed and the edited tier ranked nothing -- silently, as a
    correct-looking ordering. The hosted backend runs from /app with workspaces
    under the system temp dir, which is why this never showed up there; a local
    harness run from the parent directory hits it.

    test_the_reported_path_is_resolved_against_the_workspace pins the absolute
    half of the same comparison. This pins the half that varied.
    """
    workspace = tmp_path / "app"
    workspace.mkdir()
    for index in range(40):
        _write(workspace, f"aaa_scaffold_{index:03d}.py", NOISE)
    _write(workspace, "zzz_edited.py", NOISE)
    if cwd_is == "an_ancestor_of_the_workspace":
        monkeypatch.chdir(tmp_path)          # ruff prints "app/zzz_edited.py"
    else:
        # A sibling tree, so nothing shares a prefix: ruff prints absolutes.
        unrelated = tmp_path.parent / (tmp_path.name + "_unrelated_cwd")
        unrelated.mkdir(exist_ok=True)
        monkeypatch.chdir(unrelated)

    orchestrator = _orchestrator(
        simple_library_book_model, workspace, edited=["zzz_edited.py"])
    issues = orchestrator._collect_ruff_issues()

    assert "zzz_edited.py" in issues[0], issues[:3]


def test_a_ruff_line_it_cannot_parse_is_not_treated_as_edited(
    simple_library_book_model, tmp_path
):
    """Unparseable must fall to the untouched tier, not silently match ''."""
    from besser.spec_driven_agent.validation.toolchain import _ruff_line_path

    assert _ruff_line_path("not a ruff line", str(tmp_path)) == ""
    assert _ruff_line_path("", str(tmp_path)) == ""


@pytest.mark.parametrize("vendored", ["frontend/node_modules/flatted/python", ".venv/lib", "venv/lib"])
def test_installed_dependencies_are_not_linted(simple_library_book_model, tmp_path, vendored):
    """An explicit --exclude replaces ruff's defaults, which skip these."""
    _write(tmp_path, f"{vendored}/x.py", BLOCKER)

    issues = _orchestrator(simple_library_book_model, tmp_path)._collect_ruff_issues()

    assert issues == []


def test_one_missing_import_is_one_ruff_blocker_per_name(simple_library_book_model, tmp_path):
    """A router with 40 endpoints that forgot one import was 80 ruff blockers."""
    _write(tmp_path, "router.py", "".join(
        f"def r{i}(db=get_db()):\n    return db.query(Thing)\n" for i in range(40)))

    issues = _orchestrator(simple_library_book_model, tmp_path)._collect_ruff_issues()

    blockers = [i for i in issues if _classify_issue(i).severity == "blocker"]
    assert blockers == [
        "ruff: router.py:1:11: F821 Undefined name `get_db` (+39 more uses)",
        "ruff: router.py:2:21: F821 Undefined name `Thing` (+39 more uses)",
    ]


def test_only_runtime_defects_are_selected_and_app_config_is_not_read(tmp_path):
    """Ruff's 400+ default rules flagged the scaffold's own idioms (B008 on
    ``Depends(...)``, DTZ005); an app ``ruff.toml`` ignoring F821 hid the blocker."""
    from besser.spec_driven_agent.validation.toolchain import _collect_ruff_issues

    _write(tmp_path, "app.py", (
        "from datetime import datetime\n"
        "from fastapi import Depends\n\n\n"
        "def get_db():\n    return 1\n\n\n"
        "def endpoint(db=Depends(get_db)):\n"
        "    return datetime.now(), missing_name\n"
    ))
    _write(tmp_path, "ruff.toml", '[lint.per-file-ignores]\n"app.py" = ["F821"]\n')

    issues, _ = _collect_ruff_issues(str(tmp_path), [], frozenset(), False)

    assert len(issues) == 1, issues
    assert "F821" in issues[0] and "missing_name" in issues[0]
    assert _classify_issue(issues[0]).severity == "blocker"
