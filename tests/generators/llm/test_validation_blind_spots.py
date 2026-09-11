"""Two ways a generated app shipped "0 blockers" while being unable to start.

Both found in a 10-app live batch on 2026-09-11, where the run's own verdict
disagreed with the app in BOTH directions: one app reported "0 blockers /
23 total, done" and could not import, while two reported INCOMPLETE and were
fine (they had merely run out of turns).

1. F811 (redefinition) was filed as a STYLE code, grouped with F401 and F841.
   Those two are cosmetic; F811 is not - it means two things share a name and
   the later one silently wins. Every one of the four F811 hits across the
   batch was a real defect: a duplicated ORM model, a duplicated Create
   schema, a duplicated endpoint function that replaced the first, and an ORM
   `User` shadowed by a Pydantic `User` that was then passed to db.query().

2. A missing MODULE is invisible to ruff. ``from sql_alchemy import *`` makes
   it report "unable to detect undefined names", which EXCUSES every name the
   module needed instead of flagging it. One app imported sql_alchemy and
   pydantic_classes with neither file present anywhere.
"""

import pytest

from besser.generators.llm.orchestrator import (
    _classify_issue,
    _tool_call_detail,
    _unresolvable_local_imports,
)


# --------------------------------------------------- F811 is a blocker


def test_a_redefinition_is_a_blocker_not_a_style_nit():
    issue = _classify_issue(
        "ruff: backend/routers/user.py:5:50: F811 Redefinition of unused "
        "`User` from line 4: `User` redefined here"
    )
    assert issue.severity == "blocker"


@pytest.mark.parametrize("code", ["F821", "F822", "F823"])
def test_the_existing_blockers_are_unchanged(code):
    issue = _classify_issue(f"ruff: app/x.py:1:1: {code} Undefined name `X`")
    assert issue.severity == "blocker"


@pytest.mark.parametrize("code,name", [
    ("F401", "unused import"), ("F841", "unused variable"), ("E501", "long line"),
])
def test_genuinely_cosmetic_codes_stay_style(code, name):
    """Promoting these would make the fix loop chase whitespace."""
    issue = _classify_issue(f"ruff: app/x.py:1:1: {code} something about {name}")
    assert issue.severity == "style"


# ------------------------------------------- missing modules are blockers


def _write(root, rel, text=""):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_live_failure_is_caught(tmp_path):
    """backend/main_api.py imported sql_alchemy; the file was nowhere."""
    _write(tmp_path, "backend/main_api.py",
           "from sql_alchemy import *\nfrom database import get_db\n")
    _write(tmp_path, "backend/database.py", "def get_db():\n    pass\n")
    problems = _unresolvable_local_imports(str(tmp_path))
    assert any("sql_alchemy" in p for p in problems)
    assert all(_classify_issue(p).severity == "blocker" for p in problems)


def test_a_module_in_a_sibling_directory_does_not_count(tmp_path):
    """A copy of pydantic_classes.py under pydantic/ is not reachable from
    backend/. Accepting it hid half of the live breakage."""
    _write(tmp_path, "backend/main_api.py", "from pydantic_classes import *\n")
    _write(tmp_path, "pydantic/pydantic_classes.py", "X = 1\n")
    assert any("pydantic_classes" in p
               for p in _unresolvable_local_imports(str(tmp_path)))


def test_a_router_importing_from_its_service_root_is_fine(tmp_path):
    """The service runs with backend/ as cwd, so backend/ is importable from
    everything beneath it. Resolving only against the file's own directory
    condemned six imports in an app that demonstrably works."""
    _write(tmp_path, "backend/main_api.py", "from routers import team\n")
    _write(tmp_path, "backend/sql_alchemy.py", "class Team: pass\n")
    _write(tmp_path, "backend/database.py", "def get_db(): pass\n")
    _write(tmp_path, "backend/routers/__init__.py")
    _write(tmp_path, "backend/routers/team.py",
           "from sql_alchemy import *\nfrom database import get_db\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_a_namespace_package_needs_no_init(tmp_path):
    """Python 3 implicit namespace packages: `from routers import post` works
    without __init__.py, and demanding it flagged a working app."""
    _write(tmp_path, "main_api.py", "from routers import post\n")
    _write(tmp_path, "routers/post.py", "router = None\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


@pytest.mark.parametrize("line", [
    "from fastapi import FastAPI", "import sqlalchemy", "from jose import jwt",
    "from passlib.context import CryptContext", "import os, sys",
    "from typing import Optional", "from datetime import datetime",
])
def test_third_party_and_stdlib_imports_are_not_flagged(tmp_path, line):
    _write(tmp_path, "backend/main_api.py", line + "\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_a_relative_import_is_left_alone(tmp_path):
    _write(tmp_path, "pkg/__init__.py")
    _write(tmp_path, "pkg/a.py", "from .b import thing\n")
    _write(tmp_path, "pkg/b.py", "thing = 1\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_a_file_that_does_not_parse_is_left_to_the_syntax_check(tmp_path):
    _write(tmp_path, "backend/broken.py", "def oops(:\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_node_modules_is_not_walked(tmp_path):
    _write(tmp_path, "node_modules/dep/setup.py", "from nonexistent import x\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


# ------------------------------------------------ the stream says what happened


def test_the_detail_names_the_file_a_write_touched():
    detail = _tool_call_detail("write_file", {"path": "backend/main_api.py",
                                              "content": "x" * 5000}, 1)
    assert "backend/main_api.py" in detail
    assert "xxxx" not in detail, "file content must never reach the event stream"


def test_the_detail_shows_which_task_ids_were_marked():
    detail = _tool_call_detail("task_list", {"action": "done", "ids": [1, 2, 3]}, 1)
    assert "action=done" in detail
    assert "1,2,3" in detail


def test_batching_is_visible_only_when_it_happened():
    """1 call per turn was the invisible default across every live run."""
    assert "batched" not in _tool_call_detail("write_file", {"path": "a.py"}, 1)
    assert "batched=4" in _tool_call_detail("write_file", {"path": "a.py"}, 4)


def test_the_detail_is_bounded_and_single_line():
    detail = _tool_call_detail(
        "run_command", {"command": "echo " + "long " * 200 + "\nsecond line"}, 1)
    assert len(detail) <= 160
    assert "\n" not in detail


def test_a_tool_with_nothing_worth_reporting_gives_an_empty_detail():
    assert _tool_call_detail("list_files", {}, 1) == ""


def test_odd_input_does_not_raise():
    for bad in (None, "a string", 42, [], {"path": None}):
        _tool_call_detail("write_file", bad, 1)


def test_a_three_argument_on_progress_callback_still_works():
    """``on_progress`` is public API of a published package: a caller may still
    pass the original (turn, tool, status) callback. Passing `detail`
    unconditionally raised TypeError for them and aborted the turn."""
    from besser.generators.llm.orchestrator import LLMOrchestrator

    seen = []
    orch = object.__new__(LLMOrchestrator)       # no heavy __init__ needed
    orch.on_progress = lambda turn, tool, status: seen.append((turn, tool, status))
    orch._emit_progress(3, "write_file", "executing", "path=a.py")
    orch._emit_progress(4, "task_list", "executing", "action=done ids=1,2")
    assert seen == [(3, "write_file", "executing"), (4, "task_list", "executing")]
    # The arity is remembered, so it is not re-probed on every call.
    assert orch._progress_takes_detail is False


def test_a_four_argument_callback_receives_the_detail():
    from besser.generators.llm.orchestrator import LLMOrchestrator

    seen = []
    orch = object.__new__(LLMOrchestrator)
    orch.on_progress = lambda *args: seen.append(args)
    orch._emit_progress(1, "write_file", "executing", "path=a.py")
    assert seen == [(1, "write_file", "executing", "path=a.py")]


def test_a_raising_callback_never_breaks_the_run():
    from besser.generators.llm.orchestrator import LLMOrchestrator

    def explode(*_args):
        raise RuntimeError("callback bug")

    orch = object.__new__(LLMOrchestrator)
    orch.on_progress = explode
    orch._emit_progress(1, "write_file", "executing", "path=a.py")   # must not raise
