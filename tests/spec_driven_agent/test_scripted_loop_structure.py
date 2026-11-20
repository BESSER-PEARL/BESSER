"""Structural properties of the Phase 2 and Phase 3 loops, run against a scripted model.

The REAL orchestrator runs a whole run of the shelf app (FastAPI scaffold); only
the model is scripted (``scripted_rig``). The script is small and neutral: read
the scaffold, add a stats route and two modules, validate, close the checklist.

Tests marked ``xfail(strict=True)`` record what the loop does today and are the
acceptance gates of the packages named in their reasons: when a package lands,
its gate passes and the marker has to go.
"""

from __future__ import annotations

import hashlib
import os

import pytest

import besser.spec_driven_agent.pipeline.orchestrator as orchestrator_module
import besser.spec_driven_agent.pipeline.phase3_repair as phase3_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from tests.spec_driven_agent.scripted_rig import FakeClock, Rule, ScriptedAgent, run_scripted, script_from_trace
from tests.spec_driven_agent.scripts.trace_metrics import load_runs, metrics

TASKS = [
    "Add a GET /book/stats/ endpoint returning the total page count of all books",
    "Add a shelf statistics module with per-library page totals",
]
BOOK_ROUTER = "backend/routers/book.py"
MAIN = "backend/main_api.py"
INCLUDE = "app.include_router(book_router.router)"
SEARCH_ROUTE = '@router.get("/book/search/", response_model=None, tags=["Book"])'
STATS_ROUTE = (
    '@router.get("/book/stats/", response_model=None, tags=["Book"])\n'
    "def book_stats(database: Session = Depends(get_db)) -> dict:\n"
    '    return {"pages": sum(book.pages or 0 for book in database.query(Book).all())}\n\n\n'
)
EDIT_TOOLS = {"modify_file", "write_file", "replace_file_lines"}
# 12 turns: the script below needs 16 at one call per turn, 8 at two, 4 at four.
# With auto-fix on, a cap this small reserves no turns for Phase 3 (the reserve
# is at most a quarter of the cap and is dropped below one fix attempt).
MAX_TURNS = 12


def _module(route: str, name: str) -> str:
    return (
        "from fastapi import APIRouter, Depends\nfrom sqlalchemy.orm import Session\n\n"
        "from database import get_db\nfrom sql_alchemy import Library\n\nrouter = APIRouter()\n\n\n"
        f'@router.get("{route}", tags=["Shelf"])\n'
        f"def {name}(database: Session = Depends(get_db)) -> dict:\n"
        "    return {library.name: sum(book.pages or 0 for book in library.books)\n"
        "            for library in database.query(Library).all()}\n"
    )


def _include(module: str) -> str:
    return INCLUDE + f"\nfrom routers import {module} as {module}_router\napp.include_router({module}_router.router)"


def _done(index: int, path: str, quote: str):
    def build(agent: ScriptedAgent):
        task = next(t["id"] for t in agent.orchestrator.executor.task_snapshot() if t["text"] == TASKS[index])
        return ("task_list", {"action": "done", "id": task,
                              "evidence": [{"id": task, "path": path, "quote": quote}]})
    return build


def shelf_script(close_last_task: bool = True) -> list:
    """16 calls. Reads come first, so no batch size puts a read beside the edit that needs it."""
    return [
        ("list_files", {}),
        ("read_file", {"path": MAIN}),
        ("read_file", {"path": BOOK_ROUTER}),
        ("read_file", {"path": "backend/sql_alchemy.py"}),
        ("modify_file", {"path": BOOK_ROUTER, "old_text": SEARCH_ROUTE, "new_text": STATS_ROUTE + SEARCH_ROUTE}),
        ("write_file", {"path": "backend/routers/shelf_stats.py",
                        "content": _module("/shelf/stats/", "shelf_stats")}),
        ("modify_file", {"path": MAIN, "old_text": INCLUDE, "new_text": _include("shelf_stats")}),
        ("validate_app", {}),
        _done(0, BOOK_ROUTER, "def book_stats("),
        ("read_file", {"path": "backend/pydantic_classes.py"}),
        ("write_file", {"path": "backend/routers/shelf_totals.py",
                        "content": _module("/shelf/totals/", "shelf_totals")}),
        ("modify_file", {"path": MAIN, "old_text": INCLUDE, "new_text": _include("shelf_totals")}),
        ("validate_app", {}),
        (_done(1, "backend/routers/shelf_stats.py", "def shelf_stats(") if close_last_task
         else ("read_file", {"path": "backend/routers/shelf_stats.py"})),
        ("read_file", {"path": "backend/routers/shelf_totals.py"}),
        ("validate_app", {}),
    ]


def _no_app_boot(monkeypatch, issues=()):
    """The boot-and-create probe, replaced: a subprocess per validation, and its findings scripted."""
    monkeypatch.setattr(LLMOrchestrator, "_runtime_probe_issues", lambda self: list(issues))


def _tree(root: str) -> dict[str, str]:
    files = {}
    for folder, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for name in names:
            if not name.startswith("."):
                path = os.path.join(folder, name)
                with open(path, "rb") as handle:
                    files[os.path.relpath(path, root).replace("\\", "/")] = hashlib.sha256(handle.read()).hexdigest()
    return files


def _run(workspace, script, calls_per_turn, **kwargs) -> tuple[ScriptedAgent, dict]:
    """Run ``script``; return the agent and what Phase 3 started from."""
    entry: dict = {}

    def on_findings(attempt, issues, restored=False):
        if attempt == 0:
            entry["tree"] = _tree(str(workspace))
            entry["open"] = sorted(t["text"] for t in agent.orchestrator.executor.open_tasks())
            entry["blockers"] = sorted(i.message for i in issues if i.severity == "blocker")

    agent = ScriptedAgent(str(workspace), script, TASKS, calls_per_turn=calls_per_turn,
                          **{k: kwargs.pop(k) for k in ("phase3_calls", "clock", "seconds_per_turn", "rules")
                             if k in kwargs})
    run_scripted(workspace, agent, on_findings=on_findings, **kwargs)
    return agent, entry


@pytest.fixture(scope="module")
def by_batch_size(tmp_path_factory):
    """The shelf script at one, two and four calls per turn: {n: (agent, Phase 3 entry)}."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        _no_app_boot(monkeypatch)
        root = tmp_path_factory.mktemp("batch")
        return {n: _run(root / f"n{n}", shelf_script(), n, max_turns=MAX_TURNS) for n in (1, 2, 4)}


# ----------------------------------------------------------------------
# (a) The work done does not depend on how many calls a model makes per turn
# ----------------------------------------------------------------------

def test_the_rig_runs_every_scripted_call_when_the_budget_allows(by_batch_size):
    """Two and four calls per turn fit the cap: same files, nothing left for Phase 3."""
    for n in (2, 4):
        agent, entry = by_batch_size[n]
        assert sum(map(len, agent.sent["phase2"])) == 16
        assert agent.orchestrator._phase2_stop_reason == "completed"
        assert entry["open"] == [] and entry["blockers"] == []
    assert by_batch_size[2][1]["tree"] == by_batch_size[4][1]["tree"]
    assert {"backend/routers/shelf_stats.py", "backend/routers/shelf_totals.py"} <= set(by_batch_size[4][1]["tree"])


def test_one_two_or_four_calls_per_turn_do_the_same_work(by_batch_size):
    reference = by_batch_size[4][1]
    for n in (1, 2):
        agent, entry = by_batch_size[n]
        assert sum(map(len, agent.sent["phase2"])) == 16, f"{n} call(s) per turn"
        assert entry["tree"] == reference["tree"], f"{n} call(s) per turn"
        assert (entry["open"], entry["blockers"]) == (reference["open"], reference["blockers"])


# ----------------------------------------------------------------------
# (b) Phase 3 keeps a share of the wall clock
# ----------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "Decision needed: Phase 2 now stops at its share of the clock and Phase 3 validates with "
    "time left (test_run_budget.py), but no repair turn follows: the open checklist items are "
    "'task unverified' warnings, not blockers, and the audit rules out promoting them. Whether "
    "a budget-cut Phase 2's open items start a repair attempt is the owner's call."))
def test_phase3_gets_time_when_phase2_spends_the_runtime_cap(tmp_path, monkeypatch):
    _no_app_boot(monkeypatch)
    clock = FakeClock()
    monkeypatch.setattr(orchestrator_module, "time", clock)
    monkeypatch.setattr(phase3_module, "time", clock)
    agent, entry = _run(tmp_path, shelf_script(), 1, clock=clock, seconds_per_turn=60.0,
                        max_runtime_seconds=600)
    orchestrator = agent.orchestrator
    assert orchestrator._phase2_stop_reason == "timeout"
    assert not orchestrator._phase3_exit_reason.startswith("skipped")
    assert entry, "Phase 3 ran no validation round"
    assert agent.sent["phase3"], "Phase 3 made no repair turn"


# ----------------------------------------------------------------------
# (c) A repair attempt is not forced to edit when no blocker names a file
# ----------------------------------------------------------------------

# The probe's own wording for an observed failed create; it names the backend
# folder and the route, never a file.
CREATE_500 = (
    "create contract: backend: POST /book/ - observed a server/persistence failure creating "
    'Book (500: Internal Server Error); attempted input: {"pages": 1, "title": "x"}. This is '
    "one observed failure, not proof that every valid request fails. Preserve business "
    "constraints; handle invalid input without an unhandled server failure"
)


def test_a_repair_attempt_is_not_forced_to_edit_when_no_blocker_names_a_file(tmp_path, monkeypatch):
    _no_app_boot(monkeypatch, [CREATE_500])
    phase3 = [("read_file", {"path": BOOK_ROUTER}), ("read_file", {"path": "backend/sql_alchemy.py"})]
    agent, entry = _run(tmp_path, shelf_script(), 4, phase3_calls=phase3)
    assert entry["blockers"] == [CREATE_500]
    assert agent.sent["phase3"], "the repair loop never ran"
    assert not EDIT_TOOLS & set(agent.forced["phase3"])


# ----------------------------------------------------------------------
# (d) Only unverified findings start no repair attempt
# ----------------------------------------------------------------------

def test_only_unverified_findings_start_no_repair_attempt(tmp_path, monkeypatch):
    """No observed failure and one open checklist item: it is reported as a warning, not repaired."""
    _no_app_boot(monkeypatch)
    agent, entry = _run(tmp_path, shelf_script(close_last_task=False), 4,
                        phase3_calls=[("read_file", {"path": "backend/routers/shelf_stats.py"})])
    assert entry["open"] == [TASKS[1]] and entry["blockers"] == []
    assert [i.severity for i in agent.orchestrator._validation_issues
            if i.message.startswith("task unverified:")] == ["warning"]
    assert not agent.sent["phase3"]


# ----------------------------------------------------------------------
# The rig's own records: metrics and replay
# ----------------------------------------------------------------------

def test_trace_metrics_read_the_run_the_rig_made(by_batch_size):
    for n in (2, 4):
        [run] = load_runs(by_batch_size[n][0].orchestrator.output_dir)
        m = metrics(run)
        assert m["calls_per_turn"][0] == n
        assert m["turns"] == (16 // n, 0)
        assert m["lossy"] == 0 and sum(m["refused"].values()) == 0
        assert m["task_list_share"] == pytest.approx(2 / n / (16 // n))


def test_a_rule_answers_a_refused_edit(tmp_path, monkeypatch):
    _no_app_boot(monkeypatch)
    wrong = ("modify_file", {"path": BOOK_ROUTER, "old_text": "def no_such_function(", "new_text": "x"})
    retry = [("read_file", {"path": BOOK_ROUTER}), shelf_script()[4]]
    script = [call for index, call in enumerate(shelf_script()) if index != 4]
    script.insert(4, wrong)
    agent, entry = _run(tmp_path, script, 2, rules=[Rule("modify_file", "old_text not found", retry)])
    assert agent.rules[0].fired
    assert [call for batch in agent.sent["phase2"] for call in batch][6:8] == retry
    assert entry["open"] == [] and entry["blockers"] == []
    assert "def book_stats(" in (tmp_path / BOOK_ROUTER).read_text(encoding="utf-8")


def test_a_recorded_run_replays_at_another_batch_size(by_batch_size, tmp_path, monkeypatch):
    """The run log of the four-per-turn run, replayed one call per turn under a larger cap, writes the same tree."""
    _no_app_boot(monkeypatch)
    phase2, phase3, lossy = script_from_trace(
        os.path.join(by_batch_size[4][0].orchestrator.output_dir, ".besser_trace.jsonl"))
    assert (len(phase2), phase3, lossy) == (16, [], 0)
    agent, entry = _run(tmp_path, phase2, 1, max_turns=40)
    assert entry["tree"] == by_batch_size[4][1]["tree"]
    assert entry["open"] == [] and entry["blockers"] == []


REPLAY_FOLDER = os.environ.get("BESSER_REPLAY_TRACES")


@pytest.mark.skipif(not REPLAY_FOLDER, reason="set BESSER_REPLAY_TRACES to a folder of recorded run logs")
def test_replay_recorded_runs(tmp_path, monkeypatch, capsys):
    """Opt-in: each recorded run's Phase 2 calls, replayed on the shelf scaffold at 1, 2 and 4 per turn.

    The recorded workspace is not rebuilt, so paths the recorded run read or
    edited are mostly absent here: this measures the loop (calls executed
    before the budget ends, turns used), not the app.
    """
    _no_app_boot(monkeypatch)
    rows = []
    for index, run in enumerate(load_runs(REPLAY_FOLDER)):
        phase2 = [(c.tool, c.input) for c in run.calls if c.phase in ("phase2", None)]
        if not phase2:
            continue
        executed, sizes = {}, {}
        for n in (1, 2, 4):
            agent, _ = _run(tmp_path / f"r{index}n{n}", list(phase2), n)
            executed[n] = (sum(map(len, agent.sent["phase2"])), agent.orchestrator._phase2_stop_reason,
                           agent.orchestrator._phase2_stop_detail)
            # Characters sent per Phase 2 request: a provider-free measure of what the loop re-sends.
            per_turn = agent.transport.request_chars("phase2")
            per_request = [sum(row.values()) for row in per_turn]
            sizes[n] = (sum(per_request), max(per_request, default=0))
        rows.append(f"run {run.run_id[:8]}: {len(phase2)} Phase 2 calls; executed (stop) at 1/2/4 "
                    f"per turn: {executed}; request chars (total, largest): {sizes}")
        rows.extend(f"  turn {turn} at 4 per turn: system {row['system']:,}, tools {row['tools']:,}, "
                    f"messages {row['messages']:,}" for turn, row in enumerate(per_turn, 1))
        assert executed[4][0] > 0
    with capsys.disabled():
        print("\n" + "\n".join(rows))
