"""``run_report`` must skip an unparseable JSONL line instead of raising, and
must not report ``max(turn_start payload)`` as the turn count: only Phase 2's
loop emits ``turn_start``, while Phase 3's fix-loop turns advance
``self.total_turns`` (and appear in ``tool_call`` payloads' ``turn`` field)
without ever emitting one, so that approach undercounts badly once Phase 3
has run.
"""
from __future__ import annotations

import json
from pathlib import Path

from besser.spec_driven_agent.run_report import (
    EDIT_TOOLS,
    build_report,
    format_report,
    main,
)

FIXTURE = Path(__file__).parent / "fixtures" / "run_report_fcdh0s9k"


def _event(event: str, **payload) -> dict:
    return {"ts": 0.0, "run_id": "r1", "primary_kind": "class", "event": event, "payload": payload}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _checkpoint_json(**overrides) -> dict:
    base = {
        "schema_version": 2, "run_id": "r1", "instructions": "build a thing",
        "primary_kind": "class", "turn": 5, "total_turns": 5, "messages": [],
        "tool_calls_log": [], "validation_issues": [], "inventory": "",
        "generator_used": None, "estimated_cost_usd": 0.1, "compaction_count": 0,
        "project_fingerprint": "abc", "saved_at": 0.0, "tasks": [], "api_scenarios": [],
        "phase": "phase3", "source_revision": "", "phase2_stop_reason": "max_turns",
        "phase2_exited_cleanly": False, "repair_progress": {},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------- the real fixture


def test_the_real_truncated_trace_is_handled(tmp_path):
    """The fixture's ``.besser_trace.jsonl`` (from a real run) has two
    genuinely unparseable lines mid-file (a torn write, not at eof); a plain
    ``json.loads`` per line raises on this file."""
    report = build_report(str(FIXTURE))
    art = report["artifacts"]
    assert art["trace"]["available"] is True
    assert art["trace"]["skipped_lines"] == 2
    assert art["recipe"]["available"] is True
    assert art["checkpoint"]["available"] is True
    assert art["tool_inputs"]["available"] is True

    # The recipe's own "turns" bookkeeping (85) is far above what the old
    # max(turn_start) approach reports for this same file (13, Phase 2's
    # own count) - Phase 3 ran 7 fix attempts that are invisible to
    # turn_start but still real turns.
    naive_old_way = max(
        e["payload"]["turn"] for e in _parse_trace_ignoring_bad_lines(FIXTURE / ".besser_trace.jsonl")
        if e["event"] == "turn_start"
    )
    assert naive_old_way == 13
    assert report["cost_and_turns"]["turns"] == 85
    assert report["cost_and_turns"]["turns"] != naive_old_way

    # format_report and JSON round-trip must both work over the real file.
    text = format_report(report)
    assert "Run directory:" in text
    assert "skipped unparseable line(s): trace=2" in text
    json.dumps(report)  # every field must be JSON-serializable


def _parse_trace_ignoring_bad_lines(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


# ------------------------------------------------------------- CLI / import


def test_runnable_as_a_module(tmp_path, capsys):
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Run directory:" in out
    assert "not available" in out  # empty directory: nothing to find


# ----------------------------------------------------------- tolerant reads


def test_an_empty_or_missing_directory_never_raises(tmp_path):
    report = build_report(str(tmp_path))
    for name in ("trace", "recipe", "tool_inputs", "checkpoint"):
        assert report["artifacts"][name]["available"] is False
    assert report["cost_and_turns"] == {"turns": None, "estimated_cost_usd": None, "usage": None}
    assert report["validation"]["available"] is False
    assert format_report(report)  # renders without raising
    json.dumps(report)

    missing = tmp_path / "does-not-exist-at-all"
    report2 = build_report(str(missing))
    assert report2["artifacts"]["trace"]["available"] is False
    format_report(report2)


def test_an_unparseable_line_mid_file_is_skipped_not_fatal(tmp_path):
    path = tmp_path / ".besser_trace.jsonl"
    path.write_text(
        json.dumps(_event("turn_start", turn=1)) + "\n"
        + "not json at all {{{\n"
        + json.dumps(_event("turn_start", turn=2)) + "\n",
        encoding="utf-8",
    )
    report = build_report(str(tmp_path))
    assert report["artifacts"]["trace"]["skipped_lines"] == 1
    assert report["artifacts"]["trace"]["event_count"] == 2


def test_a_malformed_recipe_json_degrades_the_whole_file_not_a_crash(tmp_path):
    (tmp_path / ".besser_recipe.json").write_text('{"turns": 4, "validation_issues": [', encoding="utf-8")
    report = build_report(str(tmp_path))
    assert report["artifacts"]["recipe"]["available"] is False
    assert "Error" in report["artifacts"]["recipe"]["error"] or "Decode" in report["artifacts"]["recipe"]["error"]


# ------------------------------------------------------------------ phases


def test_phase_spans_and_an_open_unfinished_phase(tmp_path):
    events = [
        _event("phase_enter", phase="phase1"),
        _event("phase_exit", phase="phase1", generator_used="generate_backend"),
        _event("phase_enter", phase="phase2"),
        _event("turn_start", turn=1),
        _event("turn_start", turn=2),
        _event("phase_exit", phase="phase2", turns=2),
        _event("phase_enter", phase="phase3"),
        _event("phase_enter", phase="phase3_fix_attempt", attempt=1, blockers=3),
        _event("phase_exit", phase="phase3_fix_attempt", attempt=1, blockers_remaining=1, successful_writes=1),
        _event("phase_enter", phase="phase3_fix_attempt", attempt=2, blockers=1),
        # attempt 2 never exits: the run was interrupted mid-attempt.
    ]
    _write_jsonl(tmp_path / ".besser_trace.jsonl", events)
    report = build_report(str(tmp_path))
    phases = report["phases"]

    completed_names = [s["phase"] for s in phases["completed"]]
    assert completed_names == ["phase1", "phase2"]
    open_names = [o["phase"] for o in phases["open"]]
    assert "phase3" in open_names
    assert "phase3_fix_attempt" in open_names
    assert len(phases["repair_attempts"]) == 1  # only attempt #1 completed
    assert phases["repair_attempts"][0]["attempt"] == 1

    text = format_report(report)
    assert "phase1" in text and "generator_used=generate_backend" in text
    assert "phase3              STILL OPEN" in text or "phase3" in text
    assert "STILL OPEN" in text


# -------------------------------------------------------------- tool calls


def test_tool_call_counts_and_edit_tools_called_out(tmp_path):
    events = [
        _event("tool_call", tool="read_file", success=True),
        _event("tool_call", tool="read_file", success=True),
        _event("tool_call", tool="modify_file", success=True),
        _event("tool_call", tool="modify_file", success=False, status="error"),
        _event("tool_call", tool="write_file", success=True),
    ]
    _write_jsonl(tmp_path / ".besser_trace.jsonl", events)
    report = build_report(str(tmp_path))
    tc = report["tool_calls"]
    assert tc["per_tool"]["read_file"] == {"calls": 2, "success": 2}
    assert tc["per_tool"]["modify_file"] == {"calls": 2, "success": 1}
    assert tc["edit_attempts"] == 3  # modify_file x2 + write_file x1
    assert tc["edit_landed"] == 2
    assert set(EDIT_TOOLS) == {"modify_file", "replace_file_lines", "write_file"}

    text = format_report(report)
    assert "<-- edit" in text
    assert "read_file" in text and "<-- edit" not in text.split("read_file")[1].split("\n")[0]


# ---------------------------------------------------------- edit refusals


def test_refusal_reasons_prefer_rejection_kind_then_fall_back_to_status(tmp_path):
    events = [
        _event("tool_call", tool="modify_file", success=False, status="error", rejection_kind="stale_read"),
        _event("tool_call", tool="modify_file", success=False, status="error", rejection_kind="stale_read"),
        _event("tool_call", tool="write_file", success=False, status="error"),  # no rejection_kind
        _event("tool_call", tool="modify_file", success=True),  # not a refusal
        _event("tool_call", tool="read_file", success=False, status="not_found"),  # not an edit tool
        _event("tool_call", tool="replace_file_lines", success=False, status="error",
               edit_recovery="{'next_tool': 'read_file'}"),
    ]
    _write_jsonl(tmp_path / ".besser_trace.jsonl", events)
    report = build_report(str(tmp_path))
    er = report["edit_refusals"]
    assert er["refusals_by_reason"]["modify_file"]["stale_read"] == 2
    assert er["refusals_by_reason"]["write_file"]["error"] == 1
    assert "read_file" not in er["refusals_by_reason"]  # not an edit tool
    assert er["recovery_ladder_fired"] == 1
    json.dumps(report)  # tuple-keyed Counters would break this


# --------------------------------------------------------------- rollback


def test_rollback_narrative_hard_blockers_schema(tmp_path):
    events = [_event(
        "rollback", phase="phase3", hard_blockers_on_entry=2, hard_blockers_after_repair=160,
        hard_blockers_after_rollback=2, total_blockers_after_rollback=14,
    )]
    _write_jsonl(tmp_path / ".besser_trace.jsonl", events)
    report = build_report(str(tmp_path))
    assert report["rollback"]["fired"] is True
    text = format_report(report)
    assert "rollback fired during phase3" in text
    assert "took blockers from 2 to 160" in text
    assert "discarded that repair, reverting to 2" in text
    assert "14 total finding(s) remain" in text


def test_rollback_narrative_plain_blockers_schema(tmp_path):
    """A second schema variant has been observed in real traces (no
    ``hard_`` prefix, no ``total_blockers_after_rollback``); must not crash
    or silently print nothing."""
    events = [_event("rollback", phase="phase3", blockers_on_entry=7,
                     blockers_after_repair=10, blockers_after_rollback=23)]
    _write_jsonl(tmp_path / ".besser_trace.jsonl", events)
    report = build_report(str(tmp_path))
    text = format_report(report)
    assert "took blockers from 7 to 10" in text
    assert "discarded that repair, reverting to 23" in text


def test_no_rollback_is_reported_plainly(tmp_path):
    _write_jsonl(tmp_path / ".besser_trace.jsonl", [_event("phase_enter", phase="phase1")])
    report = build_report(str(tmp_path))
    assert report["rollback"]["fired"] is False
    assert "no rollback" in format_report(report)


def test_recipe_flag_without_a_trace_event_is_still_reported(tmp_path):
    """The recipe's ``phase3_rolled_back`` is a cross-check: if the trace is
    missing the event (truncated) but the recipe still says it happened,
    that must surface, not silently read as "no rollback"."""
    _write_json(tmp_path / ".besser_recipe.json", {"phase3_rolled_back": True, "turns": 10})
    report = build_report(str(tmp_path))
    assert report["rollback"]["fired"] is True
    assert report["rollback"]["events"] == []
    text = format_report(report)
    assert "no rollback event" in text


# ----------------------------------------------------------- validation


def test_validation_grouped_by_severity_and_message_prefix(tmp_path):
    issues = [
        {"severity": "blocker", "message": "create contract: web_app/backend: POST /x/ - boom"},
        {"severity": "blocker", "message": "create contract: web_app/backend: POST /y/ - boom too"},
        {"severity": "blocker", "message": "ruff: F821 undefined name"},
        {"severity": "warning", "message": "endpoint coherence: frontend calls a route that 404s"},
        {"severity": "style", "message": "ruff: F401 unused import"},
    ]
    _write_json(tmp_path / ".besser_recipe.json", {"validation_issues": issues, "turns": 1})
    report = build_report(str(tmp_path))
    v = report["validation"]
    assert v["available"] is True
    assert v["source"] == "recipe"
    assert v["total"] == 5
    assert v["severity_totals"]["blocker"] == 3
    assert v["by_severity"]["blocker"]["create contract"] == 2
    assert v["by_severity"]["blocker"]["ruff"] == 1
    json.dumps(report)


def test_zero_issues_is_available_not_missing(tmp_path):
    """An empty ``validation_issues`` list means the source genuinely
    reported zero issues - not that no source was available. (Caught while
    writing this module: a truthiness check on the list conflated the two.)"""
    _write_json(tmp_path / ".besser_recipe.json", {"validation_issues": [], "turns": 1})
    report = build_report(str(tmp_path))
    assert report["validation"] == {
        "available": True, "source": "recipe", "total": 0,
        "severity_totals": {}, "by_severity": {},
    }
    assert "0 issues (source: recipe)" in format_report(report)


def test_validation_falls_back_to_checkpoint_when_no_recipe(tmp_path):
    ck = _checkpoint_json(validation_issues=[{"severity": "warning", "message": "acceptance: Room has no create page"}])
    _write_json(tmp_path / ".besser_checkpoint.json", ck)
    report = build_report(str(tmp_path))
    v = report["validation"]
    assert v["available"] is True
    assert "checkpoint" in v["source"]
    assert v["total"] == 1


def test_validation_not_available_without_recipe_or_checkpoint(tmp_path):
    report = build_report(str(tmp_path))
    assert report["validation"]["available"] is False
    assert "not available" in format_report(report)


# --------------------------------------------------------------- checkpoint


def test_a_present_but_schema_incompatible_checkpoint_degrades_cleanly(tmp_path):
    _write_json(tmp_path / ".besser_checkpoint.json", {"schema_version": 99, "instructions": "x"})
    report = build_report(str(tmp_path))
    assert report["artifacts"]["checkpoint"]["available"] is False
    assert "unreadable" in report["artifacts"]["checkpoint"]["error"] or \
           "unrecognised" in report["artifacts"]["checkpoint"]["error"]


# --------------------------------------------------------------- cost/turns


def test_turns_is_the_max_across_every_source_not_just_turn_start(tmp_path):
    """The documented bug: Phase 3's fix-loop turns advance total_turns and
    appear in tool_call.turn, but never emit their own turn_start event."""
    events = [
        _event("turn_start", turn=1),
        _event("turn_start", turn=2),
        _event("turn_start", turn=3),
        _event("phase_exit", phase="phase2", turns=3),
        # Phase 3: three more turns, no turn_start for any of them.
        _event("tool_call", tool="modify_file", success=True, turn=4),
        _event("tool_call", tool="validate_app", success=True, turn=5),
        _event("tool_call", tool="modify_file", success=True, turn=6),
    ]
    _write_jsonl(tmp_path / ".besser_trace.jsonl", events)
    report = build_report(str(tmp_path))
    assert report["cost_and_turns"]["turns"] == 6
    naive = max(e["payload"]["turn"] for e in events if e["event"] == "turn_start")
    assert naive == 3
    assert report["cost_and_turns"]["turns"] != naive


def test_cost_prefers_recipe_usage_then_falls_back(tmp_path):
    _write_json(tmp_path / ".besser_recipe.json",
               {"turns": 9, "usage": {"estimated_cost_usd": 1.25, "api_calls": 7}})
    report = build_report(str(tmp_path))
    assert report["cost_and_turns"]["estimated_cost_usd"] == 1.25
    assert report["cost_and_turns"]["turns"] == 9
    assert report["cost_and_turns"]["usage"]["api_calls"] == 7


def test_cost_and_turns_fall_back_to_checkpoint_when_no_recipe(tmp_path):
    ck = _checkpoint_json(total_turns=22, estimated_cost_usd=0.42)
    _write_json(tmp_path / ".besser_checkpoint.json", ck)
    report = build_report(str(tmp_path))
    assert report["cost_and_turns"]["turns"] == 22
    assert report["cost_and_turns"]["estimated_cost_usd"] == 0.42


def test_cost_and_turns_fall_back_to_trace_cost_updates(tmp_path):
    events = [_event("cost_update", turn=1, estimated_cost_usd=0.01),
              _event("cost_update", turn=2, estimated_cost_usd=0.05)]
    _write_jsonl(tmp_path / ".besser_trace.jsonl", events)
    report = build_report(str(tmp_path))
    assert report["cost_and_turns"]["estimated_cost_usd"] == 0.05
    assert report["cost_and_turns"]["turns"] == 2
