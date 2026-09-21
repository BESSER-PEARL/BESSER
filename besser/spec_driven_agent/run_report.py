"""Summarize one Spec-Driven Agent run from the artefacts it already writes.

Every run's output directory accumulates four files as it goes:
``.besser_trace.jsonl`` (tracing.py), ``.besser_recipe.json`` (written once,
at a clean finish, by the orchestrator), ``.besser_tool_inputs.jsonl`` (the
untruncated inputs of write-tool calls only), and ``.besser_checkpoint.json``
(checkpoint.py; present only while a run is resumable). Answering "what
happened in this run?" today means reading these by hand. This module reads
them back and reports: which phases ran and how they ended, tool-call counts
and success rates per tool (edit tools called out), why edits were refused
grouped by reason, whether Phase 3 rolled back a repair and what that
discarded, validation issues by severity and message prefix, and cost/turns.

Formalizes ``verification/analyse_iteration.py`` (hand-written because the
package offered nothing) and fixes the two things it got wrong: it raised on
a truncated final JSONL line instead of skipping it (a real trace can have an
unparseable line anywhere, not only at eof - observed on a torn write), and
it reported ``max(turn_start payload)`` as the turn count. That undercounts:
only Phase 2's loop emits ``turn_start``; Phase 3's fix-loop turns advance
the same ``self.total_turns`` counter (confirmed against
``besser/spec_driven_agent/orchestrator.py``) but never emit their own
``turn_start`` event (each fix *attempt* resets its own local turn variable
for its internal cap/nudge logic, but that variable is never traced). A real
run's trace showed ``turn_start`` topping out at 34 while the run's own
``total_turns`` bookkeeping - and its ``tool_call`` events' ``turn`` field -
went to 108, then 147 across a resume. This module instead takes the max
``turn``/``turns``/``total_turns`` value across every event and artefact
that reports one, which is never lower than any single source.

Contract: read-only (never writes to the run directory), tolerant (a missing
or corrupt artefact degrades to "not available", a truncated JSONL file
degrades line-by-line), no new dependencies.

Importable::

    from besser.spec_driven_agent.run_report import build_report, format_report
    report = build_report(run_dir)   # plain, JSON-serializable dict
    print(format_report(report))

Runnable::

    python -m besser.spec_driven_agent.run_report <run_dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any

from besser.spec_driven_agent.tracing import TRACE_FILENAME

# orchestrator.py defines this alongside ``.besser_recipe.json`` but is a
# heavy module (provider SDK imports); this report has to stay usable
# without those installed, so the filenames are literals here too, exactly
# as ``verification/analyse_iteration.py`` already does.
RECIPE_FILENAME = ".besser_recipe.json"
TOOL_INPUTS_FILENAME = ".besser_tool_inputs.jsonl"

EDIT_TOOLS = ("modify_file", "replace_file_lines", "write_file")
_MAX_PREFIX_GROUPS = 20  # bound the "validation issues by prefix" listing


# ---------------------------------------------------------------------------
# Tolerant artefact readers
# ---------------------------------------------------------------------------

def _read_json(path: str) -> tuple[Any, str | None]:
    """Whole-file JSON. Returns ``(data, None)`` or ``(None, error)``.

    Unlike the JSONL readers, a malformed JSON *document* cannot be salvaged
    line-by-line, so any failure here degrades the whole artefact.
    """
    if not os.path.isfile(path):
        return None, "not found"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return json.load(fh), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _read_jsonl(path: str) -> tuple[list[dict], int, str | None]:
    """Line-delimited JSON. Returns ``(records, skipped_count, error)``.

    Each line is parsed independently: a single truncated or torn-write line
    (observed mid-file, not only at eof, in real traces) is skipped, not
    fatal to the rest of the file.
    """
    if not os.path.isfile(path):
        return [], 0, "not found"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()
    except Exception as exc:
        return [], 0, f"{type(exc).__name__}: {exc}"
    records: list[dict] = []
    skipped = 0
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            skipped += 1
    return records, skipped, None


def _load_checkpoint(run_dir: str):
    """Delegate to ``checkpoint.load_checkpoint`` - it already knows the
    schema, its defaults and its version compatibility rules; duplicating
    that here would drift. Returns ``(checkpoint_or_None, error_or_None)``."""
    path = os.path.join(run_dir, ".besser_checkpoint.json")
    if not os.path.isfile(path):
        return None, "not found (expected on a clean run - deleted after a clean Phase 2 finish)"
    try:
        from besser.spec_driven_agent.checkpoint import load_checkpoint
        checkpoint = load_checkpoint(run_dir)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if checkpoint is None:
        return None, "present but unreadable or an unrecognised schema"
    return checkpoint, None


def _load_artifacts(run_dir: str) -> dict:
    trace_events, trace_skipped, trace_err = _read_jsonl(os.path.join(run_dir, TRACE_FILENAME))
    tool_inputs, ti_skipped, ti_err = _read_jsonl(os.path.join(run_dir, TOOL_INPUTS_FILENAME))
    recipe, recipe_err = _read_json(os.path.join(run_dir, RECIPE_FILENAME))
    checkpoint, checkpoint_err = _load_checkpoint(run_dir)
    return {
        "trace_events": trace_events, "trace_skipped": trace_skipped, "trace_error": trace_err,
        "tool_inputs": tool_inputs, "tool_inputs_skipped": ti_skipped, "tool_inputs_error": ti_err,
        "recipe": recipe, "recipe_error": recipe_err,
        "checkpoint": checkpoint, "checkpoint_error": checkpoint_err,
    }


# ---------------------------------------------------------------------------
# Per-facet analysis (pure functions over already-parsed data)
# ---------------------------------------------------------------------------

def _phase_summary(events: list[dict]) -> dict:
    """Which phases ran and how they ended.

    ``phase3_fix_attempt`` (Phase 3's own bounded repair loop - up to
    ``max(5, max_turns - total_turns)`` rounds) is tracked separately from
    the top-level phases rather than as one of them.
    """
    pending: dict[Any, list[dict]] = collections.defaultdict(list)
    pending_attempts: dict[Any, dict] = {}
    completed: list[dict] = []
    attempts: list[dict] = []

    for event in events:
        payload = event.get("payload") or {}
        phase = payload.get("phase")
        name = event.get("event")
        if name == "phase_enter":
            if phase == "phase3_fix_attempt":
                pending_attempts[payload.get("attempt")] = payload
            else:
                pending[phase].append(payload)
        elif name == "phase_exit":
            if phase == "phase3_fix_attempt":
                enter = pending_attempts.pop(payload.get("attempt"), None)
                attempts.append({"attempt": payload.get("attempt"), "enter": enter, "exit": payload})
            elif pending[phase]:
                completed.append({"phase": phase, "enter": pending[phase].pop(0), "exit": payload})
            else:
                completed.append({"phase": phase, "enter": None, "exit": payload})

    open_phases = [{"phase": phase, "enter": enter}
                   for phase, entries in pending.items() for enter in entries]
    open_phases += [{"phase": "phase3_fix_attempt", "enter": enter}
                     for enter in pending_attempts.values()]
    return {"completed": completed, "open": open_phases, "repair_attempts": attempts}


def _tool_call_summary(events: list[dict]) -> dict:
    calls = [e.get("payload") or {} for e in events if e.get("event") == "tool_call"]
    per_tool: dict[str, dict] = {}
    for call in calls:
        entry = per_tool.setdefault(call.get("tool") or "?", {"calls": 0, "success": 0})
        entry["calls"] += 1
        entry["success"] += 1 if call.get("success") else 0
    edit_calls = [c for c in calls if c.get("tool") in EDIT_TOOLS]
    edit_landed = [c for c in edit_calls if c.get("success")]
    return {
        "per_tool": per_tool,
        "total_calls": len(calls),
        "edit_attempts": len(edit_calls),
        "edit_landed": len(edit_landed),
    }


def _edit_refusal_summary(events: list[dict]) -> dict:
    """``refusals_by_reason`` is ``{tool: {reason: count}}`` - a plain,
    JSON-serializable nested dict rather than a tuple-keyed Counter."""
    calls = [e.get("payload") or {} for e in events if e.get("event") == "tool_call"]
    edit_calls = [c for c in calls if c.get("tool") in EDIT_TOOLS]
    reasons: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for call in edit_calls:
        if call.get("success"):
            continue
        reason = call.get("rejection_kind") or call.get("status") or "?"
        reasons[call.get("tool") or "?"][reason] += 1
    recovery_fired = sum(1 for c in calls if c.get("edit_recovery"))
    return {"refusals_by_reason": {tool: dict(counts) for tool, counts in reasons.items()},
            "recovery_ladder_fired": recovery_fired}


def _describe_rollback(payload: dict) -> str:
    """One sentence from whatever keys are present. Two schema variants have
    been observed in real traces (a ``hard_blockers_*`` + a plain
    ``blockers_*`` form); this reads either without assuming which."""
    def pick(*keys):
        return next((payload[k] for k in keys if k in payload), None)

    phase = payload.get("phase", "?")
    entry = pick("hard_blockers_on_entry", "blockers_on_entry")
    after_repair = pick("hard_blockers_after_repair", "blockers_after_repair")
    after_rollback = pick("hard_blockers_after_rollback", "blockers_after_rollback")
    total_after = payload.get("total_blockers_after_rollback")

    parts = [f"rollback fired during {phase}"]
    if entry is not None and after_repair is not None:
        parts.append(f"the repair took blockers from {entry} to {after_repair}")
    if after_rollback is not None:
        parts.append(f"discarded that repair, reverting to {after_rollback}")
    if total_after is not None and total_after != after_rollback:
        parts.append(f"{total_after} total finding(s) remain including non-blockers")
    return "; ".join(parts)


def _rollback_summary(events: list[dict], recipe: Any) -> dict:
    rollbacks = [e.get("payload") or {} for e in events if e.get("event") == "rollback"]
    recipe_flag = recipe.get("phase3_rolled_back") if isinstance(recipe, dict) else None
    return {
        "fired": bool(rollbacks) or recipe_flag is True,
        "events": rollbacks,
        "recipe_flag": recipe_flag,
    }


def _validation_summary(recipe: Any, checkpoint: Any) -> dict:
    """Prefer the recipe's ``validation_issues``; fall back to the
    checkpoint's copy when there is no recipe (an interrupted run)."""
    issues, source = None, None
    if isinstance(recipe, dict) and isinstance(recipe.get("validation_issues"), list):
        issues, source = recipe["validation_issues"], "recipe"
    elif checkpoint is not None and isinstance(getattr(checkpoint, "validation_issues", None), list):
        issues, source = checkpoint.validation_issues, "checkpoint (no recipe - run likely interrupted)"

    if issues is None:
        return {"available": False, "source": None, "total": 0,
                "severity_totals": collections.Counter(), "by_severity": {}}

    by_severity: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    severity_totals: collections.Counter = collections.Counter()
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        severity = issue.get("severity", "?")
        prefix = str(issue.get("message", "")).split(":")[0][:40]
        by_severity[severity][prefix] += 1
        severity_totals[severity] += 1
    return {"available": True, "source": source, "total": len(issues),
            "severity_totals": severity_totals, "by_severity": dict(by_severity)}


def _cost_and_turns(events: list[dict], recipe: Any, checkpoint: Any) -> dict:
    """The true turn count is the max of every source that reports one, not
    ``max(turn_start)`` alone - see the module docstring for why that
    undercounts once Phase 3 has run."""
    turn_candidates: list[int] = []
    cost_candidates: list[float] = []
    for event in events:
        payload = event.get("payload") or {}
        for key in ("turn", "turns", "total_turns"):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                turn_candidates.append(value)
        if event.get("event") == "cost_update":
            value = payload.get("estimated_cost_usd")
            if isinstance(value, (int, float)):
                cost_candidates.append(value)

    usage = None
    if isinstance(recipe, dict):
        value = recipe.get("turns")
        if isinstance(value, int) and not isinstance(value, bool):
            turn_candidates.append(value)
        if isinstance(recipe.get("usage"), dict):
            usage = recipe["usage"]
            value = usage.get("estimated_cost_usd")
            if isinstance(value, (int, float)):
                cost_candidates.append(value)
    if checkpoint is not None:
        value = getattr(checkpoint, "total_turns", None)
        if isinstance(value, int) and not isinstance(value, bool):
            turn_candidates.append(value)
        value = getattr(checkpoint, "estimated_cost_usd", None)
        if isinstance(value, (int, float)):
            cost_candidates.append(value)

    return {
        "turns": max(turn_candidates) if turn_candidates else None,
        "estimated_cost_usd": max(cost_candidates) if cost_candidates else None,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Assembly and formatting
# ---------------------------------------------------------------------------

def build_report(run_dir: str) -> dict:
    """Read every artefact under ``run_dir`` and return one plain,
    JSON-serializable dict. Never raises: a missing directory or artefact
    simply yields "not available" fields throughout."""
    run_dir = os.fspath(run_dir)
    artifacts = _load_artifacts(run_dir)
    events = artifacts["trace_events"]
    recipe = artifacts["recipe"]
    checkpoint = artifacts["checkpoint"]
    run_ids = sorted({e["run_id"] for e in events if e.get("run_id")})

    return {
        "run_dir": run_dir,
        "artifacts": {
            "trace": {"available": artifacts["trace_error"] is None, "error": artifacts["trace_error"],
                      "event_count": len(events), "skipped_lines": artifacts["trace_skipped"]},
            "recipe": {"available": recipe is not None, "error": artifacts["recipe_error"]},
            "tool_inputs": {"available": artifacts["tool_inputs_error"] is None,
                             "error": artifacts["tool_inputs_error"],
                             "record_count": len(artifacts["tool_inputs"]),
                             "skipped_lines": artifacts["tool_inputs_skipped"]},
            "checkpoint": {"available": checkpoint is not None, "error": artifacts["checkpoint_error"]},
        },
        "run_ids": run_ids,
        "phases": _phase_summary(events),
        "tool_calls": _tool_call_summary(events),
        "edit_refusals": _edit_refusal_summary(events),
        "rollback": _rollback_summary(events, recipe),
        "validation": _validation_summary(recipe, checkpoint),
        "cost_and_turns": _cost_and_turns(events, recipe, checkpoint),
    }


def _fmt_extra(payload: dict, *, omit: tuple[str, ...] = ()) -> str:
    shown = {k: v for k, v in payload.items() if k not in omit}
    return ", ".join(f"{k}={v}" for k, v in shown.items()) if shown else "no detail"


def _fmt_section(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


def format_report(report: dict) -> str:
    """Render :func:`build_report`'s output as readable text."""
    lines: list[str] = [f"Run directory: {report['run_dir']}"]

    art = report["artifacts"]
    status = ", ".join(
        f"{name}={'ok' if info['available'] else 'not available (' + str(info.get('error')) + ')'}"
        for name, info in art.items()
    )
    lines.append(f"Artefacts: {status}")
    skipped = art["trace"]["skipped_lines"] + art["tool_inputs"]["skipped_lines"]
    if skipped:
        lines.append(f"  skipped unparseable line(s): trace={art['trace']['skipped_lines']}, "
                      f"tool_inputs={art['tool_inputs']['skipped_lines']}")
    if len(report["run_ids"]) > 1:
        lines.append(f"  trace spans {len(report['run_ids'])} run_id(s): "
                      f"{', '.join(report['run_ids'])} (this run was resumed)")

    lines.append(_fmt_section("Phases"))
    phases = report["phases"]
    if not phases["completed"] and not phases["open"]:
        lines.append("  no phase_enter/phase_exit events (trace not available, or the run never started)")
    for span in phases["completed"]:
        lines.append(f"  {span['phase'] or '?':18} ended ({_fmt_extra(span['exit'], omit=('phase',))})")
    for entry in phases["open"]:
        lines.append(f"  {entry['phase'] or '?':18} STILL OPEN - no matching phase_exit "
                      "(the run stopped mid-phase)")
    attempts = phases["repair_attempts"]
    if attempts:
        last = attempts[-1]
        last_exit = _fmt_extra(last["exit"], omit=("phase", "attempt")) if last["exit"] else "did not exit"
        lines.append(f"  phase3_fix_attempt: {len(attempts)} round(s); last is #{last['attempt']} ({last_exit})")

    lines.append(_fmt_section("Tool calls"))
    tc = report["tool_calls"]
    if not tc["per_tool"]:
        lines.append("  no tool_call events")
    else:
        lines.append(f"  {'tool':24} {'calls':>6} {'ok':>5} {'fail':>5}  rate")
        for tool, counts in sorted(tc["per_tool"].items(), key=lambda kv: -kv[1]["calls"]):
            calls, ok = counts["calls"], counts["success"]
            mark = "  <-- edit" if tool in EDIT_TOOLS else ""
            lines.append(f"  {tool:24} {calls:6} {ok:5} {calls - ok:5}  {ok / calls:5.0%}{mark}")
        if tc["edit_attempts"]:
            rate = tc["edit_landed"] / tc["edit_attempts"]
            lines.append(f"  edit tools overall: {tc['edit_attempts']} attempt(s), "
                          f"{tc['edit_landed']} landed ({rate:.0%})")

    lines.append(_fmt_section("Edit refusals"))
    er = report["edit_refusals"]
    rows = [(tool, reason, count) for tool, reasons in er["refusals_by_reason"].items()
            for reason, count in reasons.items()]
    if not rows:
        lines.append("  no refused edits")
    else:
        for tool, reason, count in sorted(rows, key=lambda row: -row[2]):
            lines.append(f"  {tool:22} {reason:20} {count}")
    lines.append(f"  recovery ladder fired: {er['recovery_ladder_fired']} time(s)")

    lines.append(_fmt_section("Rollback"))
    rb = report["rollback"]
    if not rb["fired"]:
        lines.append("  no rollback")
    elif rb["events"]:
        for payload in rb["events"]:
            lines.append(f"  {_describe_rollback(payload)}")
    else:
        lines.append("  recipe reports phase3_rolled_back=true, but the trace has no rollback event")

    lines.append(_fmt_section("Validation issues"))
    v = report["validation"]
    if not v["available"]:
        lines.append("  not available (no recipe and no usable checkpoint)")
    elif v["total"] == 0:
        lines.append(f"  0 issues (source: {v['source']})")
    else:
        totals = ", ".join(f"{sev}={n}" for sev, n in v["severity_totals"].most_common())
        lines.append(f"  {v['total']} issue(s) from {v['source']}: {totals}")
        for severity, counter in v["by_severity"].items():
            lines.append(f"  {severity}:")
            shown = counter.most_common(_MAX_PREFIX_GROUPS)
            for prefix, count in shown:
                lines.append(f"    {prefix:42} {count}")
            remaining = len(counter) - len(shown)
            if remaining > 0:
                lines.append(f"    (+{remaining} more distinct prefix(es))")

    lines.append(_fmt_section("Cost and turns"))
    ct = report["cost_and_turns"]
    lines.append(f"  turns: {ct['turns'] if ct['turns'] is not None else 'not available'}")
    cost = f"${ct['estimated_cost_usd']:.4f}" if ct["estimated_cost_usd"] is not None else "not available"
    lines.append(f"  estimated cost: {cost}")
    if ct["usage"]:
        lines.append(f"  usage: {_fmt_extra(ct['usage'], omit=('estimated_cost_usd',))}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m besser.spec_driven_agent.run_report",
        description="Summarize a Spec-Driven Agent run from its own artefacts "
                     "(.besser_trace.jsonl, .besser_recipe.json, "
                     ".besser_tool_inputs.jsonl, .besser_checkpoint.json).",
    )
    parser.add_argument("run_dir", help="The run's output directory (where those files live)")
    args = parser.parse_args(argv)
    print(format_report(build_report(args.run_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
