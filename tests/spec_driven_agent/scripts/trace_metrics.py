"""Structural metrics of Spec-Driven runs, read from their run logs and recipes.

Usage::

    python tests/spec_driven_agent/scripts/trace_metrics.py PATH [PATH ...]

A PATH is a run log (``*.trace.jsonl`` / ``.besser_trace.jsonl``), a recipe
(``*.recipe.json`` / ``.besser_recipe.json``) or a run workspace holding them,
or a folder searched for any of these. A run log holding several runs (a
create run followed by modify runs in the same workspace) gives one row per
run; a recipe next to a run log describes its last run.

Columns:

* ``calls/turn`` - tool calls per turn that made any, Phase 2 / Phase 3;
* ``1st FE write`` - Phase 2 turn of the first successful write under a
  frontend folder or to a UI source file;
* ``verify`` / ``tasks`` - share of tool-calling turns spent on verification
  tools and on ``task_list`` (a turn's calls split it: one of two calls is half
  a turn);
* ``refused`` - refused write calls; their reasons are listed below the table;
* ``off-target`` - repair attempts that edited a file no blocker names. The run
  log keeps a count of each attempt's blockers, not their text, so the
  blockers left at the end of the run stand in for them;
* ``lossy`` - tool calls whose recorded input was cut (``<<truncated`` in the
  run log, keys dropped in a recipe): such a call cannot be replayed as made.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

WRITE_TOOLS = frozenset({"write_file", "modify_file", "replace_file_lines", "delete_file"})
VERIFICATION_TOOLS = frozenset({"validate_app", "test_api", "check_syntax", "validate_model",
                                "run_command", "install_dependencies"})
_FRONTEND_EXTENSIONS = (".tsx", ".jsx", ".vue", ".svelte", ".html", ".css")
_CUT_MARKER = "<<truncated"
_CUT_RE = re.compile(r"\n?<<truncated \d+ of \d+ chars, sha256 [0-9a-f]+>>$")
_TRACE_SUFFIXES = (".trace.jsonl", ".besser_trace.jsonl")
_RECIPE_SUFFIXES = (".recipe.json", ".besser_recipe.json")


@dataclass
class Call:
    turn: int
    phase: str | None     # "phase2", "phase3" (a repair attempt), or None when unknown
    tool: str
    input: dict
    success: bool
    error: str = ""
    attempt: int | None = None
    lossy: bool = False


@dataclass
class Run:
    source: str
    run_id: str = ""
    calls: list[Call] = field(default_factory=list)
    blockers_at_end: list[str] | None = None
    calls_dropped: int = 0      # a recipe over its size budget drops calls


def _is_cut(value) -> bool:
    return _CUT_MARKER in json.dumps(value, default=str)


def _uncut(value):
    """``value`` with the run log's cut markers removed (the kept prefix stays)."""
    if isinstance(value, str):
        return _CUT_RE.sub("", value)
    if isinstance(value, dict):
        return {key: _uncut(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_uncut(item) for item in value]
    return value


def _error_of(payload: dict) -> str:
    excerpt = (payload.get("result") or {}).get("excerpt") or ""
    try:
        return str(json.loads(excerpt).get("error") or "")
    except (ValueError, AttributeError):
        match = re.search(r'"error":\s*"((?:[^"\\]|\\.)*)', excerpt)
        return match.group(1) if match else ""


def read_trace(path: str) -> list[Run]:
    """One :class:`Run` per ``run_start`` in a run log."""
    runs: list[Run] = []
    phase, attempt = None, None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind, payload = event.get("event"), event.get("payload") or {}
            if kind == "run_start" or not runs:
                runs.append(Run(source=path, run_id=str(event.get("run_id") or "")))
                phase, attempt = None, None
                if kind == "run_start":
                    continue
            if kind == "phase_enter":
                name = payload.get("phase", "")
                if name.startswith("phase2"):
                    phase, attempt = "phase2", None
                elif name == "phase3_fix_attempt":
                    phase, attempt = "phase3", payload.get("attempt")
                elif name == "phase3":
                    phase, attempt = "phase3_validation", None
            elif kind == "tool_call":
                raw = payload.get("input") if isinstance(payload.get("input"), dict) else {}
                runs[-1].calls.append(Call(
                    turn=int(payload.get("turn") or 0), phase=phase, tool=payload.get("tool", ""),
                    input=_uncut(raw), success=bool(payload.get("success")),
                    error="" if payload.get("success") else _error_of(payload),
                    attempt=attempt, lossy=_is_cut(raw),
                ))
            elif kind == "phase_exit" and payload.get("phase") == "phase3":
                phase, attempt = None, None
    return runs


def _recipe_input_cut(tool: str, data: dict) -> bool:
    """A recipe drops over-budget input values; a write without its text was cut."""
    if tool == "write_file":
        return "content" not in data
    if tool == "modify_file":
        return "old_text" not in data or "new_text" not in data
    if tool == "replace_file_lines":
        return "new_text" not in data
    return False


def read_recipe(path: str) -> tuple[Run, list[str]]:
    with open(path, encoding="utf-8") as handle:
        recipe = json.load(handle)
    run = Run(source=path)
    for entry in recipe.get("tool_calls") or []:
        data = entry.get("input") if isinstance(entry.get("input"), dict) else {}
        run.calls.append(Call(
            turn=int(entry.get("turn") or 0), phase=None, tool=entry.get("tool", ""), input=data,
            success=bool(entry.get("success")), lossy=_recipe_input_cut(entry.get("tool", ""), data),
        ))
    run.calls_dropped = int((recipe.get("recipe_shortened") or {}).get("tool_calls_dropped") or 0)
    blockers = [issue.get("message", "") for issue in recipe.get("validation_issues") or []
                if isinstance(issue, dict) and issue.get("severity") == "blocker"]
    return run, blockers


def _sibling_recipe(trace_path: str) -> str | None:
    folder, name = os.path.split(trace_path)
    for candidate in (name.replace(".trace.jsonl", ".recipe.json"), ".besser_recipe.json"):
        path = os.path.join(folder, candidate)
        if candidate != name and os.path.isfile(path):
            return path
    return None


def load_runs(path: str) -> list[Run]:
    """Every run recorded at ``path`` (a file, a workspace or a folder of them)."""
    if os.path.isdir(path):
        found: list[Run] = []
        names = sorted(os.listdir(path))
        traces = [n for n in names if n.endswith(_TRACE_SUFFIXES)]
        recipes = [n for n in names if n.endswith(_RECIPE_SUFFIXES)]
        for name in traces:
            found.extend(load_runs(os.path.join(path, name)))
        covered = {n.replace(".trace.jsonl", ".recipe.json") for n in traces}
        if ".besser_trace.jsonl" in traces:
            covered.add(".besser_recipe.json")
        for name in recipes:
            if name not in covered:
                found.extend(load_runs(os.path.join(path, name)))
        for name in names:
            sub = os.path.join(path, name)
            if os.path.isdir(sub) and name not in ("node_modules", ".git", ".besser_snapshot"):
                found.extend(load_runs(sub))
        return found
    if path.endswith(_RECIPE_SUFFIXES):
        run, blockers = read_recipe(path)
        run.blockers_at_end = blockers
        return [run]
    runs = read_trace(path)
    recipe = _sibling_recipe(path)
    if recipe and runs:
        _, runs[-1].blockers_at_end = read_recipe(recipe)
    return runs


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def _path(call: Call) -> str:
    return str(call.input.get("path") or "").replace("\\", "/")


def is_frontend_path(path: str) -> bool:
    return "frontend" in path.split("/") or path.endswith(_FRONTEND_EXTENSIONS)


def refusal_reason(error: str) -> str:
    """The first clause of a refusal, with paths and numbers generalised."""
    if not error:
        return "(not recorded)"
    text = re.split(r"\. | - | — |; |\n", error, maxsplit=1)[0]
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "<path>", text)
    text = re.sub(r"\S+/\S+|\S+\.(?:py|tsx?|jsx?|json|css|html|txt|md)\b", "<path>", text)
    return re.sub(r"\d+", "N", text).strip()[:80]


def blocker_prefix(message: str) -> str:
    match = re.match(r"\s*([A-Za-z][\w .-]*?)\s*(?:\[|:)", message)
    return match.group(1).lower() if match else message[:24].lower()


def _turns(calls: list[Call]) -> dict[int, list[Call]]:
    turns: dict[int, list[Call]] = {}
    for call in calls:
        turns.setdefault(call.turn, []).append(call)
    return turns


def _share(turns: dict[int, list[Call]], tools) -> float | None:
    if not turns:
        return None
    return sum(sum(c.tool in tools for c in calls) / len(calls) for calls in turns.values()) / len(turns)


def metrics(run: Run) -> dict:
    known = any(call.phase for call in run.calls)
    phase2 = [c for c in run.calls if c.phase == "phase2"] if known else run.calls
    phase3 = [c for c in run.calls if c.phase == "phase3"]
    turns2, turns3 = _turns(phase2), _turns(phase3)
    frontend = [c.turn for c in phase2 if c.tool in WRITE_TOOLS and c.success and is_frontend_path(_path(c))]
    first_turn = min(turns2) if turns2 else 0
    refusals = Counter(refusal_reason(c.error) for c in run.calls
                       if c.tool in WRITE_TOOLS and not c.success)
    off_target = None
    if run.blockers_at_end is not None and known:
        named = " ".join(run.blockers_at_end)
        edited: dict[int, set[str]] = {}
        for call in phase3:
            if call.tool in WRITE_TOOLS and call.success and call.attempt is not None:
                edited.setdefault(call.attempt, set()).add(_path(call))
        off_target = (sum(1 for paths in edited.values()
                          if any(p and p not in named and os.path.basename(p) not in named for p in paths)),
                      len({c.attempt for c in phase3 if c.attempt is not None}))
    return {
        "calls_per_turn": (len(phase2) / len(turns2) if turns2 else None,
                           len(phase3) / len(turns3) if turns3 else None),
        "turns": (len(turns2), len(turns3)),
        "first_frontend_write": (min(frontend) - first_turn + 1) if frontend else None,
        "verification_share": _share(turns2 | turns3, VERIFICATION_TOOLS),
        "task_list_share": _share(turns2 | turns3, {"task_list"}),
        "refused": refusals,
        "off_target_attempts": off_target,
        "blockers_by_prefix": Counter(blocker_prefix(m) for m in run.blockers_at_end or []),
        "lossy": sum(c.lossy for c in run.calls) + run.calls_dropped,
        "calls": len(run.calls) + run.calls_dropped,
        "phases_known": known,
    }


def _fmt(value, pattern="{:.2f}") -> str:
    return "-" if value is None else pattern.format(value)


def _label(run: Run, index: int, count: int) -> str:
    match = re.search(r"besser_llm_([0-9a-f]{8})", run.source)
    label = match.group(1) if match else os.path.basename(run.source)
    return f"{label}#{index + 1}" if count > 1 else label


def report(runs: list[Run]) -> str:
    lines = ["| run | turns P2/P3 | calls/turn P2 | P3 | 1st FE write | verify | tasks | refused "
             "| off-target | lossy/calls |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    details: list[str] = []
    by_source: dict[str, list[Run]] = {}
    for run in runs:
        by_source.setdefault(run.source, []).append(run)
    for source_runs in by_source.values():
        for index, run in enumerate(source_runs):
            m = metrics(run)
            label = _label(run, index, len(source_runs))
            off = m["off_target_attempts"]
            turns = f"{m['turns'][0]}/{m['turns'][1]}" if m["phases_known"] else f"{m['turns'][0]}/?"
            lines.append(
                f"| {label} | {turns} | {_fmt(m['calls_per_turn'][0])} | {_fmt(m['calls_per_turn'][1])} "
                f"| {_fmt(m['first_frontend_write'], '{}')} | {_fmt(m['verification_share'], '{:.0%}')} "
                f"| {_fmt(m['task_list_share'], '{:.0%}')} | {sum(m['refused'].values())} "
                f"| {'-' if off is None else f'{off[0]} of {off[1]}'} | {m['lossy']}/{m['calls']} |")
            if m["refused"]:
                details.append(f"- {label} refused: " + "; ".join(
                    f"{reason} x{n}" for reason, n in m["refused"].most_common(4)))
            if m["blockers_by_prefix"]:
                details.append(f"- {label} blockers at end: " + ", ".join(
                    f"{prefix} {n}" for prefix, n in m["blockers_by_prefix"].most_common()))
    return "\n".join(lines + [""] + details)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    runs: list[Run] = []
    seen: set[str] = set()
    # A modify run's workspace carries the run log of the runs before it.
    for run in (run for path in argv for run in load_runs(path)):
        if run.run_id and run.run_id in seen:
            continue
        seen.add(run.run_id)
        runs.append(run)
    if not runs:
        print("no run log or recipe found")
        return 1
    print(report(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
