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

A second table gives, per run and phase (P2, P3 repair turns, and the calls
outside the loop), fresh input, cache-read, cache-write and output tokens, cost,
the cache hit ratio, the cache misses, and the seconds the loop's model calls
and tool batches took. A loop turn's tokens are the difference of two
``cost_update`` records less the ``llm_call`` records between them. A record
belongs to the run its run id names, also when it precedes that run's
``run_start`` (a modify run routes before it); a resumed run (a second
``run_start`` with the same run id) continues its run, on a new client whose
token counters restart while its cost was seeded with the spend so far. A cache
miss is a turn that read under 0.9 of the previous turn's prompt from the cache
although the history was not rewritten: same conversation, no ``compaction`` or
``eviction`` record, a prompt that did not shrink. A turn that
read under 0.9 of its own, smaller prompt with no such record is listed apart,
as history rewritten without a record (on a log older than those records, a
compaction). A run that never reports a cache read or
write shows ``unknown``: its provider may not report them. Totals over all runs
follow, each run counted once. Dollar figures follow the price catalog of the
day; compare tokens.
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
_COUNTERS = ("input_tokens", "cache_read_tokens", "cache_creation_tokens", "output_tokens")
CACHE_MISS_SHARE = 0.9


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
    batch_index: int | None = None


@dataclass
class Turn:
    """One loop request: its tokens and cost, and the seconds of its model call and tool batch."""
    turn: int
    phase: str | None
    fresh: int
    cache_read: int
    cache_write: int
    output: int
    cost: float
    # The first request of a conversation, or the first after its history was rewritten.
    new_context: bool = False
    model_seconds: float | None = None
    tool_seconds: float = 0.0

    @property
    def prompt(self) -> int:
        return self.fresh + self.cache_read + self.cache_write


@dataclass
class Run:
    source: str
    run_id: str = ""
    calls: list[Call] = field(default_factory=list)
    blockers_at_end: list[str] | None = None
    calls_dropped: int = 0      # a recipe over its size budget drops calls
    turns: list[Turn] = field(default_factory=list)
    aux: list[tuple[str, dict]] = field(default_factory=list)    # (purpose, llm_call payload)
    wall_seconds: float | None = None
    started: bool = False       # its run_start was read


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


class _Meter:
    """Reads a run's turns, calls outside the loop and seconds from its run-log records."""

    def __init__(self, run: Run):
        self.run = run
        self.counted = dict.fromkeys(_COUNTERS, 0)      # cumulative counters at the last cost_update
        self.counted_cost = 0.0
        self.outside = dict.fromkeys(_COUNTERS, 0)      # llm_call deltas since then
        self.outside_cost = 0.0
        self.new_context = True
        self.started: float | None = None
        self.answered: float | None = None
        self.first_ts: float | None = None
        self.wall_before = 0.0      # seconds of the parts of the run before a resume

    def read(self, kind: str, payload: dict, ts: float | None, phase: str | None) -> None:
        if kind == "run_start" and payload.get("resumed"):
            # A new client: its token counters restart; its cost was seeded with the spend so far.
            self.counted = dict.fromkeys(_COUNTERS, 0)
            self.counted_cost += self.outside_cost
            self.outside, self.outside_cost = dict.fromkeys(_COUNTERS, 0), 0.0
            self.wall_before, self.first_ts = self.run.wall_seconds or 0.0, None
        if ts is not None:
            if self.first_ts is None:
                self.first_ts = ts
            self.run.wall_seconds = self.wall_before + ts - self.first_ts
        if kind in ("compaction", "eviction") or (kind == "phase_enter" and (
                str(payload.get("phase", "")).startswith("phase2") or payload.get("phase") == "phase3_fix_attempt")):
            self.new_context = True
        elif kind == "llm_call":
            self.run.aux.append((str(payload.get("purpose") or "other"), payload))
            for key in _COUNTERS:
                self.outside[key] += int(payload.get(key) or 0)
            self.outside_cost += float(payload.get("cost_usd") or 0)
        elif kind == "turn_start":
            self.started, self.answered = ts, None
        elif kind in ("turn_end", "cost_update") and self.answered is None:
            self.answered = ts
        if kind == "cost_update":
            self._turn(payload, phase)
        elif kind == "tool_call" and ts is not None and self.answered is not None and self.run.turns \
                and self.run.turns[-1].turn == int(payload.get("turn") or 0):
            last = self.run.turns[-1]
            last.tool_seconds = max(last.tool_seconds, ts - self.answered)

    def _turn(self, payload: dict, phase: str | None) -> None:
        counters = {key: int(payload.get(key) or 0) for key in _COUNTERS}
        cost = float(payload.get("estimated_cost_usd") or 0)
        delta = {key: max(counters[key] - self.counted[key] - self.outside[key], 0) for key in _COUNTERS}
        self.run.turns.append(Turn(
            turn=int(payload.get("turn") or 0), phase=phase, fresh=delta["input_tokens"],
            cache_read=delta["cache_read_tokens"], cache_write=delta["cache_creation_tokens"],
            output=delta["output_tokens"], cost=max(cost - self.counted_cost - self.outside_cost, 0.0),
            new_context=self.new_context,
            model_seconds=None if None in (self.started, self.answered) else self.answered - self.started,
        ))
        self.counted, self.counted_cost = counters, cost
        self.outside, self.outside_cost = dict.fromkeys(_COUNTERS, 0), 0.0
        self.new_context = False


def _in_model_order(calls: list[Call]) -> list[Call]:
    """A turn's parallel calls are logged as each finishes; ``batch_index`` gives the model's order."""
    ordered: list[Call] = []
    start = 0
    for end in range(1, len(calls) + 1):
        if end == len(calls) or calls[end].turn != calls[start].turn:
            group = calls[start:end]
            if all(call.batch_index is not None for call in group):
                group.sort(key=lambda call: call.batch_index)
            ordered.extend(group)
            start = end
    return ordered


def read_trace(path: str) -> list[Run]:
    """One :class:`Run` per run id in a run log. A modify run writes its routing and model-sync
    ``llm_call`` records before its ``run_start``; a resume continues the run it resumes."""
    runs: list[Run] = []
    meter: _Meter | None = None
    phase, attempt = None, None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind, payload = event.get("event"), event.get("payload") or {}
            ts = event.get("ts") if isinstance(event.get("ts"), (int, float)) else None
            run_id = str(event.get("run_id") or "")
            if kind == "run_start" or not runs or (run_id and run_id != runs[-1].run_id):
                same = bool(runs) and runs[-1].run_id == run_id
                if not (same and (payload.get("resumed") or not runs[-1].started)):
                    runs.append(Run(source=path, run_id=run_id))
                    meter = _Meter(runs[-1])
                phase, attempt = None, None
                if kind == "run_start":
                    runs[-1].started = True
                    meter.read(kind, payload, ts, phase)
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
                index = payload.get("batch_index")
                runs[-1].calls.append(Call(
                    turn=int(payload.get("turn") or 0), phase=phase, tool=payload.get("tool", ""),
                    input=_uncut(raw), success=bool(payload.get("success")),
                    error="" if payload.get("success") else _error_of(payload),
                    attempt=attempt, lossy=_is_cut(raw), batch_index=index if isinstance(index, int) else None,
                ))
            meter.read(kind, payload, ts, phase)
            if kind == "phase_exit" and payload.get("phase") == "phase3":
                phase, attempt = None, None
    for run in runs:
        run.calls = _in_model_order(run.calls)
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


def _phase_of(turn: Turn) -> str:
    return "P3" if turn.phase and turn.phase.startswith("phase3") else "P2"


def _misses(run: Run) -> tuple[list[tuple[Turn, int]] | None, list[tuple[Turn, int]]]:
    """(turn, previous prompt) of the turns that read under 0.9 of the prefix they share with the
    previous request from the cache, with no compaction or eviction record: those whose prompt did
    not shrink (misses; None when the run reports no cache at all), and those whose prompt shrank
    (a rewrite of the history that wrote no record, or a log older than those records)."""
    if not any(turn.cache_read or turn.cache_write for turn in run.turns):
        return None, []
    flagged = [(turn, previous.prompt) for previous, turn in zip(run.turns, run.turns[1:])
               if not turn.new_context and turn.cache_read < CACHE_MISS_SHARE * min(turn.prompt, previous.prompt)]
    return ([row for row in flagged if row[0].prompt >= row[1]],
            [row for row in flagged if row[0].prompt < row[1]])


def cache_misses(run: Run) -> list[Turn] | None:
    """Loop turns that read under 0.9 of an unchanged prefix from the cache (None: cache unknown)."""
    misses, _ = _misses(run)
    return None if misses is None else [turn for turn, _ in misses]


def unrecorded_rewrites(run: Run) -> list[Turn]:
    """Loop turns whose prompt shrank and that read little from the cache, with no record of a rewrite."""
    return [turn for turn, _ in _misses(run)[1]]


def _tokens(rows) -> dict:
    """Token and cost sums of loop turns (``Turn``) or of ``llm_call`` payloads."""
    sums = {"fresh": 0, "cache_read": 0, "cache_write": 0, "output": 0, "cost": 0.0}
    for row in rows:
        if isinstance(row, Turn):
            values = (row.fresh, row.cache_read, row.cache_write, row.output, row.cost)
        else:
            values = (*(int(row.get(key) or 0) for key in _COUNTERS), float(row.get("cost_usd") or 0))
        for key, value in zip(sums, values):
            sums[key] += value
    return sums


def usage(run: Run) -> dict:
    """Tokens, cost and seconds of a run, per loop phase and per purpose of the calls outside the loop."""
    phases: dict[str, dict] = {}
    for name in ("P2", "P3"):
        turns = [turn for turn in run.turns if _phase_of(turn) == name]
        if turns:
            phases[name] = {"turns": len(turns), **_tokens(turns),
                            "model_seconds": sum(turn.model_seconds or 0.0 for turn in turns),
                            "tool_seconds": sum(turn.tool_seconds for turn in turns)}
    purposes: dict[str, dict] = {}
    for purpose in dict.fromkeys(purpose for purpose, _ in run.aux):
        calls = [payload for name, payload in run.aux if name == purpose]
        purposes[purpose] = {"calls": len(calls), **_tokens(calls)}
    read = sum(turn.cache_read for turn in run.turns)
    prompts = sum(turn.prompt for turn in run.turns)
    known = _misses(run)[0] is not None
    return {"phases": phases, "purposes": purposes, "misses": cache_misses(run),
            "cache_hit": read / prompts if known and prompts else None, "wall_seconds": run.wall_seconds}


def unique(runs: list[Run]) -> list[Run]:
    """Each run once: a modify run's workspace carries the run log of the runs before it."""
    seen: set[str] = set()
    kept = []
    for run in runs:
        if run.run_id and run.run_id in seen:
            continue
        seen.add(run.run_id)
        kept.append(run)
    return kept


def totals(runs: list[Run]) -> dict:
    """Loop tokens and the calls outside the loop over every run, each counted once."""
    runs = unique(runs)
    turns = [turn for run in runs for turn in run.turns]
    purposes: dict[str, dict] = {}
    for run in runs:
        for purpose, row in usage(run)["purposes"].items():
            total = purposes.setdefault(purpose, dict.fromkeys(row, 0))
            for key, value in row.items():
                total[key] += value
    misses = [turn for run in runs for turn in cache_misses(run) or []]
    rewrites = [turn for run in runs for turn in unrecorded_rewrites(run)]
    return {"runs": len(runs), "turns": len(turns), **_tokens(turns),
            "loop_input": sum(turn.prompt for turn in turns), "purposes": purposes,
            "misses": len(misses), "miss_fresh": sum(turn.fresh for turn in misses),
            "unrecorded_rewrites": len(rewrites), "rewrite_fresh": sum(turn.fresh for turn in rewrites)}


def _fmt(value, pattern="{:.2f}") -> str:
    return "-" if value is None else pattern.format(value)


def _label(run: Run, index: int, count: int) -> str:
    if run.run_id:
        return run.run_id[:8]
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
    return "\n".join(lines + [""] + details + [""] + usage_report(runs))


def _n(value) -> str:
    return f"{value:,}"


def usage_report(runs: list[Run]) -> list[str]:
    """The token, cost, cache and seconds table, its details and the totals (see the module docstring)."""
    lines = ["| run | phase | turns/calls | fresh in | cache read | cache write | output | cost $ | cache hit "
             "| misses | model s | tools s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    details: list[str] = []
    by_source: dict[str, list[Run]] = {}
    for run in runs:
        by_source.setdefault(run.source, []).append(run)
    for source_runs in by_source.values():
        for index, run in enumerate(source_runs):
            label = _label(run, index, len(source_runs))
            figures = usage(run)
            misses, rewrites = _misses(run)
            for name, row in figures["phases"].items():
                prompts = row["fresh"] + row["cache_read"] + row["cache_write"]
                hit, missed = "unknown", "unknown"
                if misses is not None:
                    hit = f"{row['cache_read'] / prompts:.0%}" if prompts else "-"
                    missed = str(sum(_phase_of(turn) == name for turn, _ in misses))
                lines.append(f"| {label} | {name} | {row['turns']} | {_n(row['fresh'])} | {_n(row['cache_read'])} "
                             f"| {_n(row['cache_write'])} | {_n(row['output'])} | {row['cost']:.4f} | {hit} "
                             f"| {missed} | {row['model_seconds']:.0f} | {row['tool_seconds']:.0f} |")
            if figures["purposes"]:
                outside = _tokens(payload for _, payload in run.aux)
                lines.append(f"| {label} | outside the loop | {len(run.aux)} | {_n(outside['fresh'])} "
                             f"| {_n(outside['cache_read'])} | {_n(outside['cache_write'])} "
                             f"| {_n(outside['output'])} | {outside['cost']:.4f} | - | - | - | - |")
                details.append(f"- {label} outside the loop: " + "; ".join(
                    f"{purpose} {row['calls']} call(s), {_n(row['fresh'] + row['cache_read'] + row['cache_write'])} "
                    f"in, ${row['cost']:.4f}" for purpose, row in figures["purposes"].items()))
            for title, rows in (("cache misses", misses), ("history rewritten without a compaction or eviction "
                                                           "record", rewrites)):
                if rows:
                    details.append(f"- {label} {title} (turn: fresh, cache read, previous prompt): " + "; ".join(
                        f"{turn.turn}: {_n(turn.fresh)} fresh, {_n(turn.cache_read)} read after a "
                        f"{_n(previous)}-token prompt" for turn, previous in rows))
            if figures["wall_seconds"] is not None and run.turns:
                model = sum(row["model_seconds"] for row in figures["phases"].values())
                tools = sum(row["tool_seconds"] for row in figures["phases"].values())
                details.append(f"- {label} seconds: wall {figures['wall_seconds']:.0f}, loop model calls "
                               f"{model:.0f}, loop tools {tools:.0f}, other {figures['wall_seconds'] - model - tools:.0f}")
    total = totals(runs)
    loop_input = total["loop_input"]
    summary = [
        f"Totals over {total['runs']} run(s), {total['turns']} loop turns: loop input {_n(loop_input)} "
        f"(fresh {_n(total['fresh'])}, cache read {_n(total['cache_read'])}"
        + (f" = {total['cache_read'] / loop_input:.1%}" if loop_input else "")
        + f", cache write {_n(total['cache_write'])}), output {_n(total['output'])}, cost ${total['cost']:.4f}; "
        f"cache misses {total['misses']} turn(s), {_n(total['miss_fresh'])} fresh tokens; history rewritten "
        f"without a record {total['unrecorded_rewrites']} turn(s), {_n(total['rewrite_fresh'])} fresh tokens.",
    ]
    if total["purposes"]:
        summary.append("Outside the loop: " + "; ".join(
            f"{purpose} {row['calls']} call(s), {_n(row['fresh'] + row['cache_read'] + row['cache_write'])} in, "
            f"{_n(row['output'])} out, ${row['cost']:.4f}"
            for purpose, row in sorted(total["purposes"].items(), key=lambda item: -item[1]["fresh"])))
    return lines + [""] + details + [""] + summary


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    runs = unique([run for path in argv for run in load_runs(path)])
    if not runs:
        print("no run log or recipe found")
        return 1
    print(report(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
