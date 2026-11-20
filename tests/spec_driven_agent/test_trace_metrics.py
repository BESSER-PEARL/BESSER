"""Token, cost, cache and time figures that ``scripts/trace_metrics.py`` reads from a run log.

The run log keeps the client's cumulative counters on every ``cost_update`` (one per
loop turn) and a delta per ``llm_call`` (calls outside the loop: planning, the judge).
A loop turn's tokens are the difference of two ``cost_update`` records less the
``llm_call`` records between them. The synthetic logs below follow the shape of
recorded production run logs.
"""

from __future__ import annotations

import json

import pytest

from tests.spec_driven_agent.scripts.trace_metrics import (
    cache_misses, load_runs, report, totals, usage,
)


class _Log:
    """Writes a run log; counters are cumulative like the client's."""

    def __init__(self, run_id: str = "run-a"):
        self.run_id = run_id
        self.records: list[dict] = []
        self.ts = 1_000.0
        self.counters = {"input_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0,
                         "output_tokens": 0}
        self.cost = 0.0

    def event(self, kind: str, at: float | None = None, **payload):
        if at is not None:
            self.ts = at
        self.records.append({"ts": self.ts, "run_id": self.run_id, "event": kind, "payload": payload})

    def _spend(self, fresh, read, write, output, cost):
        for key, value in zip(self.counters, (fresh, read, write, output)):
            self.counters[key] += value
        self.cost += cost

    def llm_call(self, purpose, fresh, output=0, read=0, cost=0.0):
        self._spend(fresh, read, 0, output, cost)
        self.event("llm_call", purpose=purpose, api_calls=1, input_tokens=fresh, output_tokens=output,
                   cache_read_tokens=read, cache_creation_tokens=0, cost_usd=cost)

    def turn(self, number, fresh, read=0, write=0, output=100, cost=0.01, *, start, answered, tools_done=None,
             phase=None, compacted=False):
        extra = {"phase": phase} if phase else {}
        self.event("turn_start", at=start, turn=number, **extra)
        if compacted:
            self.event("compaction", turn=number, tokens_before=11_000, tokens_after=4_000)
        self._spend(fresh, read, write, output, cost)
        self.event("cost_update", at=answered, turn=number, estimated_cost_usd=self.cost, **extra,
                   **self.counters)
        if tools_done is not None:
            self.event("tool_call", at=tools_done, turn=number, tool="read_file", success=True,
                       input={"path": "app.py"}, blocks_in_turn=1)

    def save(self, path) -> str:
        path.write_text("".join(json.dumps(r) + "\n" for r in self.records), encoding="utf-8")
        return str(path)


def _one_run(tmp_path, log: _Log):
    [run] = load_runs(log.save(tmp_path / "besser_llm_0123abcd_x.trace.jsonl"))
    return run


def test_loop_tokens_are_split_by_phase_and_the_calls_outside_the_loop_by_purpose(tmp_path):
    log = _Log()
    log.event("run_start", at=1_000.0)
    log.event("phase_enter", phase="phase2")
    log.llm_call("gap_analysis", 1_000, output=100, cost=0.01)
    log.turn(1, 5_000, output=200, cost=0.05, start=1_010.0, answered=1_020.0, tools_done=1_023.0)
    log.turn(2, 500, read=5_000, output=100, cost=0.02, start=1_024.0, answered=1_030.0, tools_done=1_031.0)
    log.event("phase_exit", phase="phase2")
    log.event("phase_enter", phase="phase3")
    log.llm_call("requirements_judge", 3_000, output=50, cost=0.03)
    log.llm_call("requirements_rejudge", 1_000, output=40, cost=0.01)
    log.event("phase_enter", phase="phase3_fix_attempt", attempt=1)
    log.turn(3, 4_000, write=300, output=60, cost=0.04, start=1_040.0, answered=1_050.0, tools_done=1_052.0,
             phase="phase3_fix")
    log.event("run_end", at=1_100.0)

    figures = usage(_one_run(tmp_path, log))

    assert figures["phases"]["P2"] == {
        "turns": 2, "fresh": 5_500, "cache_read": 5_000, "cache_write": 0, "output": 300,
        "cost": pytest.approx(0.07), "model_seconds": pytest.approx(16.0), "tool_seconds": pytest.approx(4.0)}
    assert figures["phases"]["P3"] == {
        "turns": 1, "fresh": 4_000, "cache_read": 0, "cache_write": 300, "output": 60,
        "cost": pytest.approx(0.04), "model_seconds": pytest.approx(10.0), "tool_seconds": pytest.approx(2.0)}
    assert {purpose: (row["calls"], row["fresh"], row["output"]) for purpose, row in figures["purposes"].items()} == {
        "gap_analysis": (1, 1_000, 100), "requirements_judge": (1, 3_000, 50),
        "requirements_rejudge": (1, 1_000, 40)}
    assert figures["wall_seconds"] == pytest.approx(100.0)


def test_a_turn_that_reads_little_of_an_unchanged_prefix_from_cache_is_flagged(tmp_path):
    log = _Log()
    log.event("run_start", at=1_000.0)
    log.event("phase_enter", phase="phase2")
    log.turn(1, 10_000, start=1_001.0, answered=1_002.0)            # first request: nothing cached yet
    log.turn(2, 500, read=10_000, start=1_003.0, answered=1_004.0)    # hit
    log.turn(3, 10_600, start=1_005.0, answered=1_006.0)            # same prefix, nothing read: a miss
    log.turn(4, 300, read=10_600, start=1_007.0, answered=1_008.0)    # hit
    # Rewritten history (an eviction that a large new result outgrew): expected.
    log.turn(5, 11_000, start=1_009.0, answered=1_010.0, compacted=True)
    log.turn(6, 2_000, start=1_011.0, answered=1_012.0)             # shrank with no record: listed apart
    log.turn(7, 100, read=1_900, start=1_013.0, answered=1_014.0)     # hit
    log.turn(8, 100, read=1_750, start=1_015.0, answered=1_016.0)     # shrank, prefix kept: a hit
    log.event("phase_enter", phase="phase3_fix_attempt", attempt=1)
    log.turn(9, 8_000, start=1_017.0, answered=1_018.0)             # a new conversation
    log.turn(10, 100, read=8_000, start=1_019.0, answered=1_020.0)    # hit
    log.event("run_end", at=1_021.0)

    run = _one_run(tmp_path, log)
    text = report([run])

    assert [turn.turn for turn in cache_misses(run)] == [3]
    assert "3: 10,600 fresh, 0 read after a 10,500-token prompt" in text
    # A rewrite that writes no record is what the counter must expose, apart from the misses.
    assert ("history rewritten without a compaction or eviction record (turn: fresh, cache read, previous "
            "prompt): 6: 2,000 fresh, 0 read after a 11,000-token prompt\n") in text
    assert totals([run])["misses"] == 1 and totals([run])["unrecorded_rewrites"] == 1


def test_cache_is_unknown_when_the_provider_reports_no_cache_reads(tmp_path):
    log = _Log()
    log.event("run_start", at=1_000.0)
    log.event("phase_enter", phase="phase2")
    log.turn(1, 10_000, start=1_001.0, answered=1_002.0)
    log.turn(2, 10_500, start=1_003.0, answered=1_004.0)
    log.event("run_end", at=1_005.0)

    run = _one_run(tmp_path, log)

    assert cache_misses(run) is None
    assert usage(run)["cache_hit"] is None
    assert "| unknown | unknown |" in report([run])


def test_totals_count_each_run_once(tmp_path):
    first, second = _Log("run-a"), _Log("run-b")
    for log in (first, second):
        log.event("run_start", at=1_000.0)
        log.event("phase_enter", phase="phase2")
        log.llm_call("requirements_judge", 2_000, output=10)
        log.turn(1, 1_000, read=4_000, start=1_001.0, answered=1_002.0)
        log.event("run_end", at=1_003.0)
    # A modify run's log also carries the run it was seeded from.
    second.records = first.records + second.records
    runs = load_runs(first.save(tmp_path / "a.trace.jsonl")) + load_runs(second.save(tmp_path / "b.trace.jsonl"))

    figures = totals(runs)

    assert figures["runs"] == 2
    assert (figures["loop_input"], figures["cache_read"]) == (10_000, 8_000)
    assert figures["purposes"]["requirements_judge"]["calls"] == 2


def test_a_resumed_run_keeps_the_turns_of_both_parts(tmp_path):
    """A resume keeps the run id and appends to the same run log; its new client's token
    counters restart at zero while its cost is seeded with what the run already spent."""
    log = _Log()
    log.event("run_start", at=1_000.0)
    log.event("phase_enter", phase="phase2")
    log.turn(1, 10_000, cost=0.01, start=1_001.0, answered=1_002.0)
    log.turn(2, 10_000, cost=0.01, start=1_003.0, answered=1_004.0)
    log.event("run_end", at=1_010.0)
    log.counters = dict.fromkeys(log.counters, 0)
    log.event("run_start", at=5_000.0, resumed=True, resume_from_turn=2, resume_phase="phase2")
    log.event("phase_enter", phase="phase2_resume")
    log.turn(3, 20_000, cost=0.02, start=5_001.0, answered=5_002.0)
    log.turn(4, 20_000, cost=0.02, start=5_003.0, answered=5_004.0)
    log.turn(5, 20_000, cost=0.02, start=5_005.0, answered=5_006.0)
    log.event("run_end", at=5_020.0)

    run = _one_run(tmp_path, log)
    figures = totals([run])

    assert [turn.turn for turn in run.turns] == [1, 2, 3, 4, 5]
    assert [turn.fresh for turn in run.turns] == [10_000, 10_000, 20_000, 20_000, 20_000]
    assert [turn.cost for turn in run.turns] == pytest.approx([0.01, 0.01, 0.02, 0.02, 0.02])
    assert (figures["runs"], figures["turns"], figures["fresh"]) == (1, 5, 80_000)
    assert figures["cost"] == pytest.approx(0.08)
    assert run.wall_seconds == pytest.approx(30.0)      # the time between the parts is not the run's


def test_a_turns_calls_are_read_in_the_order_the_model_emitted_them(tmp_path):
    log = _Log()
    log.event("run_start", at=1_000.0)
    log.event("phase_enter", phase="phase2")
    log.event("turn_start", turn=1)
    # Parallel calls are logged as each finishes.
    for path, index in (("b.py", 1), ("c.py", 2), ("a.py", 0)):
        log.event("tool_call", turn=1, tool="read_file", success=True, input={"path": path},
                  blocks_in_turn=3, batch_index=index)
    log.event("tool_call", turn=2, tool="read_file", success=True, input={"path": "d.py"}, blocks_in_turn=1,
              batch_index=0)

    run = _one_run(tmp_path, log)

    assert [call.input["path"] for call in run.calls] == ["a.py", "b.py", "c.py", "d.py"]
