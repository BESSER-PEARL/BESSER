"""Run the real orchestrator against a scripted model that makes N tool calls per turn.

``ScriptedAgent`` extends :class:`run_transport.ScriptedModel`: instead of a
list of replies it holds an ordered list of tool calls and sends them
``calls_per_turn`` at a time, Phase 2's first and then Phase 3's. When its list
is empty it ends the turn. A :class:`Rule` reacts to a tool result: when the
result of a call to ``tool`` contains ``text``, its calls are sent next (once).

Everything below the SDK is real: the provider client, the orchestrator's
Phase 2 loop, Phase 3 validation and repair, the tool executor, the run log.

``script_from_trace`` turns a recorded run log into such a list; see
``scripts/trace_metrics.py`` for which recorded inputs are complete.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from tests.spec_driven_agent.run_transport import ScriptedModel, Step, claude_run_client, plain
from tests.spec_driven_agent.scripts.trace_metrics import load_runs
from tests.spec_driven_agent.shelf_app import shelf_model

MODEL_ID = "claude-sonnet-4-6"
REQUEST = "Build a REST API for my book shelf, with a stats endpoint for page totals."
# A script entry is (tool name, input), or a callable taking the agent that
# returns one when it is sent (e.g. a task_list call that looks up a task id).


@dataclass
class Rule:
    tool: str
    text: str
    then: list = field(default_factory=list)
    fired: bool = False


class FakeClock:
    """A ``time`` module whose ``monotonic`` only moves when told to."""

    def __init__(self, start: float = 1_000.0):
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __getattr__(self, name):
        return getattr(_time, name)


class ScriptedAgent(ScriptedModel):
    def __init__(self, workspace: str, calls: list, tasks: list[str], *, calls_per_turn: int = 1,
                 phase3_calls: list | None = None, rules: list[Rule] | None = None,
                 clock: FakeClock | None = None, seconds_per_turn: float = 0.0):
        super().__init__(workspace, [], tasks)
        self.queue = {"phase2": list(calls), "phase3": list(phase3_calls or [])}
        self.calls_per_turn = calls_per_turn
        self.rules = list(rules or [])
        self.clock, self.seconds_per_turn = clock, seconds_per_turn
        self.transport = None   # the fake SDK client, set by run_scripted
        self.sent: dict[str, list[list[tuple[str, dict]]]] = {"phase2": [], "phase3": []}
        self.forced: dict[str, list[str | None]] = {"phase2": [], "phase3": []}

    def phase(self) -> str:
        return "phase3" if self.orchestrator.phase3_started_at is not None else "phase2"

    def _last_results(self) -> list[tuple[str, str]]:
        """(tool name, result text) of the calls the last request answers."""
        messages = plain(self.transport.requests[-1].kwargs.get("messages") or [])
        if len(messages) < 2 or not isinstance(messages[-1].get("content"), list):
            return []
        names = {block.get("id"): block.get("name") for block in messages[-2].get("content") or []
                 if isinstance(block, dict) and block.get("type") == "tool_use"}
        results = []
        for block in messages[-1]["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                content = block.get("content")
                text = content if isinstance(content, str) else str(plain(content))
                results.append((names.get(block.get("tool_use_id"), ""), text))
        return results

    def reply(self, kind: str):
        if kind != "phase2":   # routing, planning: the fixed answers
            return super().reply(kind)
        phase = self.phase()
        queue = self.queue[phase]
        for tool, text in self._last_results():
            for rule in self.rules:
                if not rule.fired and rule.tool == tool and rule.text in text:
                    rule.fired = True
                    queue[:0] = rule.then
        choice = self.transport.requests[-1].kwargs.get("tool_choice") or {}
        self.forced[phase].append(choice.get("name"))
        batch = [entry(self) if callable(entry) else entry for entry in queue[:self.calls_per_turn]]
        del queue[:self.calls_per_turn]
        self.sent[phase].append(batch)
        if self.clock:
            self.clock.advance(self.seconds_per_turn)
        self.phase2 = [Step(calls=batch) if batch else Step(text="Done.")]
        return super().reply(kind)


def run_scripted(workspace, agent: ScriptedAgent, *, max_turns: int | None = None,
                 max_runtime_seconds: int = 1200, auto_fix: bool = True,
                 on_findings=None) -> LLMOrchestrator:
    """One whole run of the shelf app (FastAPI scaffold) driven by ``agent``."""
    client, fake = claude_run_client(MODEL_ID, agent, with_thinking=False)
    agent.transport = fake
    orchestrator = LLMOrchestrator(
        llm_client=client, domain_model=shelf_model(), output_dir=str(workspace),
        max_turns=max_turns, max_runtime_seconds=max_runtime_seconds, use_streaming=False,
        enable_requirements_ledger=False, auto_fix_issues=auto_fix, on_findings=on_findings,
    )
    agent.orchestrator = orchestrator
    orchestrator.run(REQUEST)
    return orchestrator


def script_from_trace(path: str, run: int = -1) -> tuple[list, list, int]:
    """(Phase 2 calls, Phase 3 calls, calls whose recorded input was cut) of one recorded run.

    A cut input keeps its recorded prefix, so such a write lands shorter than it
    was made, or is refused where its old_text was cut.
    """
    recorded = load_runs(path)[run]
    phase2 = [(c.tool, c.input) for c in recorded.calls if c.phase in ("phase2", None)]
    phase3 = [(c.tool, c.input) for c in recorded.calls if c.phase == "phase3"]
    return phase2, phase3, sum(c.lossy for c in recorded.calls) + recorded.calls_dropped
