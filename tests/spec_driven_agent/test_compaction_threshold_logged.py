"""The run log names the compaction threshold the run's model resolved to.

A threshold that a provider catalog or a configuration change moves (80k to
about 180k for a free model whose catalog window loaded) multiplies every
turn's prompt and leaves no trace in the run log. The Phase 2 ``phase_enter``
record carries it, resolved with the run's own output reserve.
"""

import json

from besser.spec_driven_agent.agent.compaction import effective_threshold
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from tests.spec_driven_agent.scripted_rig import MODEL_ID, ScriptedAgent, run_scripted


def test_phase2_entry_records_the_resolved_compaction_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMOrchestrator, "_runtime_probe_issues", lambda self: [])
    agent = ScriptedAgent(str(tmp_path), [("list_files", {})], ["List the project files"])
    run_scripted(tmp_path, agent)
    records = [json.loads(line) for line in (tmp_path / ".besser_trace.jsonl").read_text(
        encoding="utf-8").splitlines()]
    [entry] = [r["payload"] for r in records
               if r["event"] == "phase_enter" and r["payload"].get("phase") == "phase2"]
    reserve = max(16_000, agent.orchestrator.client.max_tokens)
    assert entry["compact_threshold"] == effective_threshold(MODEL_ID, reserve=reserve)
