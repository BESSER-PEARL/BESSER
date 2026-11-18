"""One trace event name for an LLM call outside the Phase 2 loop.

Live evidence: r11-p81 traced $1.32 of cost_update while the run cost $1.64;
the routing, gap-analysis, requirements-judge and Phase 3 calls left no record.
"""

import json

from besser.spec_driven_agent.state.tracing import EVENT_LLM_CALL, TraceWriter


def test_llm_call_event_round_trips(tmp_path):
    writer = TraceWriter(str(tmp_path), run_id="r1")

    writer.write(EVENT_LLM_CALL, purpose="requirements_judge", outcome="ok",
                 input_tokens=1200, output_tokens=300, cost_usd=0.0123)

    record = json.loads((tmp_path / ".besser_trace.jsonl").read_text(encoding="utf-8"))
    assert record["event"] == "llm_call"
    assert record["payload"]["purpose"] == "requirements_judge"
    assert record["payload"]["cost_usd"] == 0.0123
