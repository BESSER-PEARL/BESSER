"""What one Phase 3 fix round sends, and what it no longer repeats.

Measured on a 10-class model, a round's first message was 6.8k tokens: the
domain model as plain ``json.dumps``, the whole model's mutation manifest for
a single syntax error, and every blocker pasted verbatim.
"""
from __future__ import annotations

import os

from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.spec_driven_agent.model_serializer import serialize_domain_model, to_prompt_json
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue

SYNTAX = ValidationIssue("blocker", "Syntax error in app/main.py line 1: invalid syntax")


class _Usage:
    estimated_cost = 0.0


class _Block:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _RecordingClient:
    """Records each request; ends every attempt with prose."""
    model = "mock-model"

    def __init__(self):
        self.usage = _Usage()
        self.calls: list[dict] = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.calls.append({"system": system, "messages": list(messages)})
        return {"stop_reason": "end_turn", "content": [_Block("text", text="done")]}


def _build(tmp_path, domain_model=None) -> tuple[LLMOrchestrator, _RecordingClient]:
    os.makedirs(tmp_path / "app", exist_ok=True)
    (tmp_path / "app" / "main.py").write_text("x = 1\n", encoding="utf-8")
    client = _RecordingClient()
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=domain_model or DomainModel(
            name="Shop", types={Class(name="Café", attributes={Property(name="größe", type=StringType)})}),
        output_dir=str(tmp_path),
        max_cost_usd=10.0,
        enable_tracing=False,
        enable_checkpointing=False,
    )
    return orch, client


def _first_message(call: dict) -> str:
    content = call["messages"][0]["content"]
    return content if isinstance(content, str) else content[0]["text"]


def _whole_prompt(call: dict) -> str:
    return call["system"] + "\n" + _first_message(call)


def test_the_run_constant_sections_are_a_byte_identical_system_prompt(tmp_path):
    """Anthropic caches on block boundaries and OpenAI-compatible providers on
    the byte prefix: the request, model and ledger must not share a block with
    the per-round inventory and blockers, or every round re-writes them."""
    orch, client = _build(tmp_path)
    orch._instructions = "Every Café needs a größe."
    orch._phase2_inspection_handoff = "Phase 2 inspected app/main.py."
    orch._requirements = [{"id": 1, "kind": "rule", "text": "A Café without a größe is rejected."}]

    orch._invoke_phase3_fix_loop([SYNTAX], is_first_attempt=True)
    (tmp_path / "app" / "extra.py").write_text("y = 2\n", encoding="utf-8")
    later = ValidationIssue("blocker", "undefined name: app/extra.py line 1: name 'z' is not defined")
    orch._invoke_phase3_fix_loop([later], is_first_attempt=False)

    first, second = client.calls[0], client.calls[-1]
    assert first["system"] == second["system"]
    for constant in ("Every Café needs a größe.", "Phase 2 inspected app/main.py.",
                     "R1. A Café without a größe is rejected.", "Domain model and conversion losses"):
        assert constant in first["system"]
        assert constant not in _first_message(first)
    assert SYNTAX.message in _first_message(first) and "app/extra.py" in _first_message(second)
    assert "app/extra.py" not in second["system"] and later.message not in second["system"]


def test_the_domain_model_is_embedded_as_compact_readable_json(tmp_path):
    orch, client = _build(tmp_path)
    orch._invoke_phase3_fix_loop([SYNTAX], is_first_attempt=True)

    prompt = _whole_prompt(client.calls[0])
    assert to_prompt_json(serialize_domain_model(orch.domain_model)) in prompt
    assert "Café" in prompt and "größe" in prompt and "\\u00e9" not in prompt


def test_the_blocker_list_is_capped_with_an_honest_remainder(tmp_path):
    from besser.spec_driven_agent.pipeline.phase3_repair import _FIX_PROMPT_MAX_BLOCKERS

    orch, client = _build(tmp_path)
    startup = ValidationIssue("blocker", "missing module: app/main.py line 1: imports 'app.db', which does not exist")
    cascade = [ValidationIssue("blocker", f"tsc [frontend]: src/P{n}.tsx(1,1): error TS2304: Cannot find name 'api'")
               for n in range(100)]
    judged = [ValidationIssue("blocker", f"requirement: R{n} — rule {n} is not implemented") for n in range(5)]
    orch._invoke_phase3_fix_loop(judged + cascade + [startup], is_first_attempt=True)

    message = _first_message(client.calls[0])
    bullets = [line for line in message.splitlines() if line.startswith("- ")]
    assert len(bullets) == _FIX_PROMPT_MAX_BLOCKERS + 1
    assert bullets[0] == f"- {startup.message}", "the cap keeps the repair priority order"
    rest = 106 - _FIX_PROMPT_MAX_BLOCKERS
    assert bullets[-1].startswith(f"- (+{rest} more blockers of these kinds: tsc [frontend] x")
    assert bullets[-1].endswith(", requirement x5; the next validation lists them again)")


def test_a_long_blocker_is_cut_with_a_marker_keeping_its_tail(tmp_path):
    from besser.spec_driven_agent.pipeline.phase3_repair import _FIX_PROMPT_BLOCKER_CHARS

    orch, client = _build(tmp_path)
    traceback = ("application startup: app: Traceback (most recent call last):\n"
                 + "  File \"lib.py\", line 1, in f\n" * 120
                 + "NameError: name 'Base' is not defined. Fix site: app/main.py line 1")
    orch._invoke_phase3_fix_loop([ValidationIssue("blocker", traceback)], is_first_attempt=True)

    message = _first_message(client.calls[0])
    start = message.index("- application startup:")
    bullet = message[start:message.index("NameError", start) + 200].split("\n\n")[0]
    assert "chars cut ...]" in bullet
    assert bullet.endswith("NameError: name 'Base' is not defined. Fix site: app/main.py line 1")
    assert len(bullet) < _FIX_PROMPT_BLOCKER_CHARS + 60
    assert "The offending lines" in message, "excerpts still come from the full message"
