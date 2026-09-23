"""Tests for the A2A wire-tag parser and the standalone annotation pass.

- `parse_a2a_line` / `parse_a2a_out_block` are pure (no backend deps, base-runnable).
- `annotate_agent_with_a2a` over a real `process_agent_diagram(json)` agent needs the
  converter import (besser env), and proves the legacy-tolerant contract (§10): a
  diagram with no `a2a:` tag yields an agent with NO `_a2a` attribute.
"""
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.a2a_tags import (
    annotate_agent_with_a2a,
    parse_a2a_line,
    parse_a2a_out_block,
)


# ---------------------------------------------------------------------------
# parse_a2a_line — §4 acceptance
# ---------------------------------------------------------------------------

def test_parse_inbound_line():
    d = parse_a2a_line("a2a:in;peer=AgentReviewer;ref=u;flow=flow-09;kind=supervises")
    assert d == {
        "dir": "in", "peer": "AgentReviewer", "ref": "u", "flow": "flow-09",
        "order": 9999, "kind": "supervises",
    }


def test_parse_outbound_line_with_order():
    d = parse_a2a_line("a2a:out;peer=AgentCoder;ref=u;flow=flow-09;order=1;kind=supervises")
    assert d["dir"] == "out"
    assert d["order"] == 1
    assert d["peer"] == "AgentCoder"


def test_absent_kind_is_none_plain_channel():
    d = parse_a2a_line("a2a:out;peer=AgentCoder;ref=u;flow=f;order=2")
    assert d["kind"] is None


def test_unknown_kind_degrades_to_none():
    d = parse_a2a_line("a2a:out;peer=X;kind=foo")
    assert d["kind"] is None


def test_non_tag_lines_return_none():
    assert parse_a2a_line("hello world") is None
    assert parse_a2a_line("") is None
    assert parse_a2a_line(None) is None


def test_empty_ref_preserved():
    d = parse_a2a_line("a2a:out;peer=AgentCoder;ref=;flow=f;order=1;kind=delegates")
    assert d["ref"] == ""


def test_malformed_order_degrades_to_sentinel():
    d = parse_a2a_line("a2a:out;peer=X;order=notanint;kind=delegates")
    assert d["order"] == 9999


# ---------------------------------------------------------------------------
# parse_a2a_out_block — multi-line, order-sorted
# ---------------------------------------------------------------------------

def test_out_block_sorted_by_order():
    block = "a2a:out;peer=B;order=2\na2a:out;peer=A;order=1"
    edges = parse_a2a_out_block(block)
    assert [e["peer"] for e in edges] == ["A", "B"]


def test_out_block_ignores_prose_lines():
    block = "Some human note about this state.\na2a:out;peer=A;order=1\nmore prose"
    edges = parse_a2a_out_block(block)
    assert [e["peer"] for e in edges] == ["A"]


def test_out_block_empty_when_no_tag():
    assert parse_a2a_out_block("just prose") == []
    assert parse_a2a_out_block(None) == []
    assert parse_a2a_out_block("") == []


# ---------------------------------------------------------------------------
# annotate_agent_with_a2a — raw JSON (no converter needed for the parse itself)
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, name="A"):
        self.name = name


def _tagged_diagram_json():
    """A minimal AgentDiagram JSON shaped like the frozen wire contract (§1):
    an AgentState carrying two a2a:out lines on `description`, and a
    when_intent_matched transition carrying a2a:in on its `name`."""
    return {
        "id": "diagram-coder",
        "model": {
            "elements": {
                "s-init": {"type": "StateInitialNode", "name": ""},
                "s-greet": {"type": "AgentState", "name": "greeting"},
                "s-write": {
                    "type": "AgentState",
                    "name": "write_code",
                    "description": (
                        "a2a:out;peer=AgentReviewer;ref=rev-uuid;flow=flow-09;order=2;kind=supervises\n"
                        "a2a:out;peer=AgentTester;ref=test-uuid;flow=flow-09;order=1;kind=delegates"
                    ),
                },
            },
            "relationships": {
                "t-init": {
                    "type": "AgentStateTransitionInit",
                    "name": "",
                    "source": {"element": "s-init"},
                    "target": {"element": "s-greet"},
                },
                "t-in": {
                    "type": "AgentStateTransition",
                    "name": "a2a:in;peer=AgentSupervisor;ref=sup-uuid;flow=flow-09;kind=delegates",
                    "source": {"element": "s-greet"},
                    "target": {"element": "s-write"},
                    "predefined": {"predefinedType": "when_intent_matched",
                                   "intentName": "recv_AgentSupervisor_task"},
                },
            },
        },
    }


def test_annotate_populates_outbound_and_inbound():
    agent = annotate_agent_with_a2a(_FakeAgent("AgentCoder"), _tagged_diagram_json())
    a2a = agent._a2a
    # outbound order-sorted: Tester (order=1) before Reviewer (order=2)
    assert [e["peer"] for e in a2a["outbound"]] == ["AgentTester", "AgentReviewer"]
    assert a2a["outbound"][0]["kind"] == "delegates"
    assert a2a["outbound"][0]["state"] == "write_code"
    assert a2a["outbound"][1]["kind"] == "supervises"

    assert len(a2a["inbound"]) == 1
    inb = a2a["inbound"][0]
    assert inb["peer"] == "AgentSupervisor"
    assert inb["kind"] == "delegates"
    assert inb["intent"] == "recv_AgentSupervisor_task"
    assert inb["source_state"] == "greeting"
    assert inb["target_state"] == "write_code"


def test_annotate_legacy_diagram_sets_no_attribute():
    """Legacy guard (§10): a diagram with no a2a: tag anywhere → no _a2a attribute."""
    legacy = {
        "id": "d",
        "model": {
            "elements": {
                "s1": {"type": "AgentState", "name": "idle"},
            },
            "relationships": {
                "t1": {"type": "AgentStateTransition", "name": "",
                       "source": {"element": "s1"}, "target": {"element": "s1"},
                       "predefined": {"predefinedType": "auto", "conditionValue": ""}},
            },
        },
    }
    agent = annotate_agent_with_a2a(_FakeAgent(), legacy)
    assert not hasattr(agent, "_a2a")


def test_annotate_never_raises_on_garbage():
    annotate_agent_with_a2a(_FakeAgent(), {})
    annotate_agent_with_a2a(_FakeAgent(), {"model": None})
    annotate_agent_with_a2a(_FakeAgent(), None)  # not a dict → no-op


# ---------------------------------------------------------------------------
# annotate over the REAL converter (besser env) — proves the converter is untouched
# ---------------------------------------------------------------------------

def test_annotate_over_real_converter_keeps_intent_and_states():
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        process_agent_diagram,
    )

    json_data = _tagged_diagram_json()
    # Build the intents the transition references, so the converter resolves them.
    json_data["model"]["elements"]["i-recv"] = {
        "type": "AgentIntent", "name": "recv_AgentSupervisor_task", "bodies": [],
    }
    # Point the transition's intent at the real intent element name.
    agent = process_agent_diagram(json_data)
    annotate_agent_with_a2a(agent, json_data)

    # The annotation is read-only over the JSON: states + intent still built by the
    # converter, and _a2a is populated.
    state_names = {getattr(s, "name", "") for s in agent.states}
    assert {"greeting", "write_code"} <= state_names
    assert any(i.name == "recv_AgentSupervisor_task" for i in agent.intents)
    assert agent._a2a["inbound"][0]["intent"] == "recv_AgentSupervisor_task"
