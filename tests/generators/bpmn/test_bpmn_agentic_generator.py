"""Tests for the SEAA'25 agentic extensions in the BPMN 2.0 XML generator.

Mirrors ``test_bpmn_generator.py`` but exercises the new
``<bpmn:extensionElements><agentic:agentic .../></bpmn:extensionElements>``
emission.

P3' rationalization: CollaborationMode, MergingStrategy, AgenticMessageFlow
have been removed. Tests updated accordingly.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from besser.BUML.metamodel.bpmn import (
    AgenticGateway,
    AgenticLane,
    AgenticTask,
    AgentRole,
    BPMNModel,
    Collaboration,
    EndEvent,
    GatewayRole,
    GatewayType,
    Lane,
    MessageFlow,
    Participant,
    Process,
    ReflectionMode,
    SequenceFlow,
    StartEvent,
    Task,
    TaskType,
)
from besser.generators.bpmn import BPMNGenerator


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

_NS = {
    "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "agentic": "https://www.besser-pearl.org/bpmn/agentic",
}


def _generate(model: BPMNModel, tmp_path) -> Path:
    gen = BPMNGenerator(model, output_dir=str(tmp_path))
    out = gen.generate()
    return Path(out)


def _parse(path: Path) -> ET.Element:
    tree = ET.parse(str(path))
    return tree.getroot()


def _findall(root, prefix_local: str):
    prefix, local = prefix_local.split(":", 1)
    return root.findall(f".//{{{_NS[prefix]}}}{local}")


def _find(root, prefix_local: str):
    prefix, local = prefix_local.split(":", 1)
    return root.find(f".//{{{_NS[prefix]}}}{local}")


def _agentic_inner(root) -> list:
    """Return every ``<agentic:agentic ...>`` element under ``root``."""
    return _findall(root, "agentic:agentic")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _agentic_task_model() -> BPMNModel:
    """StartEvent -> AgenticTask -> EndEvent, single process."""
    s = StartEvent(name="go")
    t = AgenticTask(
        name="Review",
        task_type=TaskType.USER,
        reflection_mode=ReflectionMode.CROSS,
        trust_score=85,
    )
    e = EndEvent(name="done")
    p = Process(name="P", flow_nodes={s, t, e},
                sequence_flows={SequenceFlow(s, t), SequenceFlow(t, e)})
    return BPMNModel(name="AgTaskModel", processes={p})


def _agentic_gateway_merging_model() -> BPMNModel:
    """Single merging AgenticGateway -- trust_score=85."""
    g = AgenticGateway(
        name="Vote",
        gateway_type=GatewayType.PARALLEL,
        gateway_role=GatewayRole.MERGING,
        trust_score=85,
    )
    p = Process(name="P", flow_nodes={g})
    return BPMNModel(name="AgGatewayMerging", processes={p})


def _agentic_gateway_diverging_model() -> BPMNModel:
    """Single diverging AgenticGateway -- trust_score=85, no strategy."""
    g = AgenticGateway(
        name="Fork",
        gateway_type=GatewayType.INCLUSIVE,
        gateway_role=GatewayRole.DIVERGING,
        trust_score=85,
    )
    p = Process(name="P", flow_nodes={g})
    return BPMNModel(name="AgGatewayDiverging", processes={p})


def _agentic_lane_model() -> BPMNModel:
    """One AgenticLane wrapping a single Task."""
    t = Task(name="Code")
    lane = AgenticLane(
        name="Reviewer",
        role=AgentRole.MANAGER,
        trust_score=85,
        flow_nodes={t},
    )
    p = Process(name="P", flow_nodes={t}, lanes={lane})
    return BPMNModel(name="AgLaneModel", processes={p})


def _vanilla_model() -> BPMNModel:
    """No agentic classes anywhere -- StartEvent -> Task -> EndEvent."""
    s = StartEvent(name="")
    t = Task(name="Do")
    e = EndEvent(name="")
    p = Process(name="P", flow_nodes={s, t, e},
                sequence_flows={SequenceFlow(s, t), SequenceFlow(t, e)})
    return BPMNModel(name="Vanilla", processes={p})


def _vanilla_lane_model() -> BPMNModel:
    """Base Lane (not AgenticLane) wrapping a single Task."""
    t = Task(name="Do")
    lane = Lane(name="Worker", flow_nodes={t})
    p = Process(name="P", flow_nodes={t}, lanes={lane})
    return BPMNModel(name="VanillaLane", processes={p})


def _mixed_model() -> BPMNModel:
    """One AgenticTask + one base Task in the same process."""
    a = AgenticTask(name="Agentic", reflection_mode=ReflectionMode.SELF, trust_score=50)
    b = Task(name="Vanilla", task_type=TaskType.USER)
    p = Process(name="P", flow_nodes={a, b})
    return BPMNModel(name="Mixed", processes={p})


# ---------------------------------------------------------------------------
# G-1 -- namespace declared on root
# ---------------------------------------------------------------------------

class TestNamespaceDeclaration:
    def test_namespace_declared_on_root_for_agentic_model(self, tmp_path):  # G-1
        path = _generate(_agentic_task_model(), tmp_path)
        text = path.read_text(encoding="utf-8")
        assert 'xmlns:agentic="https://www.besser-pearl.org/bpmn/agentic"' in text


# ---------------------------------------------------------------------------
# G-2 / G-3 -- AgenticTask emission
# ---------------------------------------------------------------------------

class TestAgenticTaskEmission:
    def test_agentic_task_emits_extension_element(self, tmp_path):  # G-2
        root = _parse(_generate(_agentic_task_model(), tmp_path))
        ext = _findall(root, "bpmn:extensionElements")
        assert len(ext) == 1
        inner = _agentic_inner(root)
        assert len(inner) == 1
        # The <agentic:agentic> sits inside the <bpmn:extensionElements>.
        assert list(ext[0]) == inner

    def test_agentic_task_attributes_match_metamodel(self, tmp_path):  # G-3
        root = _parse(_generate(_agentic_task_model(), tmp_path))
        inner = _agentic_inner(root)[0]
        assert inner.attrib.get("reflectionMode") == "cross"
        assert inner.attrib.get("trustScore") == "85"
        # No agentDiagramRef when the task is not linked.
        assert "agentDiagramRef" not in inner.attrib
        # Gateway-only / lane-only attrs absent.
        for forbidden in ("gatewayRole", "mergingStrategy", "role",
                          "collaborationMode"):
            assert forbidden not in inner.attrib


# ---------------------------------------------------------------------------
# G-4 / G-5 -- AgenticGateway emission (merging vs diverging)
# ---------------------------------------------------------------------------

class TestAgenticGatewayEmission:
    def test_agentic_gateway_merging_emits_attrs(self, tmp_path):  # G-4
        root = _parse(_generate(_agentic_gateway_merging_model(), tmp_path))
        inner = _agentic_inner(root)
        assert len(inner) == 1
        attrs = inner[0].attrib
        assert attrs.get("gatewayRole") == "merging"
        assert attrs.get("trustScore") == "85"
        # collaborationMode and mergingStrategy no longer emitted (P3').
        for forbidden in ("reflectionMode", "role", "collaborationMode", "mergingStrategy"):
            assert forbidden not in attrs

    def test_agentic_gateway_diverging_emits_attrs(self, tmp_path):  # G-5
        root = _parse(_generate(_agentic_gateway_diverging_model(), tmp_path))
        inner = _agentic_inner(root)
        assert len(inner) == 1
        attrs = inner[0].attrib
        assert attrs.get("gatewayRole") == "diverging"
        assert attrs.get("trustScore") == "85"
        for forbidden in ("mergingStrategy", "collaborationMode"):
            assert forbidden not in attrs, (
                f"diverging AgenticGateway must NOT emit {forbidden}"
            )


# ---------------------------------------------------------------------------
# G-6 -- AgenticLane emission
# ---------------------------------------------------------------------------

class TestAgenticLaneEmission:
    def test_agentic_lane_emits_extension_element(self, tmp_path):  # G-6
        root = _parse(_generate(_agentic_lane_model(), tmp_path))
        inner = _agentic_inner(root)
        assert len(inner) == 1
        attrs = inner[0].attrib
        assert attrs.get("role") == "manager"
        assert attrs.get("trustScore") == "85"
        # Task/gateway-only attrs absent.
        for forbidden in ("reflectionMode", "gatewayRole",
                          "collaborationMode", "mergingStrategy"):
            assert forbidden not in attrs
        # 3c: default-1 lane carries no multiplicity attribute.
        assert "multiplicity" not in attrs

    def test_agentic_lane_emits_multiplicity_when_gt_one(self, tmp_path):  # 3c
        t = Task(name="Code")
        lane = AgenticLane(name="Reviewer", role=AgentRole.MANAGER,
                           trust_score=85, multiplicity=3, flow_nodes={t})
        p = Process(name="P", flow_nodes={t}, lanes={lane})
        model = BPMNModel(name="AgLaneSwarm", processes={p})
        root = _parse(_generate(model, tmp_path))
        attrs = _agentic_inner(root)[0].attrib
        assert attrs.get("multiplicity") == "3"

    def test_agentic_lane_omits_multiplicity_when_one(self, tmp_path):  # 3c
        # _agentic_lane_model() defaults multiplicity to 1.
        root = _parse(_generate(_agentic_lane_model(), tmp_path))
        attrs = _agentic_inner(root)[0].attrib
        assert "multiplicity" not in attrs


# ---------------------------------------------------------------------------
# G-7 / G-8 -- schema first-child order
# ---------------------------------------------------------------------------

class TestSchemaFirstChildOrder:
    def test_extension_element_is_first_child_of_flow_node(self, tmp_path):  # G-7
        root = _parse(_generate(_agentic_task_model(), tmp_path))
        # Find the userTask element (AgenticTask with task_type=USER -> userTask).
        task = _find(root, "bpmn:userTask")
        assert task is not None
        assert len(list(task)) >= 1
        first_child = list(task)[0]
        assert first_child.tag == f"{{{_NS['bpmn']}}}extensionElements", (
            f"extensionElements must be the first child of tFlowNode; got {first_child.tag}"
        )

    def test_extension_element_is_first_child_of_lane(self, tmp_path):  # G-8
        root = _parse(_generate(_agentic_lane_model(), tmp_path))
        lane = _find(root, "bpmn:lane")
        assert lane is not None
        assert len(list(lane)) >= 1
        first_child = list(lane)[0]
        assert first_child.tag == f"{{{_NS['bpmn']}}}extensionElements", (
            f"extensionElements must be the first child of tLane; got {first_child.tag}"
        )


# ---------------------------------------------------------------------------
# G-9 / G-10 -- vanilla untouched, mixed models
# ---------------------------------------------------------------------------

class TestVanillaAndMixed:
    @pytest.mark.parametrize("fixture_fn", [_vanilla_model, _vanilla_lane_model])
    def test_vanilla_model_has_no_extension_elements(self, fixture_fn, tmp_path):  # G-9
        root = _parse(_generate(fixture_fn(), tmp_path))
        assert _agentic_inner(root) == []
        assert _findall(root, "bpmn:extensionElements") == []

    def test_mixed_model_emits_one_extension_block_only(self, tmp_path):  # G-10
        root = _parse(_generate(_mixed_model(), tmp_path))
        # Exactly one <agentic:agentic> for the AgenticTask; the vanilla Task
        # remains untouched.
        inner = _agentic_inner(root)
        assert len(inner) == 1
        assert inner[0].attrib.get("reflectionMode") == "self"
        assert inner[0].attrib.get("trustScore") == "50"


# ---------------------------------------------------------------------------
# G-11 -- enum values match WME's lowercase strings
# ---------------------------------------------------------------------------

class TestEnumValueStrings:
    @pytest.mark.parametrize("reflection,expected", [
        (ReflectionMode.NONE, "none"),
        (ReflectionMode.SELF, "self"),
        (ReflectionMode.CROSS, "cross"),
        (ReflectionMode.HUMAN, "human"),
    ])
    def test_reflection_mode_strings(self, reflection, expected, tmp_path):  # G-11a
        t = AgenticTask(name="X", reflection_mode=reflection, trust_score=10)
        p = Process(name="P", flow_nodes={t})
        model = BPMNModel(name="M", processes={p})
        root = _parse(_generate(model, tmp_path))
        assert _agentic_inner(root)[0].attrib["reflectionMode"] == expected

    @pytest.mark.parametrize("role,expected", [
        (AgentRole.WORKER, "worker"),
        (AgentRole.MANAGER, "manager"),
    ])
    def test_role_strings(self, role, expected, tmp_path):  # G-11c
        t = Task(name="x")
        lane = AgenticLane(name="L", role=role, trust_score=10, flow_nodes={t})
        p = Process(name="P", flow_nodes={t}, lanes={lane})
        model = BPMNModel(name="M", processes={p})
        root = _parse(_generate(model, tmp_path))
        assert _agentic_inner(root)[0].attrib["role"] == expected


# ---------------------------------------------------------------------------
# G-12 -- determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("fixture_fn", [
        _agentic_task_model,
        _agentic_gateway_merging_model,
        _agentic_gateway_diverging_model,
        _agentic_lane_model,
        _mixed_model,
    ])
    def test_same_agentic_model_produces_same_bytes(self, fixture_fn, tmp_path):  # G-12
        # The model must be the SAME instance -- sort_by_timestamp keys off the
        # NamedElement timestamp; building a fresh model would give fresh
        # timestamps (mirrors test_bpmn_generator.test_same_model_same_bytes).
        model = fixture_fn()
        out1 = tmp_path / "first.bpmn"
        out2 = tmp_path / "second.bpmn"
        BPMNGenerator(model, output_dir=str(tmp_path),
                      file_name="first.bpmn").generate()
        BPMNGenerator(model, output_dir=str(tmp_path),
                      file_name="second.bpmn").generate()
        assert out1.read_bytes() == out2.read_bytes()


# ---------------------------------------------------------------------------
# G-13 -- OQ-5 generator catch-up: task agentDiagramRef, gateway governance
# ---------------------------------------------------------------------------

_GOV_DSL = "Scopes:\n    Tasks:\n        Merge\nMajorityPolicy P {\n    ratio : 0.5\n}"


class TestOQ5Emission:
    def test_task_emits_agent_diagram_ref(self, tmp_path):  # G-13a
        t = AgenticTask(name="Review", task_type=TaskType.USER,
                        reflection_mode=ReflectionMode.CROSS, trust_score=80,
                        agent_diagram_ref="agent-uuid-123")
        model = BPMNModel(name="M", processes={Process(name="P", flow_nodes={t})})
        inner = _agentic_inner(_parse(_generate(model, tmp_path)))[0]
        assert inner.attrib.get("agentDiagramRef") == "agent-uuid-123"

    def test_gateway_emits_governance_child(self, tmp_path):  # G-13b
        g = AgenticGateway(name="Vote", gateway_type=GatewayType.PARALLEL,
                           gateway_role=GatewayRole.MERGING,
                           governance_dsl=_GOV_DSL)
        model = BPMNModel(name="M", processes={Process(name="P", flow_nodes={g})})
        root = _parse(_generate(model, tmp_path))
        gov = _findall(root, "agentic:governance")
        assert len(gov) == 1
        assert gov[0].text == _GOV_DSL  # verbatim (ET.indent leaves leaf text alone)

    def test_gateway_without_governance_emits_no_child(self, tmp_path):  # G-13c
        root = _parse(_generate(_agentic_gateway_merging_model(), tmp_path))
        assert _findall(root, "agentic:governance") == []

    def test_base_message_flow_emits_no_extension(self, tmp_path):  # G-13d (updated P3')
        """A base MessageFlow no longer gets an agentic extension element."""
        t1, t2 = Task(name="T1"), Task(name="T2")
        p1 = Process(name="proc1", flow_nodes={t1})
        p2 = Process(name="proc2", flow_nodes={t2})
        part1 = Participant(name="P1", process=p1)
        part2 = Participant(name="P2", process=p2)
        mf = MessageFlow(source=t1, target=t2)
        coll = Collaboration(name="C", participants={part1, part2}, message_flows={mf})
        model = BPMNModel(name="MFModel", processes={p1, p2}, collaboration=coll)
        root = _parse(_generate(model, tmp_path))
        # No agentic extension on a base message flow.
        inner = _agentic_inner(root)
        assert len(inner) == 0
