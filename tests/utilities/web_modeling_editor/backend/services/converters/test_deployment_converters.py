"""Tests for the Deployment diagram converters (v4 ``{nodes, edges}`` wire shape).

Covers:
  * JSON -> ``DeploymentModel`` (``process_deployment_diagram``).
  * ``DeploymentModel`` -> JSON (``deployment_object_to_json``).
  * ``DeploymentAssociation`` discrimination (Artifact->Node vs Node<->Node).
  * ``DeploymentComponent`` round-trip.
  * Multiplicity parser / formatter.
  * Parent-link dedup vs explicit ``DeploymentAssociation``.
"""

import pytest

from besser.BUML.metamodel.structural import Multiplicity, UNLIMITED_MAX_MULTIPLICITY
from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentComponent,
    DeploymentModel,
    DeploymentRelation,
    InterfaceRequired,
    Node,
    NodeKind,
)
from besser.utilities.buml_code_builder.deployment_model_builder import (
    deployment_model_to_code,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.deployment_diagram_converter import (
    deployment_buml_to_json,
    deployment_object_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.deployment_diagram_processor import (
    process_deployment_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.multiplicity_format import (
    format_to_name,
    parse_from_name,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
)


# ---------------------------------------------------------------------------
# v4 fixture builders
# ---------------------------------------------------------------------------

def _node(node_id, node_type, x, y, width, height, data, parent=None):
    node = {
        "id": node_id, "type": node_type,
        "position": {"x": x, "y": y},
        "width": width, "height": height,
        "measured": {"width": width, "height": height},
        "data": data,
    }
    if parent is not None:
        node["parentId"] = parent
    return node


def _edge(edge_id, edge_type, source, target, points, stereotype=None,
          source_handle="right", target_handle="left"):
    data = {"name": "", "label": "", "points": points, "isManuallyLayouted": False}
    if stereotype is not None:
        data["stereotype"] = stereotype
    return {
        "id": edge_id, "source": source, "target": target, "type": edge_type,
        "sourceHandle": source_handle, "targetHandle": target_handle, "data": data,
    }


def _diagram(title, nodes, edges):
    return {
        "title": title,
        "model": {
            "version": "4.0.0", "type": "DeploymentDiagram", "title": title,
            "nodes": nodes, "edges": edges,
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


def _nodes_by_id(payload):
    return {n["id"]: n for n in payload["nodes"]}


# ---------------------------------------------------------------------------
# Fixtures (mirror the editor's real Deployment-diagram export and a
# hand-built multiplicity diagram)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_deployment_diagram():
    """Real-export style: Node holding a nested Component, plus free-standing
    Component / Artifact / Interface, with all four relationship types."""
    return _diagram(
        "Deployment Diagram",
        [
            _node("n1", "deploymentNode", -500, -300, 440, 340,
                  {"name": "ServerHost", "stereotype": "node", "isComponentHeaderShown": True}),
            _node("cmp-inside", "deploymentComponent", 50, 100, 160, 100,
                  {"name": "AuthService", "stereotype": "", "isComponentHeaderShown": True},
                  parent="n1"),
            _node("cmp-free", "deploymentComponent", 150, -190, 160, 100,
                  {"name": "Reporter", "stereotype": "", "isComponentHeaderShown": True}),
            _node("n2", "deploymentNode", 230, 0, 160, 100,
                  {"name": "Cache", "stereotype": "node", "isComponentHeaderShown": True}),
            _node("a1", "deploymentArtifact", -250, 120, 160, 40,
                  {"name": "ReportArchive", "manifests": []}),
            _node("i1", "deploymentInterface", -10, 120, 20, 20, {"name": "MetricsApi"}),
        ],
        [
            _edge("r-dep", "DeploymentDependency", "cmp-inside", "cmp-free",
                  [{"x": -290, "y": -150}, {"x": 150, "y": -150}]),
            _edge("r-assoc", "DeploymentAssociation", "n1", "n2",
                  [{"x": -60, "y": 40}, {"x": 230, "y": 40}],
                  source_handle="right-bottom"),
            _edge("r-prov", "DeploymentProvidedInterface", "cmp-free", "i1",
                  [{"x": 230, "y": -90}, {"x": 0, "y": 140}],
                  source_handle="bottom", target_handle="left-bottom"),
            _edge("r-req", "DeploymentRequiredInterface", "a1", "i1",
                  [{"x": -90, "y": 160}, {"x": -10, "y": 140}],
                  source_handle="right-bottom", target_handle="bottom-left"),
        ],
    )


@pytest.fixture
def multiplicity_deployment_diagram():
    """Hand-built style: `device` Node holding a `[3]`-multiplicity artifact,
    plus a Node-Node DeploymentAssociation (CommunicationPath)."""
    return _diagram(
        "Deployment Diagram",
        [
            _node("elem-edge-device", "deploymentNode", 100, 100, 320, 260,
                  {"name": "EdgeDevice", "stereotype": "device", "isComponentHeaderShown": True}),
            _node("elem-agent-runtime", "deploymentArtifact", 40, 80, 200, 60,
                  {"name": "agent-runtime [3]", "manifests": []},
                  parent="elem-edge-device"),
            _node("elem-cloud", "deploymentNode", 620, 100, 280, 200,
                  {"name": "Cloud", "stereotype": "node", "isComponentHeaderShown": True}),
        ],
        [
            _edge("rel-artifact-deploys", "DeploymentAssociation",
                  "elem-agent-runtime", "elem-edge-device",
                  [{"x": 240, "y": 240}, {"x": 240, "y": 360}], stereotype="",
                  source_handle="bottom", target_handle="bottom"),
            _edge("rel-comm", "DeploymentAssociation", "elem-edge-device", "elem-cloud",
                  [{"x": 420, "y": 200}, {"x": 620, "y": 200}], stereotype=""),
        ],
    )


# ---------------------------------------------------------------------------
# JSON -> BUML
# ---------------------------------------------------------------------------

class TestProcessDeploymentDiagram:
    def test_real_diagram_constructs(self, real_deployment_diagram):
        model = process_deployment_diagram(real_deployment_diagram)
        assert isinstance(model, DeploymentModel)
        # 2 Nodes (ServerHost, Cache); 2 DeploymentComponents + 1 native
        # ReportArchive = 3 artifacts total; 1 Interface.
        assert len(model.all_nodes()) == 2
        assert len(model.all_artifacts()) == 3
        assert len(model.interfaces) == 1

    def test_deployment_component_becomes_deployment_component_without_manifests(
        self, real_deployment_diagram,
    ):
        model = process_deployment_diagram(real_deployment_diagram)
        components = [a for a in model.all_artifacts()
                      if isinstance(a, DeploymentComponent)]
        assert sorted(c.name for c in components) == ["AuthService", "Reporter"]
        # The node's own id is never fabricated into manifests.
        assert all(c.manifests == [] for c in components)

    def test_artifact_manifests_are_read_from_the_node(self):
        payload = _diagram("T", [_node("a1", "deploymentArtifact", 0, 0, 160, 40,
                                       {"name": "Coder", "manifests": ["component-1"]})], [])
        (artifact,) = process_deployment_diagram(payload).all_artifacts()
        assert type(artifact) is Artifact
        assert artifact.manifests == ["component-1"]

    def test_parent_chain_nests_artifact_in_node(self, real_deployment_diagram):
        model = process_deployment_diagram(real_deployment_diagram)
        server_host = next(n for n in model.all_nodes() if n.name == "ServerHost")
        # cmp-inside is the DeploymentComponent AuthService.
        nested_names = [a.name for a in server_host.nested_artifacts]
        assert "AuthService" in nested_names

    def test_multiplicity_parses_from_name(self, multiplicity_deployment_diagram):
        model = process_deployment_diagram(multiplicity_deployment_diagram)
        # Artifact name parses to clean name + Multiplicity(3, 3) on the
        # explicit DeploymentRelation.
        artifact = next(a for a in model.all_artifacts())
        assert artifact.name == "agent-runtime"
        rels = [r for r in model.relationships if isinstance(r, DeploymentRelation)]
        assert len(rels) == 1
        rel = rels[0]
        assert rel.multiplicity.min == 3
        assert rel.multiplicity.max == 3

    def test_communication_path_discriminated_from_association(
        self, multiplicity_deployment_diagram,
    ):
        model = process_deployment_diagram(multiplicity_deployment_diagram)
        comm_paths = [r for r in model.relationships
                      if isinstance(r, CommunicationPath)]
        assert len(comm_paths) == 1
        comm = comm_paths[0]
        assert {comm.source.name, comm.target.name} == {"EdgeDevice", "Cloud"}

    def test_node_kind_resolved_from_stereotype(self, multiplicity_deployment_diagram):
        model = process_deployment_diagram(multiplicity_deployment_diagram)
        edge = next(n for n in model.all_nodes() if n.name == "EdgeDevice")
        cloud = next(n for n in model.all_nodes() if n.name == "Cloud")
        assert edge.kind is NodeKind.DEVICE
        assert cloud.kind is NodeKind.GENERIC

    def test_parent_link_and_explicit_edge_dedup(self, multiplicity_deployment_diagram):
        """When both parent nesting and an explicit DeploymentAssociation
        encode the same artifact-on-node, only one DeploymentRelation is
        emitted (the explicit one wins; the parent link is dedup'd)."""
        model = process_deployment_diagram(multiplicity_deployment_diagram)
        relations = [r for r in model.relationships
                     if isinstance(r, DeploymentRelation)]
        # The fixture has an artifact inside a node AND an explicit edge
        # → exactly one DeploymentRelation, not two.
        assert len(relations) == 1
        assert relations[0].source.name == "agent-runtime"
        assert relations[0].target.name == "EdgeDevice"

    def test_required_interface_socket_variant_reads_as_required(self):
        payload = _diagram(
            "Sockets",
            [
                _node("a", "deploymentArtifact", 0, 0, 160, 40, {"name": "App", "manifests": []}),
                _node("i", "deploymentInterface", 300, 10, 20, 20, {"name": "Port"}),
            ],
            [_edge("req", "DeploymentRequiredThreeQuarterInterface", "a", "i",
                   [{"x": 160, "y": 20}, {"x": 300, "y": 20}])],
        )
        model = process_deployment_diagram(payload)
        (rel,) = model.relationships
        assert isinstance(rel, InterfaceRequired)
        out = deployment_object_to_json(model)
        assert out["edges"][0]["type"] == "DeploymentRequiredThreeQuarterInterface"


# ---------------------------------------------------------------------------
# BUML -> JSON
# ---------------------------------------------------------------------------

class TestDeploymentObjectToJson:
    def test_envelope_shape(self):
        model = DeploymentModel(name="Empty")
        out = deployment_object_to_json(model)
        assert out["type"] == "DeploymentDiagram"
        assert out["version"] == "4.0.0"
        assert out["title"] == "Empty"
        assert out["nodes"] == []
        assert out["edges"] == []
        assert out["interactive"] == {"elements": {}, "relationships": {}}
        assert out["assessments"] == {}

    def test_emits_node_with_kind_stereotype(self):
        node = Node(name="EdgeDevice", kind=NodeKind.DEVICE)
        model = DeploymentModel(name="m", nodes={node})
        out = deployment_object_to_json(model)
        entries = [n for n in out["nodes"] if n["type"] == "deploymentNode"]
        assert len(entries) == 1
        assert entries[0]["data"]["stereotype"] == "device"

    def test_artifact_emits_multiplicity_in_name(self):
        node = Node(name="VM")
        artifact = Artifact(name="worker")
        rel = DeploymentRelation(
            source=artifact, target=node, multiplicity=Multiplicity(3, 3),
        )
        model = DeploymentModel(
            name="m", nodes={node}, artifacts={artifact}, relationships={rel},
        )
        out = deployment_object_to_json(model)
        artifact_entry = [n for n in out["nodes"] if n["type"] == "deploymentArtifact"][0]
        assert artifact_entry["data"]["name"] == "worker [3]"

    def test_artifact_default_multiplicity_no_suffix(self):
        node = Node(name="VM")
        artifact = Artifact(name="worker")
        # Default Multiplicity(1, 1)
        rel = DeploymentRelation(source=artifact, target=node)
        model = DeploymentModel(
            name="m", nodes={node}, artifacts={artifact}, relationships={rel},
        )
        out = deployment_object_to_json(model)
        artifact_entry = [n for n in out["nodes"] if n["type"] == "deploymentArtifact"][0]
        assert artifact_entry["data"]["name"] == "worker"

    def test_deployment_relation_and_communication_path_both_emit_association(self):
        n1 = Node(name="N1")
        n2 = Node(name="N2")
        a = Artifact(name="A")
        rel = DeploymentRelation(source=a, target=n1)
        comm = CommunicationPath(source=n1, target=n2)
        model = DeploymentModel(
            name="m", nodes={n1, n2}, artifacts={a},
            relationships={rel, comm},
        )
        out = deployment_object_to_json(model)
        assoc_types = {e["type"] for e in out["edges"]}
        assert assoc_types == {"DeploymentAssociation"}

    def test_nested_model_without_layout_places_children_inside_parents(self):
        """A model built in Python has no editor layout: a nested node and its
        artifact sit inside their parents, parents first."""
        host = Node(name="Host")
        runtime = Node(name="Runtime")
        host.add_nested_node(runtime)
        artifact = Artifact(name="app")
        runtime.add_artifact(artifact)
        model = DeploymentModel(name="m", nodes={host}, artifacts=set())
        out = deployment_object_to_json(model)
        nodes = {n["data"]["name"]: n for n in out["nodes"]}
        order = [n["data"]["name"] for n in out["nodes"]]
        assert order.index("Host") < order.index("Runtime") < order.index("app")
        for child_name, parent_name in (("Runtime", "Host"), ("app", "Runtime")):
            child, parent = nodes[child_name], nodes[parent_name]
            assert child["parentId"] == parent["id"]
            assert child["position"]["x"] + child["width"] <= parent["width"]
            assert child["position"]["y"] + child["height"] <= parent["height"]


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_real_diagram_ids_survive(self, real_deployment_diagram):
        model = process_deployment_diagram(real_deployment_diagram)
        out = deployment_object_to_json(model)
        assert sorted(_nodes_by_id(out)) == sorted(_nodes_by_id(real_deployment_diagram["model"]))
        assert sorted(e["id"] for e in out["edges"]) == sorted(
            e["id"] for e in real_deployment_diagram["model"]["edges"])

    def test_real_diagram_types_survive(self, real_deployment_diagram):
        model = process_deployment_diagram(real_deployment_diagram)
        out_nodes = _nodes_by_id(deployment_object_to_json(model))
        for node_id, original in _nodes_by_id(real_deployment_diagram["model"]).items():
            assert out_nodes[node_id]["type"] == original["type"]

    def test_real_diagram_parent_chains_survive(self, real_deployment_diagram):
        model = process_deployment_diagram(real_deployment_diagram)
        out_nodes = _nodes_by_id(deployment_object_to_json(model))
        for node_id, original in _nodes_by_id(real_deployment_diagram["model"]).items():
            assert out_nodes[node_id].get("parentId") == original.get("parentId")
            assert out_nodes[node_id]["position"] == original["position"]

    def test_multiplicity_round_trip_preserves_suffix(
        self, multiplicity_deployment_diagram,
    ):
        model = process_deployment_diagram(multiplicity_deployment_diagram)
        out = deployment_object_to_json(model)
        artifact = _nodes_by_id(out)["elem-agent-runtime"]
        assert artifact["data"]["name"] == "agent-runtime [3]"


# ---------------------------------------------------------------------------
# Multiplicity formatter / parser unit tests
# ---------------------------------------------------------------------------

class TestMultiplicityFormat:
    @pytest.mark.parametrize("raw, expected_name, expected_lo, expected_hi", [
        ("Code Tester",          "Code Tester",      None, None),
        ("Code Tester [3]",      "Code Tester",      3,    3),
        ("Code Tester [1..*]",   "Code Tester",      1,    UNLIMITED_MAX_MULTIPLICITY),
        ("Code Tester [2..5]",   "Code Tester",      2,    5),
        ("Code Tester [0..1]",   "Code Tester",      0,    1),
        ("",                     "",                 None, None),
    ])
    def test_parse_from_name(self, raw, expected_name, expected_lo, expected_hi):
        name, mult = parse_from_name(raw)
        assert name == expected_name
        if expected_lo is None:
            assert mult is None
        else:
            assert mult.min == expected_lo
            assert mult.max == expected_hi

    def test_parse_non_matching_brackets_stays_in_name(self):
        # "Code [Tester" — opening bracket but no closing in the right place.
        name, mult = parse_from_name("Code [Tester")
        assert name == "Code [Tester"
        assert mult is None

    def test_format_to_name_default_no_suffix(self):
        assert format_to_name("worker", None) == "worker"
        assert format_to_name("worker", Multiplicity(1, 1)) == "worker"

    def test_format_to_name_with_multiplicity(self):
        assert format_to_name("worker", Multiplicity(3, 3)) == "worker [3]"
        assert format_to_name("worker", Multiplicity(2, 5)) == "worker [2..5]"
        assert (
            format_to_name("worker", Multiplicity(1, UNLIMITED_MAX_MULTIPLICITY))
            == "worker [1..*]"
        )


# ---------------------------------------------------------------------------
# deployment_buml_to_json exec wrapper
# ---------------------------------------------------------------------------

class TestDeploymentBumlToJson:
    def test_round_trips_via_builder(self, real_deployment_diagram):
        """Full pipeline: JSON -> DeploymentModel -> .py source -> JSON."""
        model = process_deployment_diagram(real_deployment_diagram)
        source = deployment_model_to_code(model)
        result = deployment_buml_to_json(source)
        assert result["type"] == "DeploymentDiagram"
        assert result["version"] == "4.0.0"
        assert sorted(_nodes_by_id(result)) == sorted(_nodes_by_id(real_deployment_diagram["model"]))

    def test_exec_failure_raises_conversion_error(self):
        with pytest.raises(ConversionError):
            deployment_buml_to_json("raise NameError('boom')")

    def test_missing_model_raises_conversion_error(self):
        with pytest.raises(ConversionError) as exc_info:
            deployment_buml_to_json("x = 1")
        assert "no DeploymentModel" in str(exc_info.value)

    def test_finds_model_under_any_var_name(self):
        source = (
            "from besser.BUML.metamodel.uml_deployment import DeploymentModel\n"
            "xyz = DeploymentModel(name='Found')\n"
        )
        result = deployment_buml_to_json(source)
        assert result["type"] == "DeploymentDiagram"

    def test_executable_statement_is_rejected(self):
        with pytest.raises(ConversionError, match="Top-level statement not allowed: Raise"):
            deployment_buml_to_json("raise RuntimeError('boom')")

    def test_multiplicity_survives_full_pipeline(self, multiplicity_deployment_diagram):
        """The [3] suffix should round-trip through the .py builder cleanly."""
        model = process_deployment_diagram(multiplicity_deployment_diagram)
        source = deployment_model_to_code(model)
        result = deployment_buml_to_json(source)
        # Find the artifact in the result.
        artifacts = [n for n in result["nodes"] if n["type"] == "deploymentArtifact"]
        assert len(artifacts) == 1
        assert artifacts[0]["data"]["name"] == "agent-runtime [3]"


# ---------------------------------------------------------------------------
# agentModelRef threading (processor reads the wire key onto the field)
# ---------------------------------------------------------------------------

class TestAgentModelRefProcessor:
    def _make_payload(self, node_type: str, agent_model_ref=None) -> dict:
        data = {"name": "Researcher"}
        if agent_model_ref is not None:
            data["agentModelRef"] = agent_model_ref
        return _diagram("T", [_node("elem-1", node_type, 0, 0, 160, 40, data)], [])

    def test_deployment_artifact_with_agent_model_ref(self):
        """deploymentArtifact with agentModelRef → artifact.agent_model_ref set."""
        payload = self._make_payload("deploymentArtifact", agent_model_ref="uuid-abc")
        model = process_deployment_diagram(payload)
        artifacts = list(model.all_artifacts())
        assert len(artifacts) == 1
        assert artifacts[0].agent_model_ref == "uuid-abc"

    def test_deployment_artifact_without_agent_model_ref(self):
        """deploymentArtifact with no agentModelRef → agent_model_ref is None."""
        payload = self._make_payload("deploymentArtifact")
        model = process_deployment_diagram(payload)
        artifacts = list(model.all_artifacts())
        assert len(artifacts) == 1
        assert artifacts[0].agent_model_ref is None

    def test_deployment_component_with_agent_model_ref(self):
        """deploymentComponent also threads agentModelRef."""
        payload = self._make_payload("deploymentComponent", agent_model_ref="uuid-xyz")
        model = process_deployment_diagram(payload)
        artifacts = list(model.all_artifacts())
        assert len(artifacts) == 1
        assert artifacts[0].agent_model_ref == "uuid-xyz"

    def test_deployment_component_without_agent_model_ref(self):
        """deploymentComponent with no agentModelRef → agent_model_ref is None."""
        payload = self._make_payload("deploymentComponent")
        model = process_deployment_diagram(payload)
        artifacts = list(model.all_artifacts())
        assert len(artifacts) == 1
        assert artifacts[0].agent_model_ref is None
