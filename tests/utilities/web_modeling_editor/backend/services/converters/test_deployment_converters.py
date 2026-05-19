"""Tests for the Deployment diagram converters (02-... §12).

Covers:
  * JSON -> ``DeploymentModel`` (``process_deployment_diagram``).
  * ``DeploymentModel`` -> JSON (``deployment_object_to_json``).
  * ``DeploymentAssociation`` discrimination (Artifact->Node vs Node<->Node).
  * ``DeploymentComponent`` synthesis (D12) round-trip.
  * Multiplicity parser / formatter.
  * Owner-link dedup vs explicit ``DeploymentAssociation``.
"""

import pytest

from besser.BUML.metamodel.structural import Multiplicity, UNLIMITED_MAX_MULTIPLICITY
from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentModel,
    DeploymentRelation,
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
# Fixtures (mirror 02-D7-Deployment.json shape + the hand-built
# multiplicity snapshot b2-deployment-snapshot.json)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_deployment_diagram():
    """Real-export style: Node holding nested Component, plus free-standing
    Component / Artifact / Interface, with all four relationship types."""
    return {
        "title": "Deployment Diagram",
        "model": {
            "version": "3.0.0",
            "type": "DeploymentDiagram",
            "size": {"width": 1040, "height": 640},
            "elements": {
                "n1": {
                    "id": "n1", "name": "ServerHost", "type": "DeploymentNode",
                    "owner": None,
                    "bounds": {"x": -500, "y": -300, "width": 440, "height": 340},
                    "stereotype": "node", "displayStereotype": True,
                },
                "cmp-inside": {
                    "id": "cmp-inside", "name": "AuthService",
                    "type": "DeploymentComponent",
                    "owner": "n1",
                    "bounds": {"x": -450, "y": -200, "width": 160, "height": 100},
                    "displayStereotype": True,
                },
                "cmp-free": {
                    "id": "cmp-free", "name": "Reporter",
                    "type": "DeploymentComponent",
                    "owner": None,
                    "bounds": {"x": 150, "y": -190, "width": 160, "height": 100},
                    "displayStereotype": True,
                },
                "n2": {
                    "id": "n2", "name": "Cache", "type": "DeploymentNode",
                    "owner": None,
                    "bounds": {"x": 230, "y": 0, "width": 160, "height": 100},
                    "stereotype": "node", "displayStereotype": True,
                },
                "a1": {
                    "id": "a1", "name": "ReportArchive",
                    "type": "DeploymentArtifact",
                    "owner": None,
                    "bounds": {"x": -250, "y": 120, "width": 160, "height": 40},
                },
                "i1": {
                    "id": "i1", "name": "MetricsApi",
                    "type": "DeploymentInterface",
                    "owner": None,
                    "bounds": {"x": -10, "y": 120, "width": 20, "height": 20},
                },
            },
            "relationships": {
                "r-dep": {
                    "id": "r-dep", "name": "", "type": "DeploymentDependency",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 440, "height": 1},
                    "path": [{"x": 0, "y": 0}, {"x": 440, "y": 0}],
                    "source": {"direction": "Right", "element": "cmp-inside"},
                    "target": {"direction": "Left", "element": "cmp-free"},
                    "isManuallyLayouted": False,
                },
                "r-assoc": {
                    "id": "r-assoc", "name": "", "type": "DeploymentAssociation",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 290, "height": 1},
                    "path": [{"x": 0, "y": 0}, {"x": 290, "y": 0}],
                    "source": {"direction": "Downright", "element": "n1"},
                    "target": {"direction": "Left", "element": "n2"},
                    "isManuallyLayouted": False,
                },
                "r-prov": {
                    "id": "r-prov", "name": "",
                    "type": "DeploymentInterfaceProvided",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 280, "height": 225},
                    "path": [{"x": 280, "y": 0}, {"x": 40, "y": 225}],
                    "source": {"direction": "Down", "element": "cmp-free"},
                    "target": {"direction": "Downleft", "element": "i1"},
                    "isManuallyLayouted": False,
                },
                "r-req": {
                    "id": "r-req", "name": "",
                    "type": "DeploymentInterfaceRequired",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 99, "height": 63},
                    "path": [{"x": 0, "y": 23}, {"x": 85, "y": 13}],
                    "source": {"direction": "Downright", "element": "a1"},
                    "target": {"direction": "Bottomleft", "element": "i1"},
                    "isManuallyLayouted": False,
                },
            },
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


@pytest.fixture
def multiplicity_deployment_diagram():
    """Hand-built style: `device` Node holding `[3]`-multiplicity artifact,
    plus a Node-Node DeploymentAssociation (CommunicationPath)."""
    return {
        "title": "Deployment Diagram",
        "model": {
            "version": "3.0.0",
            "type": "DeploymentDiagram",
            "size": {"width": 1400, "height": 740},
            "elements": {
                "elem-edge-device": {
                    "id": "elem-edge-device",
                    "name": "EdgeDevice", "type": "DeploymentNode",
                    "owner": None,
                    "bounds": {"x": 100, "y": 100, "width": 320, "height": 260},
                    "stereotype": "device", "displayStereotype": True,
                },
                "elem-agent-runtime": {
                    "id": "elem-agent-runtime",
                    "name": "agent-runtime [3]", "type": "DeploymentArtifact",
                    "owner": "elem-edge-device",
                    "bounds": {"x": 140, "y": 180, "width": 200, "height": 60},
                },
                "elem-cloud": {
                    "id": "elem-cloud",
                    "name": "Cloud", "type": "DeploymentNode",
                    "owner": None,
                    "bounds": {"x": 620, "y": 100, "width": 280, "height": 200},
                    "stereotype": "node", "displayStereotype": True,
                },
            },
            "relationships": {
                "rel-artifact-deploys": {
                    "id": "rel-artifact-deploys",
                    "name": "", "type": "DeploymentAssociation",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 1, "height": 120},
                    "path": [{"x": 0, "y": 0}, {"x": 0, "y": 120}],
                    "source": {
                        "element": "elem-agent-runtime", "direction": "Down",
                    },
                    "target": {
                        "element": "elem-edge-device", "direction": "Down",
                    },
                    "isManuallyLayouted": False,
                    "stereotype": "",
                },
                "rel-comm": {
                    "id": "rel-comm",
                    "name": "", "type": "DeploymentAssociation",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 200, "height": 1},
                    "path": [{"x": 0, "y": 0}, {"x": 200, "y": 0}],
                    "source": {
                        "element": "elem-edge-device", "direction": "Right",
                    },
                    "target": {
                        "element": "elem-cloud", "direction": "Left",
                    },
                    "isManuallyLayouted": False,
                    "stereotype": "",
                },
            },
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


# ---------------------------------------------------------------------------
# JSON -> BUML
# ---------------------------------------------------------------------------

class TestProcessDeploymentDiagram:
    def test_real_diagram_constructs(self, real_deployment_diagram):
        model = process_deployment_diagram(real_deployment_diagram)
        assert isinstance(model, DeploymentModel)
        # 2 Nodes (ServerHost, Cache); 2 Artifacts from DeploymentComponent
        # synthesis + 1 native ReportArchive = 3 artifacts total; 1 Interface.
        assert len(model.all_nodes()) == 2
        assert len(model.all_artifacts()) == 3
        assert len(model.interfaces) == 1

    def test_deployment_component_synthesises_artifact_with_manifests(
        self, real_deployment_diagram,
    ):
        model = process_deployment_diagram(real_deployment_diagram)
        # The nested DeploymentComponent (cmp-inside) becomes a synthetic
        # Artifact with manifests=["cmp-inside"].
        artifacts_with_manifest = [
            a for a in model.all_artifacts() if a.manifests
        ]
        assert len(artifacts_with_manifest) == 2
        manifest_ids = sorted(
            {m for a in artifacts_with_manifest for m in a.manifests}
        )
        assert manifest_ids == ["cmp-free", "cmp-inside"]

    def test_owner_chain_nests_artifact_in_node(self, real_deployment_diagram):
        model = process_deployment_diagram(real_deployment_diagram)
        server_host = next(n for n in model.all_nodes() if n.name == "ServerHost")
        # cmp-inside is the synthetic Artifact for AuthService.
        nested_names = [a.name for a in server_host.nested_artifacts]
        assert "AuthService" in nested_names

    def test_multiplicity_parses_from_name(self, multiplicity_deployment_diagram):
        model = process_deployment_diagram(multiplicity_deployment_diagram)
        # Artifact name parses to clean name + Multiplicity(3, 3) on the
        # synthesised/explicit DeploymentRelation.
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

    def test_owner_link_and_explicit_edge_dedup(self, multiplicity_deployment_diagram):
        """When both owner-nesting and an explicit DeploymentAssociation
        encode the same artifact-on-node, only one DeploymentRelation is
        emitted (the explicit one wins; the owner-link is dedup'd)."""
        model = process_deployment_diagram(multiplicity_deployment_diagram)
        relations = [r for r in model.relationships
                     if isinstance(r, DeploymentRelation)]
        # The fixture has an artifact owned by a node AND an explicit edge
        # → exactly one DeploymentRelation, not two.
        assert len(relations) == 1
        assert relations[0].source.name == "agent-runtime"
        assert relations[0].target.name == "EdgeDevice"


# ---------------------------------------------------------------------------
# BUML -> JSON
# ---------------------------------------------------------------------------

class TestDeploymentObjectToJson:
    def test_envelope_shape(self):
        model = DeploymentModel(name="Empty")
        out = deployment_object_to_json(model)
        assert out["type"] == "DeploymentDiagram"
        assert out["version"] == "3.0.0"
        assert out["elements"] == {}
        assert out["relationships"] == {}
        assert out["interactive"] == {"elements": {}, "relationships": {}}
        assert out["assessments"] == {}

    def test_emits_node_with_kind_stereotype(self):
        node = Node(name="EdgeDevice", kind=NodeKind.DEVICE)
        model = DeploymentModel(name="m", nodes={node})
        out = deployment_object_to_json(model)
        entries = [e for e in out["elements"].values()
                   if e["type"] == "DeploymentNode"]
        assert len(entries) == 1
        assert entries[0]["stereotype"] == "device"

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
        artifact_entry = [e for e in out["elements"].values()
                          if e["type"] == "DeploymentArtifact"][0]
        assert artifact_entry["name"] == "worker [3]"

    def test_artifact_default_multiplicity_no_suffix(self):
        node = Node(name="VM")
        artifact = Artifact(name="worker")
        # Default Multiplicity(1, 1)
        rel = DeploymentRelation(source=artifact, target=node)
        model = DeploymentModel(
            name="m", nodes={node}, artifacts={artifact}, relationships={rel},
        )
        out = deployment_object_to_json(model)
        artifact_entry = [e for e in out["elements"].values()
                          if e["type"] == "DeploymentArtifact"][0]
        assert artifact_entry["name"] == "worker"

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
        assoc_types = {r["type"] for r in out["relationships"].values()}
        assert assoc_types == {"DeploymentAssociation"}


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_real_diagram_ids_survive(self, real_deployment_diagram):
        model = process_deployment_diagram(real_deployment_diagram)
        out = deployment_object_to_json(model)
        in_elements = sorted(real_deployment_diagram["model"]["elements"].keys())
        out_elements = sorted(out["elements"].keys())
        assert in_elements == out_elements

    def test_real_diagram_types_survive(self, real_deployment_diagram):
        model = process_deployment_diagram(real_deployment_diagram)
        out = deployment_object_to_json(model)
        for elem_id, original in (
            real_deployment_diagram["model"]["elements"].items()
        ):
            assert out["elements"][elem_id]["type"] == original["type"]

    def test_real_diagram_owner_chains_survive(self, real_deployment_diagram):
        model = process_deployment_diagram(real_deployment_diagram)
        out = deployment_object_to_json(model)
        for elem_id, original in (
            real_deployment_diagram["model"]["elements"].items()
        ):
            assert out["elements"][elem_id].get("owner") == original.get("owner")

    def test_multiplicity_round_trip_preserves_suffix(
        self, multiplicity_deployment_diagram,
    ):
        model = process_deployment_diagram(multiplicity_deployment_diagram)
        out = deployment_object_to_json(model)
        artifact = out["elements"]["elem-agent-runtime"]
        assert artifact["name"] == "agent-runtime [3]"


# ---------------------------------------------------------------------------
# Multiplicity formatter / parser unit tests (02-... §3.6.2)
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
# deployment_buml_to_json exec wrapper (03-... §7)
# ---------------------------------------------------------------------------

class TestDeploymentBumlToJson:
    def test_round_trips_via_builder(self, real_deployment_diagram):
        """Full pipeline: JSON -> DeploymentModel -> .py source -> JSON."""
        model = process_deployment_diagram(real_deployment_diagram)
        source = deployment_model_to_code(model)
        result = deployment_buml_to_json(source)
        assert result["type"] == "DeploymentDiagram"
        in_ids = sorted(real_deployment_diagram["model"]["elements"].keys())
        out_ids = sorted(result["elements"].keys())
        assert in_ids == out_ids

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

    def test_runtime_error_propagates(self):
        with pytest.raises(RuntimeError):
            deployment_buml_to_json("raise RuntimeError('boom')")

    def test_multiplicity_survives_full_pipeline(self, multiplicity_deployment_diagram):
        """The [3] suffix should round-trip through the .py builder cleanly."""
        model = process_deployment_diagram(multiplicity_deployment_diagram)
        source = deployment_model_to_code(model)
        result = deployment_buml_to_json(source)
        # Find the artifact in the result.
        artifacts = [e for e in result["elements"].values()
                     if e["type"] == "DeploymentArtifact"]
        assert len(artifacts) == 1
        assert artifacts[0]["name"] == "agent-runtime [3]"
