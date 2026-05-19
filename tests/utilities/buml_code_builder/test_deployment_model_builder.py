"""Tests for the Deployment model code builder (03-... §9.C).

Covers exec round-trip, multiplicity emission (including UNLIMITED upper
bound), DeploymentComponent synthesis preserved via layout, nested-Node
ordering, CommunicationPath, layout passthrough, manifests
dot-assignment, banner stability, determinism.
"""

import pytest

from besser.BUML.metamodel.structural import (
    Multiplicity,
    UNLIMITED_MAX_MULTIPLICITY,
)
from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentDependency,
    DeploymentModel,
    DeploymentRelation,
    Interface,
    InterfaceProvided,
    Locality,
    Node,
    NodeKind,
)
from besser.utilities.buml_code_builder.deployment_model_builder import (
    deployment_model_to_code,
)


def _exec_and_get_model(source: str) -> DeploymentModel:
    namespace: dict = {}
    exec(source, namespace)
    return namespace["deployment_model"]


class TestDeploymentModelBuilder:
    def test_empty_model_emits_valid_python(self):
        source = deployment_model_to_code(DeploymentModel(name="Empty"))
        compile(source, "<test>", "exec")
        rebuilt = _exec_and_get_model(source)
        assert isinstance(rebuilt, DeploymentModel)
        assert rebuilt.name == "Empty"

    def test_minimal_model_round_trips(self):
        n = Node(name="VM")
        a = Artifact(name="App")
        rel = DeploymentRelation(source=a, target=n)
        model = DeploymentModel(
            name="m", nodes={n}, artifacts={a}, relationships={rel},
        )
        source = deployment_model_to_code(model)
        rebuilt = _exec_and_get_model(source)
        assert len(rebuilt.all_nodes()) == 1
        assert len(rebuilt.all_artifacts()) == 1
        assert len(rebuilt.relationships) == 1

    def test_multiplicity_emitted_when_non_default(self):
        n = Node(name="VM")
        a = Artifact(name="worker")
        rel = DeploymentRelation(source=a, target=n, multiplicity=Multiplicity(3, 3))
        model = DeploymentModel(
            name="m", nodes={n}, artifacts={a}, relationships={rel},
        )
        source = deployment_model_to_code(model)
        assert "Multiplicity(3, 3)" in source
        rebuilt = _exec_and_get_model(source)
        rebuilt_rel = next(iter(rebuilt.relationships))
        assert rebuilt_rel.multiplicity.min == 3
        assert rebuilt_rel.multiplicity.max == 3

    def test_multiplicity_default_omitted(self):
        n = Node(name="VM")
        a = Artifact(name="App")
        rel = DeploymentRelation(source=a, target=n)  # default Multiplicity(1, 1)
        model = DeploymentModel(
            name="m", nodes={n}, artifacts={a}, relationships={rel},
        )
        source = deployment_model_to_code(model)
        # Compact output — default multiplicity isn't emitted.
        assert "Multiplicity(1, 1)" not in source
        rebuilt = _exec_and_get_model(source)
        rebuilt_rel = next(iter(rebuilt.relationships))
        assert rebuilt_rel.multiplicity.min == 1
        assert rebuilt_rel.multiplicity.max == 1

    def test_unlimited_multiplicity_emits_constant_import(self):
        n = Node(name="VM")
        a = Artifact(name="worker")
        rel = DeploymentRelation(
            source=a, target=n,
            multiplicity=Multiplicity(1, UNLIMITED_MAX_MULTIPLICITY),
        )
        model = DeploymentModel(
            name="m", nodes={n}, artifacts={a}, relationships={rel},
        )
        source = deployment_model_to_code(model)
        assert "UNLIMITED_MAX_MULTIPLICITY" in source
        assert "Multiplicity(1, UNLIMITED_MAX_MULTIPLICITY)" in source
        rebuilt = _exec_and_get_model(source)
        rebuilt_rel = next(iter(rebuilt.relationships))
        assert rebuilt_rel.multiplicity.max == UNLIMITED_MAX_MULTIPLICITY

    def test_unlimited_constant_not_imported_when_unused(self):
        n = Node(name="VM")
        a = Artifact(name="App")
        rel = DeploymentRelation(source=a, target=n, multiplicity=Multiplicity(3, 5))
        model = DeploymentModel(
            name="m", nodes={n}, artifacts={a}, relationships={rel},
        )
        source = deployment_model_to_code(model)
        # The import line only pulls in Multiplicity, not UNLIMITED constant.
        assert "Multiplicity" in source
        assert "UNLIMITED_MAX_MULTIPLICITY" not in source

    def test_communication_path_round_trips(self):
        n1 = Node(name="N1")
        n2 = Node(name="N2", locality=Locality.EXTERNAL)
        comm = CommunicationPath(source=n1, target=n2, name="API")
        model = DeploymentModel(
            name="m", nodes={n1, n2}, relationships={comm},
        )
        source = deployment_model_to_code(model)
        rebuilt = _exec_and_get_model(source)
        comms = [r for r in rebuilt.relationships if isinstance(r, CommunicationPath)]
        assert len(comms) == 1
        assert comms[0].name == "API"

    def test_nested_node_topological_order(self):
        outer = Node(name="Outer")
        inner = Node(name="Inner")
        outer.add_nested_node(inner)
        model = DeploymentModel(name="m", nodes={outer})
        source = deployment_model_to_code(model)
        # The outer Node constructor must appear before the
        # parent.add_nested_node(inner) call. Find offsets.
        outer_construct = source.find("node_outer = Node")
        inner_add = source.find(".add_nested_node(node_inner)")
        assert outer_construct >= 0
        assert inner_add >= 0
        assert outer_construct < inner_add
        rebuilt = _exec_and_get_model(source)
        # Containment survived.
        rebuilt_outer = next(n for n in rebuilt.nodes if n.name == "Outer")
        assert len(rebuilt_outer.nested_nodes) == 1

    def test_artifact_nested_in_node_round_trips(self):
        n = Node(name="Host")
        a = Artifact(name="Embedded")
        n.add_artifact(a)
        model = DeploymentModel(name="m", nodes={n})
        source = deployment_model_to_code(model)
        rebuilt = _exec_and_get_model(source)
        rebuilt_n = next(iter(rebuilt.nodes))
        assert len(rebuilt_n.nested_artifacts) == 1
        nested = next(iter(rebuilt_n.nested_artifacts))
        assert nested.name == "Embedded"
        assert nested.parent is rebuilt_n

    def test_node_kind_and_locality_round_trip(self):
        n = Node(name="Edge", kind=NodeKind.DEVICE, locality=Locality.EXTERNAL)
        model = DeploymentModel(name="m", nodes={n})
        source = deployment_model_to_code(model)
        assert "kind=NodeKind.DEVICE" in source
        assert "locality=Locality.EXTERNAL" in source
        rebuilt = _exec_and_get_model(source)
        rebuilt_n = next(iter(rebuilt.nodes))
        assert rebuilt_n.kind is NodeKind.DEVICE
        assert rebuilt_n.locality is Locality.EXTERNAL

    def test_manifests_dot_assignment(self):
        a = Artifact(name="X", manifests=["cmp-uuid-1", "cmp-uuid-2"])
        model = DeploymentModel(name="m", artifacts={a})
        source = deployment_model_to_code(model)
        assert ".manifests = ['cmp-uuid-1', 'cmp-uuid-2']" in source
        rebuilt = _exec_and_get_model(source)
        rebuilt_a = next(iter(rebuilt.artifacts))
        assert rebuilt_a.manifests == ["cmp-uuid-1", "cmp-uuid-2"]

    def test_interface_provided_round_trips(self):
        n = Node(name="VM")
        iface = Interface(name="Api")
        rel = InterfaceProvided(source=n, target=iface)
        model = DeploymentModel(
            name="m", nodes={n}, interfaces={iface}, relationships={rel},
        )
        source = deployment_model_to_code(model)
        rebuilt = _exec_and_get_model(source)
        assert len(rebuilt.interfaces) == 1
        rebuilt_rels = [r for r in rebuilt.relationships
                        if isinstance(r, InterfaceProvided)]
        assert len(rebuilt_rels) == 1

    def test_layout_passthrough(self):
        a = Artifact(name="X", layout={"id": "u1", "wme_type": "DeploymentComponent"})
        model = DeploymentModel(name="m", artifacts={a})
        source = deployment_model_to_code(model)
        assert ".layout = " in source
        rebuilt = _exec_and_get_model(source)
        rebuilt_a = next(iter(rebuilt.artifacts))
        assert rebuilt_a.layout["wme_type"] == "DeploymentComponent"

    def test_banner_and_imports_stable(self):
        source = deployment_model_to_code(DeploymentModel(name="X"))
        lines = source.splitlines()
        assert lines[0] == "####################"
        assert lines[1] == "# DEPLOYMENT MODEL #"
        assert lines[2] == "####################"

    def test_determinism_same_model_same_source(self):
        n = Node(name="A")
        a = Artifact(name="B")
        model = DeploymentModel(name="m", nodes={n}, artifacts={a})
        s1 = deployment_model_to_code(model)
        s2 = deployment_model_to_code(model)
        assert s1 == s2

    def test_rejects_non_deployment_model(self):
        with pytest.raises(TypeError):
            deployment_model_to_code("not a model")
