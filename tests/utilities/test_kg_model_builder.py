"""Tests for the KnowledgeGraph → Python code builder.

The builder's output is never imported — ``kg_buml_to_json`` strips the import
block and ``exec()``s the rest against a restricted namespace. These tests
therefore always assert through that reader rather than against the emitted
text, so a change that produces plausible-looking code the reader cannot
execute still fails.
"""

from __future__ import annotations

import pytest

from besser.BUML.metamodel.kg import (
    DisjointClassesAxiom,
    EquivalentClassesAxiom,
    HasKeyAxiom,
    ImportAxiom,
    InversePropertiesAxiom,
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
    PropertyChainAxiom,
    SubPropertyOfAxiom,
)
from besser.utilities.buml_code_builder.kg_model_builder import kg_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters import (
    kg_buml_to_json,
)


def _round_trip(kg: KnowledgeGraph, tmp_path) -> dict:
    """Generate code for ``kg`` and read it back through the project reader."""
    path = tmp_path / "kg_model.py"
    kg_model_to_code(kg, str(path))
    return kg_buml_to_json(path.read_text(encoding="utf-8"))


@pytest.fixture
def sample_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph(name="sample_kg")
    person = KGClass(id="c1", label="Person", iri="http://ex.org/Person")
    agent = KGClass(id="c2", label="Agent", iri="http://ex.org/Agent")
    knows = KGProperty(id="p1", label="knows", iri="http://ex.org/knows")
    alice = KGIndividual(id="i1", label="Alice", iri="http://ex.org/alice")
    age = KGLiteral(id="l1", value="30", datatype="http://www.w3.org/2001/XMLSchema#integer")
    blank = KGBlank(id="b1", label="")
    for node in (person, agent, knows, alice, age, blank):
        kg.add_node(node)
    kg.add_edge(KGEdge(
        id="e1", source=person, target=agent, label="subClassOf",
        iri="http://www.w3.org/2000/01/rdf-schema#subClassOf",
    ))
    kg.add_edge(KGEdge(id="e2", source=alice, target=age, label="age", iri="http://ex.org/age"))
    kg.add_axiom(EquivalentClassesAxiom(class_ids=["c1", "c2"]))
    return kg


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------


def test_round_trip_preserves_nodes_edges_and_axioms(sample_kg, tmp_path):
    model = _round_trip(sample_kg, tmp_path)

    assert model["type"] == "KnowledgeGraphDiagram"
    assert len(model["nodes"]) == 6
    assert len(model["edges"]) == 2
    assert len(model["axioms"]) == 1

    by_id = {n["id"]: n for n in model["nodes"]}
    assert by_id["c1"]["nodeType"] == "class"
    assert by_id["p1"]["nodeType"] == "property"
    assert by_id["i1"]["nodeType"] == "individual"
    assert by_id["l1"]["nodeType"] == "literal"
    assert by_id["b1"]["nodeType"] == "blank"
    assert by_id["c1"]["iri"] == "http://ex.org/Person"


def test_round_trip_preserves_literal_value_and_datatype(sample_kg, tmp_path):
    model = _round_trip(sample_kg, tmp_path)
    literal = {n["id"]: n for n in model["nodes"]}["l1"]
    assert literal["value"] == "30"
    assert literal["datatype"] == "http://www.w3.org/2001/XMLSchema#integer"


def test_round_trip_preserves_edge_endpoints(sample_kg, tmp_path):
    model = _round_trip(sample_kg, tmp_path)
    edges = {e["id"]: e for e in model["edges"]}
    assert edges["e1"]["source"] == "c1"
    assert edges["e1"]["target"] == "c2"
    assert edges["e1"]["iri"].endswith("subClassOf")


def test_emits_the_section_banner_the_project_reader_looks_for(sample_kg, tmp_path):
    # ``project_to_code`` only prepends a numbered header when a project holds
    # more than one KG, so a lone KG relies entirely on this banner to be found.
    path = tmp_path / "kg_model.py"
    kg_model_to_code(sample_kg, str(path))
    assert "# KNOWLEDGE_GRAPH MODEL" in path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Constraint payloads
# ----------------------------------------------------------------------


def test_constraint_specs_survive_the_round_trip(tmp_path):
    # Losing metadata silently disarms several preflight detectors, so this is
    # the payload that matters most.
    kg = KnowledgeGraph(name="constrained")
    node_c = KGNodeConstraint(id="nc1", label="Shape", metadata={
        "constraintSpecs": [
            {"kind": "disjointWith", "vocab": ["owl"],
             "params": {"classes": ["http://ex.org/Agent"]}},
        ],
    })
    prop_c = KGPropertyConstraint(id="pc1", label="MinOne", metadata={
        "constraintSpecs": [
            {"kind": "minCardinality", "vocab": ["owl", "shacl"], "params": {"value": 1}},
        ],
    })
    kg.add_node(node_c)
    kg.add_node(prop_c)

    model = _round_trip(kg, tmp_path)
    by_id = {n["id"]: n for n in model["nodes"]}
    assert by_id["nc1"]["nodeType"] == "nodeConstraint"
    assert by_id["pc1"]["nodeType"] == "propertyConstraint"
    assert by_id["nc1"]["metadata"]["constraintSpecs"][0]["kind"] == "disjointWith"
    assert by_id["pc1"]["metadata"]["constraintSpecs"][0]["params"]["value"] == 1


def test_all_axiom_kinds_round_trip(tmp_path):
    kg = KnowledgeGraph(name="axioms")
    kg.add_node(KGClass(id="c1", label="A"))
    for axiom in (
        EquivalentClassesAxiom(class_ids=["c1"]),
        DisjointClassesAxiom(class_ids=["c1"]),
        SubPropertyOfAxiom(sub_property_id="p1", super_property_id="p2"),
        InversePropertiesAxiom(property_a_id="p1", property_b_id="p2"),
        PropertyChainAxiom(property_id="p1", chain_property_ids=["p2", "p3"]),
        HasKeyAxiom(class_id="c1", property_ids=["p1"]),
        ImportAxiom(target_iri="http://ex.org/other", source_ontology_iri=None),
    ):
        kg.add_axiom(axiom)

    model = _round_trip(kg, tmp_path)
    kinds = sorted(a["kind"] for a in model["axioms"])
    assert kinds == [
        "DisjointClassesAxiom",
        "EquivalentClassesAxiom",
        "HasKeyAxiom",
        "ImportAxiom",
        "InversePropertiesAxiom",
        "PropertyChainAxiom",
        "SubPropertyOfAxiom",
    ]
    chain = next(a for a in model["axioms"] if a["kind"] == "PropertyChainAxiom")
    assert chain["chain_property_ids"] == ["p2", "p3"]


# ----------------------------------------------------------------------
# Hostile input and edge cases
# ----------------------------------------------------------------------


@pytest.mark.parametrize("label", [
    'quote " inside',
    "apostrophe ' inside",
    "back\\slash",
    "new\nline",
    "carriage\rreturn",
    "unicode ünïcødé ☃",
    "'''triple''' and \"\"\"quotes\"\"\"",
    "trailing backslash \\",
])
def test_hostile_labels_round_trip_verbatim(label, tmp_path):
    # The generated source is exec()'d, so a mis-escaped label is a syntax
    # error or worse rather than a cosmetic defect.
    kg = KnowledgeGraph(name="hostile")
    kg.add_node(KGClass(id="c1", label=label, iri="http://ex.org/C"))
    model = _round_trip(kg, tmp_path)
    assert model["nodes"][0]["label"] == label


def test_labels_that_are_not_identifiers_do_not_collide(tmp_path):
    # Two labels sanitising to the same variable name must still produce two
    # distinct nodes.
    kg = KnowledgeGraph(name="collisions")
    kg.add_node(KGClass(id="c1", label="my class", iri="http://ex.org/1"))
    kg.add_node(KGClass(id="c2", label="my-class", iri="http://ex.org/2"))
    kg.add_node(KGClass(id="c3", label="my/class", iri="http://ex.org/3"))
    kg.add_node(KGClass(id="c4", label="", iri="http://ex.org/4"))
    kg.add_node(KGClass(id="c5", label="123", iri="http://ex.org/5"))
    kg.add_node(KGClass(id="c6", label="class", iri="http://ex.org/6"))

    model = _round_trip(kg, tmp_path)
    assert len(model["nodes"]) == 6
    assert {n["id"] for n in model["nodes"]} == {"c1", "c2", "c3", "c4", "c5", "c6"}


def test_empty_graph_round_trips(tmp_path):
    # ``{}`` is an empty dict, not an empty set — the builder has to emit
    # ``set()`` for the nodes and edges arguments.
    kg = KnowledgeGraph(name="empty")
    model = _round_trip(kg, tmp_path)
    assert model["nodes"] == []
    assert model["edges"] == []
    assert model["axioms"] == []


def test_output_is_deterministic(sample_kg, tmp_path):
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    kg_model_to_code(sample_kg, str(first))
    kg_model_to_code(sample_kg, str(second))
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_model_var_name_is_configurable(sample_kg, tmp_path):
    # project_to_code passes kg_model_1, kg_model_2, … for multi-KG projects.
    path = tmp_path / "kg_model.py"
    kg_model_to_code(sample_kg, str(path), model_var_name="kg_model_2")
    text = path.read_text(encoding="utf-8")
    assert "kg_model_2 = KnowledgeGraph(" in text
    assert kg_buml_to_json(text)["nodes"]
