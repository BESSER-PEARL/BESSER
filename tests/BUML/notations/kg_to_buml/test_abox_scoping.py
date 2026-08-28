"""Tests for individual-scoped KG → Object Diagram conversion.

Converting a whole ABox stops being useful past a few dozen individuals, so
the editor lets the user pick a starting individual. ``scope_abox`` prunes the
graph before conversion, and only the ABox: the TBox survives intact so the
class diagram that types the surviving objects is unchanged.
"""

from __future__ import annotations

import pytest

from besser.BUML.metamodel.kg import (
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KnowledgeGraph,
)
from besser.BUML.notations.kg_to_buml import (
    kg_to_class_diagram,
    kg_to_object_diagram,
    scope_abox,
)


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
EX = "http://ex.org/"


@pytest.fixture
def chain_kg() -> KnowledgeGraph:
    """A → B → C, plus X → A (reverse), plus an unconnected Iso.

    Every individual is a :Person and carries an :age literal, so the fixture
    also exercises literal attachment and rdf:type retention.
    """
    kg = KnowledgeGraph(name="chain")
    person = KGClass(id="Person", label="Person", iri=f"{EX}Person")
    kg.add_node(person)

    individuals = {}
    for name in ("A", "B", "C", "X", "Iso"):
        node = KGIndividual(id=name, label=name, iri=f"{EX}{name}")
        individuals[name] = node
        kg.add_node(node)
        kg.add_edge(KGEdge(id=f"type_{name}", source=node, target=person, iri=RDF_TYPE))

        literal = KGLiteral(id=f"age_{name}", value="30")
        kg.add_node(literal)
        kg.add_edge(KGEdge(id=f"has_age_{name}", source=node, target=literal, iri=f"{EX}age"))

    kg.add_edge(KGEdge(id="ab", source=individuals["A"], target=individuals["B"], iri=f"{EX}knows"))
    kg.add_edge(KGEdge(id="bc", source=individuals["B"], target=individuals["C"], iri=f"{EX}knows"))
    kg.add_edge(KGEdge(id="xa", source=individuals["X"], target=individuals["A"], iri=f"{EX}knows"))
    return kg


def _individual_ids(kg: KnowledgeGraph) -> set:
    return {n.id for n in kg.nodes if isinstance(n, KGIndividual)}


# ----------------------------------------------------------------------
# Traversal
# ----------------------------------------------------------------------


def test_full_component_is_reached_by_default(chain_kg):
    scoped = scope_abox(chain_kg, ["A"])
    # X is reached against the edge direction; Iso is unconnected.
    assert _individual_ids(scoped) == {"A", "B", "C", "X"}


def test_max_depth_bounds_the_traversal(chain_kg):
    assert _individual_ids(scope_abox(chain_kg, ["A"], max_depth=1)) == {"A", "B", "X"}
    assert _individual_ids(scope_abox(chain_kg, ["A"], max_depth=2)) == {"A", "B", "C", "X"}


def test_traversal_follows_edges_in_both_directions(chain_kg):
    # X → A, so rooting at A must still reach X at depth 1.
    assert "X" in _individual_ids(scope_abox(chain_kg, ["A"], max_depth=1))


def test_isolated_individual_yields_only_itself(chain_kg):
    assert _individual_ids(scope_abox(chain_kg, ["Iso"])) == {"Iso"}


def test_multiple_roots_are_unioned(chain_kg):
    scoped = scope_abox(chain_kg, ["Iso", "C"], max_depth=1)
    assert _individual_ids(scoped) == {"Iso", "C", "B"}


def test_source_graph_is_not_mutated(chain_kg):
    before_nodes = len(chain_kg.nodes)
    before_edges = len(chain_kg.edges)
    scope_abox(chain_kg, ["Iso"])
    assert len(chain_kg.nodes) == before_nodes
    assert len(chain_kg.edges) == before_edges


# ----------------------------------------------------------------------
# What survives besides the reachable individuals
# ----------------------------------------------------------------------


def test_tbox_survives_scoping(chain_kg):
    scoped = scope_abox(chain_kg, ["Iso"])
    assert any(isinstance(n, KGClass) and n.id == "Person" for n in scoped.nodes)


def test_kept_individuals_keep_their_literals_and_types(chain_kg):
    scoped = scope_abox(chain_kg, ["Iso"])
    node_ids = {n.id for n in scoped.nodes}
    assert "age_Iso" in node_ids
    edge_ids = {e.id for e in scoped.edges}
    assert "type_Iso" in edge_ids
    assert "has_age_Iso" in edge_ids


def test_excluded_individuals_do_not_drag_their_literals_along(chain_kg):
    scoped = scope_abox(chain_kg, ["Iso"])
    node_ids = {n.id for n in scoped.nodes}
    assert "age_A" not in node_ids
    assert "age_C" not in node_ids


def test_literals_do_not_extend_the_frontier(chain_kg):
    # Two individuals sharing a literal must not become neighbours through it.
    kg = KnowledgeGraph(name="shared")
    p = KGClass(id="Person", label="Person", iri=f"{EX}Person")
    a = KGIndividual(id="A", label="A", iri=f"{EX}A")
    b = KGIndividual(id="B", label="B", iri=f"{EX}B")
    shared = KGLiteral(id="lit", value="30")
    for node in (p, a, b, shared):
        kg.add_node(node)
    kg.add_edge(KGEdge(id="ta", source=a, target=p, iri=RDF_TYPE))
    kg.add_edge(KGEdge(id="tb", source=b, target=p, iri=RDF_TYPE))
    kg.add_edge(KGEdge(id="la", source=a, target=shared, iri=f"{EX}age"))
    kg.add_edge(KGEdge(id="lb", source=b, target=shared, iri=f"{EX}age"))

    assert _individual_ids(scope_abox(kg, ["A"])) == {"A"}


def test_rdf_type_edges_do_not_extend_the_frontier(chain_kg):
    # All five individuals share the :Person type; that must not connect them.
    assert _individual_ids(scope_abox(chain_kg, ["Iso"])) == {"Iso"}


# ----------------------------------------------------------------------
# Rejected input
# ----------------------------------------------------------------------


def test_unknown_root_id_raises(chain_kg):
    with pytest.raises(ValueError, match="Unknown node id"):
        scope_abox(chain_kg, ["nope"])


def test_non_individual_root_raises(chain_kg):
    with pytest.raises(ValueError, match="not an individual"):
        scope_abox(chain_kg, ["Person"])


# ----------------------------------------------------------------------
# Through the converter
# ----------------------------------------------------------------------


def test_scoped_conversion_only_emits_reachable_objects(chain_kg):
    result = kg_to_object_diagram(chain_kg, root_individual_ids=["A"], max_depth=1)
    names = {o.name for o in result.object_model.objects}
    assert names == {"A", "B", "X"}


def test_unscoped_conversion_is_unchanged(chain_kg):
    result = kg_to_object_diagram(chain_kg)
    names = {o.name for o in result.object_model.objects}
    assert names == {"A", "B", "C", "X", "Iso"}


def test_scoping_emits_an_abox_scoped_warning(chain_kg):
    result = kg_to_object_diagram(chain_kg, root_individual_ids=["Iso"])
    scoped = [w for w in result.warnings if w.code == "ABOX_SCOPED"]
    assert len(scoped) == 1
    assert "1 of 5 individual(s) included" in scoped[0].message


def test_class_diagram_is_identical_scoped_or_not(chain_kg):
    # The property that makes ABox-only pruning safe: the typing reference the
    # objects resolve against does not move when the diagram is scoped.
    full = kg_to_class_diagram(chain_kg)
    scoped_result = kg_to_object_diagram(chain_kg, root_individual_ids=["Iso"])

    full_classes = {c.name for c in full.domain_model.get_classes()}
    scoped_classes = {c.name for c in scoped_result.domain_model.get_classes()}
    assert full_classes == scoped_classes
