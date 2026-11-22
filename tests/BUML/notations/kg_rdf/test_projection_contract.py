"""The contract between the TTL export and the projection BESSER reasons over.

BESSER writes a ``KnowledgeGraph`` as RDF in two places:

=================  =========================================  ==========================
                   ``knowledge_graph_to_rdf`` (exporter)      ``kg_to_rdf`` (projection)
=================  =========================================  ==========================
Purpose            the user-facing Turtle / RDF-XML export    input to the KG -> UML
                                                              conversion and, with
                                                              ``mirror_shapes=True``, to
                                                              the consistency check
Constraint nodes   rebuilt from their structured specs in     re-materialised onto the
                   the vocabulary the caller chooses          node's own term
Vocabularies       ``owl`` / ``shacl`` / ``both``             the source's own;
                                                              ``mirror_shapes`` adds the
                                                              other (as ``both`` does)
Individuals        as the KG has them                         also declared
                                                              ``owl:NamedIndividual``
=================  =========================================  ==========================

Both take constraint *parameters* from the specs (the importer keeps no raw
edge for them), and they share their leaf emitters (``kg_rdf._emit.emit_*``), the
constraint indexes and the list rebuilding, so they cannot drift on how any
single construct is written. This module pins what is left:

* OWL only: isomorphic apart from the projection's ``owl:NamedIndividual``
  declarations.
* SHACL included (``vocab="both"`` against ``mirror_shapes=True``): isomorphic
  on OWL-sourced fixtures once each restriction the projection also wrote as a
  shape is split into the two nodes the exporter uses; on a shapes graph, the
  same constraint parameters, except the SHACL qualified counts only the
  projection has (the importer does not fold them into specs).
* The consistency check reads the mirrored projection, so a spec edit reaches
  the class diagram and the consistency report alike.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import OWL, RDF, BNode, Graph, Literal, Namespace, URIRef
from rdflib.compare import isomorphic

from besser.BUML.notations.kg_to_buml import consistency
from besser.BUML.notations.kg_rdf import kg_to_rdf, knowledge_graph_to_rdf, owl_file_to_knowledge_graph
from besser.BUML.notations.kg_rdf._emit import OWL_RESTRICTION_KINDS, SHACL_PROPERTY_KINDS

SH = Namespace("http://www.w3.org/ns/shacl#")

_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"      # tests/fixtures
_KG = Path(__file__).resolve().parents[1] / "kg_to_buml" / "fixtures"


def _without_shacl(graph: Graph) -> Graph:
    out = Graph()
    for s, p, o in graph:
        if str(p).startswith(str(SH)) or str(o).startswith(str(SH)):
            continue
        out.add((s, p, o))
    return out


def _without_individual_declarations(graph: Graph) -> Graph:
    """Drop the converter's ``x a owl:NamedIndividual`` declarations."""
    out = Graph()
    for s, p, o in graph:
        if p == RDF.type and o == OWL.NamedIndividual:
            continue
        out.add((s, p, o))
    return out


def _views(path: Path):
    kg = owl_file_to_knowledge_graph(str(path))
    exported = _without_individual_declarations(_without_shacl(knowledge_graph_to_rdf(kg, vocab="owl")))
    converted = _without_individual_declarations(_without_shacl(kg_to_rdf(kg)[0]))
    return exported, converted


def test_the_projections_agree_on_a_graph_with_restrictions_and_individuals():
    exported, converted = _views(_FIXTURES / "complex_kg.ttl")

    assert len(exported) > 0
    assert isomorphic(exported, converted)


def test_individual_declarations_are_the_only_thing_the_contract_sets_aside():
    """Without stripping them the graphs differ; with only them stripped they match.

    So the exclusion in ``_views`` is exactly the converter's declarations and
    hides nothing else. (Compared by isomorphism: the exporter mints fresh blank
    nodes, so a raw triple set-difference would be meaningless.)
    """
    kg = owl_file_to_knowledge_graph(str(_FIXTURES / "complex_kg.ttl"))
    exported = _without_shacl(knowledge_graph_to_rdf(kg, vocab="owl"))
    converted = _without_shacl(kg_to_rdf(kg)[0])

    def declarations(graph):
        return [1 for _, p, o in graph if p == RDF.type and o == OWL.NamedIndividual]

    assert not isomorphic(exported, converted)
    assert isomorphic(
        _without_individual_declarations(exported), _without_individual_declarations(converted)
    )
    # The KG may already declare some individuals (the exporter keeps those); the
    # converter's pass declares the rest.
    assert len(declarations(converted)) > len(declarations(exported))


def test_the_projections_agree_on_a_real_ontology():
    exported, converted = _views(_KG / "bibo" / "bibo.ttl")

    assert isomorphic(exported, converted)


_ONE_OF_TTL = """
@prefix ex: <http://ex.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:Color a owl:Class ; owl:equivalentClass [ a owl:Class ; owl:oneOf ( ex:red ex:green ) ] .
ex:size a owl:DatatypeProperty ;
    rdfs:range [ a rdfs:Datatype ; owl:oneOf ( "S" "M" "3"^^xsd:integer ) ] .
"""


def _one_of_lists(graph: Graph):
    return sorted(
        sorted((type(m).__name__, str(m), str(getattr(m, "datatype", None) or "")) for m in graph.items(head))
        for head in graph.objects(None, OWL.oneOf)
    )


def test_the_projections_agree_on_a_literal_one_of(tmp_path: Path):
    """A datatype ``owl:oneOf`` keeps literal members (with their datatype) in both.

    The exporter rebuilt a blank node's list from the recorded member *strings*
    as IRIs, so ``("S" "M")`` came out as ``(<S> <M>)``: a different statement
    from the one the projection, reading the member edges, made.
    """
    path = tmp_path / "one_of.ttl"
    path.write_text(_ONE_OF_TTL, encoding="utf-8")
    exported, converted = _views(path)

    assert _one_of_lists(exported) == _one_of_lists(converted) == [
        [("Literal", "3", str(XSD_INTEGER)), ("Literal", "M", ""), ("Literal", "S", "")],
        [("URIRef", "http://ex.org/green", ""), ("URIRef", "http://ex.org/red", "")],
    ]
    assert isomorphic(exported, converted)


XSD_INTEGER = URIRef("http://www.w3.org/2001/XMLSchema#integer")


def _dangling_fillers(graph: Graph):
    """``allValuesFrom`` fillers that are IRIs but not absolute, with no triples."""
    return [
        filler
        for _, _, filler in graph.triples((None, OWL.allValuesFrom, None))
        if isinstance(filler, URIRef)
        and "://" not in str(filler)
        and not str(filler).startswith("urn:")
    ]


def test_exporter_keeps_a_blank_node_restriction_filler_connected():
    """``allValuesFrom`` whose filler is a blank-node class stays a blank node.

    A restriction can name an anonymous class (here a union) as its filler. The
    exporter rebuilt the filler with ``URIRef(str(value))``, so the blank node's
    label became a relative IRI that nothing else in the graph mentions: a
    dangling reference, and a different one on every import, since the label is
    minted per import.
    """
    kg = owl_file_to_knowledge_graph(str(_KG / "bibo" / "bibo.ttl"))

    assert _dangling_fillers(knowledge_graph_to_rdf(kg, vocab="owl")) == []
    assert _dangling_fillers(kg_to_rdf(kg)[0]) == []          # the converter never had the bug


def test_a_blank_node_filler_is_the_same_node_the_kg_declares():
    """Not just "no dangling IRI": the filler must be the blank node that has the
    union's own triples, so the restriction is actually connected to it."""
    kg = owl_file_to_knowledge_graph(str(_KG / "bibo" / "bibo.ttl"))
    graph = knowledge_graph_to_rdf(kg, vocab="owl")

    blank_fillers = [
        f for _, _, f in graph.triples((None, OWL.allValuesFrom, None)) if not isinstance(f, URIRef)
    ]
    assert blank_fillers, "bibo has a restriction over an anonymous class"
    for filler in blank_fillers:
        assert any(True for _ in graph.triples((filler, None, None))), f"{filler} has no description"


# ---------------------------------------------------------------------------
# With SHACL included
# ---------------------------------------------------------------------------


def _split_restriction_shapes(graph: Graph) -> Graph:
    """Move the SHACL half of each node that is both an ``owl:Restriction`` and a
    ``sh:PropertyShape`` onto a node of its own, as the exporter writes it.

    The projection mirrors an OWL restriction onto the constraint's own term, so
    ``sh:sourceShape`` in a validation report still names the KG node; the
    exporter puts the shape on that term and the restriction on a fresh blank
    node. The two are the same statements either way.
    """
    merged = {
        s for s in graph.subjects(RDF.type, OWL.Restriction) if (s, RDF.type, SH.PropertyShape) in graph
    }
    twin = {node: BNode() for node in merged}
    out = Graph()
    for s, p, o in graph:
        if s in twin and (str(p).startswith(str(SH)) or (p == RDF.type and o == SH.PropertyShape)):
            s = twin[s]
        if o in twin and p == SH.property:
            o = twin[o]
        out.add((s, p, o))
    return out


@pytest.mark.parametrize(
    "path", [_FIXTURES / "complex_kg.ttl", _KG / "bibo" / "bibo.ttl"], ids=lambda p: p.name,
)
def test_with_shacl_included_the_projections_agree(path: Path):
    kg = owl_file_to_knowledge_graph(str(path))
    exported = _without_individual_declarations(knowledge_graph_to_rdf(kg, vocab="both"))
    projected = _without_individual_declarations(kg_to_rdf(kg, mirror_shapes=True)[0])

    assert any(projected.triples((None, RDF.type, SH.PropertyShape))), "nothing was mirrored"
    assert isomorphic(exported, _split_restriction_shapes(projected))


_QUALIFIED_COUNTS = {SH.qualifiedMinCount, SH.qualifiedMaxCount}


def _parameters(graph: Graph):
    """Every constraint parameter, as ``(vocabulary, property, predicate, value)``."""
    out = set()
    for shape, path in graph.subject_objects(SH.path):
        for predicate, value in graph.predicate_objects(shape):
            if predicate in set(SHACL_PROPERTY_KINDS.values()) and predicate not in (SH["in"], SH.languageIn):
                out.add(("sh", str(path), str(predicate), value))
    for restriction, prop in graph.subject_objects(OWL.onProperty):
        for predicate, value in graph.predicate_objects(restriction):
            if predicate in set(OWL_RESTRICTION_KINDS.values()):
                out.add(("owl", str(prop), str(predicate), value))
    return out


def test_on_a_shapes_graph_the_projections_set_the_same_parameters():
    """A shapes file is all punning and SHACL, where the structure legitimately
    differs (the export drops edges around constraint nodes). The parameters must
    not: both read them from the specs."""
    kg = owl_file_to_knowledge_graph(str(_KG / "bibo" / "bibo-shapes.ttl"))
    exported = _parameters(knowledge_graph_to_rdf(kg, vocab="both"))
    projected = _parameters(kg_to_rdf(kg, mirror_shapes=True)[0])

    assert exported, "the export lost every property shape's sh:path"
    only_projected = {p for p in projected - exported if URIRef(p[2]) not in _QUALIFIED_COUNTS}
    assert exported - projected == set()
    assert only_projected == set()


# ---------------------------------------------------------------------------
# One projection for both consumers
# ---------------------------------------------------------------------------


_EDITABLE_SHAPE_TTL = """
@prefix ex: <http://ex.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:name a owl:DatatypeProperty ; rdfs:domain ex:Person ; rdfs:range xsd:string .
ex:Person a owl:Class, sh:NodeShape ;
   sh:property [ sh:path ex:name ; sh:minCount 1 ] .
ex:bob a ex:Person ; ex:name "Bob" .
"""


def _edited(tmp_path: Path, min_count: int):
    """Import, then change ``minCount`` as the inspector does: only
    ``metadata.constraintSpecs`` changes, the imported edge stays."""
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        kg_to_json,
        process_kg_diagram,
    )

    path = tmp_path / "shape.ttl"
    path.write_text(_EDITABLE_SHAPE_TTL, encoding="utf-8")
    payload = kg_to_json(owl_file_to_knowledge_graph(str(path)))
    for node in payload["model"]["nodes"]:
        for spec in (node.get("metadata") or {}).get("constraintSpecs", []):
            if spec["kind"] == "minCardinality":
                spec["value"] = min_count
    return process_kg_diagram(payload)


def test_a_spec_edit_reaches_the_class_diagram_and_the_consistency_check(tmp_path: Path):
    """Editing ``minCount`` 1 -> 3 used to change the consistency report (the
    exporter reads specs) but not the class diagram (the conversion trusted the
    imported edge)."""
    from besser.BUML.notations.kg_to_buml import kg_to_class_diagram

    kg = _edited(tmp_path, 3)

    bodies = [c.expression for c in kg_to_class_diagram(kg).domain_model.constraints]
    assert any("self.name->size() >= 3" in b for b in bodies), bodies
    assert not any(">= 1" in b for b in bodies), bodies

    messages = [issue.message for issue in consistency.check_kg_consistency(kg).issues]
    assert any("Less than 3 values" in m for m in messages), messages


def test_both_consumers_see_the_same_parameter(tmp_path: Path):
    kg = _edited(tmp_path, 3)

    for mirror in (False, True):
        graph, _ = kg_to_rdf(kg, mirror_shapes=mirror)
        assert sorted(o.toPython() for o in graph.objects(None, SH.minCount)) == [3], mirror
    assert [o for o in knowledge_graph_to_rdf(kg).objects(None, SH.minCount)] == [Literal(3, datatype=_XSD_NNI)]


_XSD_NNI = URIRef("http://www.w3.org/2001/XMLSchema#nonNegativeInteger")


def test_a_key_edit_reaches_the_export_and_the_projection(tmp_path: Path):
    """``owl:hasKey`` is stored once, as the spec the inspector edits. It used to
    be stored three times, and the projection let the import-time record win
    over an edited spec while the exporter followed the spec."""
    path = tmp_path / "key.ttl"
    path.write_text(
        "@prefix : <http://ex.org/> .\n@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        ":ssn a owl:DatatypeProperty . :passport a owl:DatatypeProperty .\n"
        ":P a owl:Class ; owl:hasKey ( :ssn ) .\n",
        encoding="utf-8",
    )
    kg = owl_file_to_knowledge_graph(str(path))
    (key_node,) = [n for n in kg.nodes if any(s["kind"] == "hasKey" for s in n.metadata.get("constraintSpecs", []))]
    key_node.metadata = {**key_node.metadata, "constraintSpecs": [{"kind": "hasKey", "value": ["http://ex.org/passport"]}]}

    for graph in (knowledge_graph_to_rdf(kg, vocab="owl"), kg_to_rdf(kg)[0]):
        keys = [[str(m) for m in graph.items(head)] for head in graph.objects(None, OWL.hasKey)]
        assert keys == [["http://ex.org/passport"]]
