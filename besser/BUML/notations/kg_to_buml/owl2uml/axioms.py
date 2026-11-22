"""A typed OWL-2 view over a raw rdflib graph.

This isolates the one genuinely messy job — reconstructing OWL-2 axioms from
triples — so the mapping code reads ``ax.object_properties()`` rather than
poking at ``graph.triples(...)`` directly.
"""
from __future__ import annotations

from collections.abc import Iterator

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.collection import Collection as RDFList
from rdflib.namespace import OWL, RDF, RDFS

from besser.BUML.notations.kg_to_buml.datatype_mapping import xsd_to_primitive

from .naming import local_name

# rdf:type values that must NOT be treated as a class-assertion classifier (D39)
_META_TYPES = {OWL.Thing, OWL.NamedIndividual, RDFS.Resource,
               OWL.Class, RDFS.Class, OWL.Ontology,
               OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty}


def _sorted_pairs(pairs):
    """Deterministic ordering for a ``(subject, object)`` result set."""
    return sorted(pairs, key=lambda t: (str(t[0]), str(t[1])))


def primitive_for(datatype) -> str:
    """BUML primitive name for an XSD/RDF datatype IRI (D05, D06, D07).

    Deviation from the reference, which returns UML-ish names
    (``String``/``Integer``/``Real``/``URI``/``GYear``/…). BUML whitelists
    exactly nine primitives and cannot be extended, so we resolve through the
    canonical 40-entry table in
    :mod:`besser.BUML.notations.kg_to_buml.datatype_mapping` instead. Every
    type name that reaches a generated OCL body is therefore a real BUML type.

    Consequence: ``xsd:anyURI``, ``xsd:gYear`` and ``xsd:gYearMonth`` collapse
    onto ``str`` rather than becoming distinct pseudo-primitives.
    """
    primitive, _known = xsd_to_primitive(str(datatype))
    return primitive.name


class AxiomIndex:
    """A typed OWL-2 view over the raw triples.

    Every declaration set is kept twice: as a ``set`` for O(1) membership
    tests, and as a ``sorted_*`` tuple for iteration. rdflib returns results in
    hash order, so iterating the sets directly makes the whole transformation
    output-nondeterministic (verified against the reference: three runs, three
    different digests). Always iterate the sorted views.
    """

    def __init__(self, graph: Graph) -> None:
        self.g = graph
        self.object_props: set = set(graph.subjects(RDF.type, OWL.ObjectProperty))
        self.data_props: set = set(graph.subjects(RDF.type, OWL.DatatypeProperty))
        self.annotation_props: set = set(graph.subjects(RDF.type, OWL.AnnotationProperty))
        self.individuals: set = set(graph.subjects(RDF.type, OWL.NamedIndividual))
        self._named_classes: set = {
            c for c in graph.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)
        }
        self._datatypes: set = {
            d for d in graph.subjects(RDF.type, RDFS.Datatype) if isinstance(d, URIRef)
        }
        # Deterministic iteration views.
        self.sorted_object_props = tuple(sorted(self.object_props, key=str))
        self.sorted_data_props = tuple(sorted(self.data_props, key=str))
        self.sorted_individuals = tuple(sorted(self.individuals, key=str))
        self.sorted_named_classes = tuple(sorted(self._named_classes, key=str))
        self.sorted_datatypes = tuple(sorted(self._datatypes, key=str))

    # ---- declarations ----------------------------------------------------
    def ontology(self) -> URIRef | BNode | None:
        """Return the ontology header node (the first ``owl:Ontology`` subject by IRI), or ``None``."""
        return next(iter(sorted(self.g.subjects(RDF.type, OWL.Ontology), key=str)), None)

    def named_classes(self) -> set:
        """Return the set of classes declared with an IRI (anonymous class expressions excluded)."""
        return self._named_classes

    def datatypes(self) -> set:
        """Return the set of custom ``rdfs:Datatype`` IRIs declared in the ontology."""
        return self._datatypes

    # ---- property classification ----------------------------------------
    def is_object_property(self, p) -> bool:
        """Return whether ``p`` is declared an ``owl:ObjectProperty``."""
        return p in self.object_props

    def is_data_property(self, p) -> bool:
        """Return whether ``p`` is declared an ``owl:DatatypeProperty``."""
        return p in self.data_props

    def is_annotation_only(self, p) -> bool:
        """Return whether ``p`` is an ``owl:AnnotationProperty`` and neither an object nor a data property."""
        return (p in self.annotation_props
                and p not in self.object_props
                and p not in self.data_props)

    def domains(self, p) -> list:
        """Return the ``rdfs:domain`` values of ``p``, sorted by IRI."""
        return sorted(self.g.objects(p, RDFS.domain), key=str)

    def ranges(self, p) -> list:
        """Return the ``rdfs:range`` values of ``p``, sorted by IRI."""
        return sorted(self.g.objects(p, RDFS.range), key=str)

    def has_type(self, s, t) -> bool:
        """Return whether the graph asserts ``s rdf:type t``."""
        return (s, RDF.type, t) in self.g

    def is_functional(self, p) -> bool:
        """Return whether ``p`` is an ``owl:FunctionalProperty``."""
        return self.has_type(p, OWL.FunctionalProperty)

    def is_inverse_functional(self, p) -> bool:
        """Return whether ``p`` is an ``owl:InverseFunctionalProperty``."""
        return self.has_type(p, OWL.InverseFunctionalProperty)

    # ---- class-level axioms ---------------------------------------------
    def subclass_axioms(self) -> Iterator[tuple]:
        """Yield (sub, super) where sub is a URIRef (named class)."""
        for s, o in _sorted_pairs(self.g.subject_objects(RDFS.subClassOf)):
            if isinstance(s, URIRef):
                yield s, o

    def equivalent_classes(self) -> list:
        """Return the ``owl:equivalentClass`` ``(subject, object)`` pairs, sorted."""
        return _sorted_pairs(self.g.subject_objects(OWL.equivalentClass))

    def inverse_pairs(self) -> list:
        """Unordered ``owl:inverseOf`` pairs, deduplicated and sorted.

        The reference iterates the raw ``subject_objects`` result, so whichever
        direction of a symmetric ``inverseOf`` assertion surfaces first decides
        which property becomes the association's source role — the single
        source of run-to-run output drift we measured. Canonicalising each pair
        to sorted-IRI order removes it.
        """
        canonical = {
            (a, b) if str(a) <= str(b) else (b, a)
            for a, b in self.g.subject_objects(OWL.inverseOf)
        }
        return _sorted_pairs(canonical)

    def sub_property_axioms(self) -> Iterator[tuple]:
        """Yield ``(sub, super)`` for each ``rdfs:subPropertyOf`` between two IRI-named properties, sorted."""
        for s, o in _sorted_pairs(self.g.subject_objects(RDFS.subPropertyOf)):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                yield s, o

    def property_chains(self) -> list[tuple]:
        """``(property, [p1, ..., pn])`` for each ``owl:propertyChainAxiom``, sorted by property IRI."""
        out = []
        for p, head in _sorted_pairs(self.g.subject_objects(OWL.propertyChainAxiom)):
            chain = list(RDFList(self.g, head))
            if isinstance(p, URIRef) and chain and all(isinstance(c, URIRef) for c in chain):
                out.append((p, chain))
        return out

    def has_keys(self) -> list[tuple]:
        """``(class, [key properties])`` for each ``owl:hasKey`` on a named class, sorted by class IRI."""
        out = []
        for c, head in _sorted_pairs(self.g.subject_objects(OWL.hasKey)):
            keys = [k for k in RDFList(self.g, head) if isinstance(k, URIRef)]
            if isinstance(c, URIRef) and keys:
                out.append((c, keys))
        return out

    def all_disjoint_properties(self) -> list[list]:
        """Member lists of every ``owl:AllDisjointProperties`` axiom, in a stable order."""
        out = []
        for node in sorted(self.g.subjects(RDF.type, OWL.AllDisjointProperties), key=str):
            head = self.g.value(node, OWL.members)
            if head is not None:
                members = [m for m in RDFList(self.g, head) if isinstance(m, URIRef)]
                if members:
                    out.append(members)
        return out

    def has_assertions(self, p) -> bool:
        """Whether any individual carries an assertion on property ``p``.

        Separates a property that is *exclusively derived* (O19, O27 derivation)
        from one that may also be asserted directly (O20).
        """
        return any(s in self.individuals for s in self.g.subjects(p, None))

    def is_deprecated(self, s) -> bool:
        """Return whether ``s`` carries ``owl:deprecated true``."""
        return (s, OWL.deprecated, Literal(True)) in self.g

    # ---- individuals -----------------------------------------------------
    def classifier_of(self, ind) -> URIRef | None:
        """Most specific declared class asserted for an individual (D39)."""
        types = [t for t in self.g.objects(ind, RDF.type)
                 if isinstance(t, URIRef) and t not in _META_TYPES]
        if not types:
            return None
        # prefer a type that is a subclass of the others (more specific);
        # the IRI tiebreaker keeps equally-deep types in a stable order
        def depth(t):
            return len(list(self.g.transitive_objects(t, RDFS.subClassOf)))
        return sorted(types, key=lambda t: (-depth(t), str(t)))[0]

    def data_assertions(self, ind) -> Iterator[tuple]:
        """(property, literal) data-property assertions on an individual."""
        for p, o in _sorted_pairs(self.g.predicate_objects(ind)):
            if isinstance(o, Literal) and (p in self.data_props
                                           or local_name(p) == "name"):
                yield p, o

    def object_assertions(self, ind) -> Iterator[tuple]:
        """(property, individual) object-property assertions on an individual."""
        for p, o in _sorted_pairs(self.g.predicate_objects(ind)):
            if p in self.object_props and o in self.individuals:
                yield p, o
