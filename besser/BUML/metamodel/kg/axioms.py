"""OWL axiom records attached to a :class:`KnowledgeGraph`.

Some OWL/OWL2 constructs don't naturally belong on a single node or edge:
``owl:equivalentClass`` relates *sets* of classes; ``owl:disjointUnionOf``
relates a class to a *list* of classes; ``owl:propertyChainAxiom`` relates a
property to an ordered list of properties; ``owl:hasKey`` relates a class to
a list of properties; etc.

This module reifies those constructs into small dataclasses kept on
``KnowledgeGraph.axioms``. Each axiom carries node/edge **ids** (strings),
not direct references — this keeps axioms JSON-friendly and decoupled from
the in-memory node identities, mirroring how the metamodel keeps edge
endpoints by id when serialised.

The records are *informational*: the deterministic KG → BUML class diagram
converter reads them in order to surface preflight issues and to lift
constructs into BUML where appropriate (e.g., ``owl:FunctionalProperty``
becomes ``max=1`` on the corresponding attribute). They never silently
change the BUML output without going through the preflight pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


__all__ = [
    "KGAxiom",
    "EquivalentClassesAxiom",
    "DisjointClassesAxiom",
    "DisjointUnionAxiom",
    "SubPropertyOfAxiom",
    "InversePropertiesAxiom",
    "PropertyChainAxiom",
    "HasKeyAxiom",
    "ImportAxiom",
]


@dataclass
class KGAxiom:
    """Base class for all axiom records.

    Subclasses carry the typed payload; this base exists so a single
    ``KnowledgeGraph.axioms: List[KGAxiom]`` field can hold any of them.
    """


@dataclass
class EquivalentClassesAxiom(KGAxiom):
    """``owl:equivalentClass`` between two or more classes.

    All listed class ids are semantically equivalent (sameInstances).
    """

    class_ids: List[str] = field(default_factory=list)


@dataclass
class DisjointClassesAxiom(KGAxiom):
    """``owl:disjointWith`` / ``owl:AllDisjointClasses`` axiom.

    The listed classes are pairwise disjoint (no shared instances).
    """

    class_ids: List[str] = field(default_factory=list)


@dataclass
class DisjointUnionAxiom(KGAxiom):
    """``owl:disjointUnionOf``: the union class is the disjoint union of its parts."""

    union_class_id: str = ""
    part_class_ids: List[str] = field(default_factory=list)


@dataclass
class SubPropertyOfAxiom(KGAxiom):
    """``rdfs:subPropertyOf`` between two properties (sub ⊑ super)."""

    sub_property_id: str = ""
    super_property_id: str = ""


@dataclass
class InversePropertiesAxiom(KGAxiom):
    """``owl:inverseOf`` between two object properties."""

    property_a_id: str = ""
    property_b_id: str = ""


@dataclass
class PropertyChainAxiom(KGAxiom):
    """``owl:propertyChainAxiom``: a property is the composition of an ordered chain.

    Example: ``hasUncle`` ← (``hasParent``, ``hasBrother``).
    """

    property_id: str = ""
    chain_property_ids: List[str] = field(default_factory=list)


@dataclass
class HasKeyAxiom(KGAxiom):
    """``owl:hasKey``: the listed properties uniquely identify instances of the class."""

    class_id: str = ""
    property_ids: List[str] = field(default_factory=list)


@dataclass
class ImportAxiom(KGAxiom):
    """``owl:imports``: the source ontology imports another by IRI.

    The importer does **not** follow these references; this record exists so
    the UI can surface the dependency to the user.
    """

    target_iri: str = ""
    source_ontology_iri: Optional[str] = None
