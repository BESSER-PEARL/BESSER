"""OWL axiom records attached to a :class:`KnowledgeGraph`.

Some OWL/OWL2 constructs don't naturally belong on a single node or edge:
``owl:equivalentClass`` relates *sets* of classes; ``owl:disjointUnionOf``
relates a class to a *list* of classes; ``owl:propertyChainAxiom`` relates a
property to an ordered list of properties; ``owl:hasKey`` relates a class to
a list of properties; etc.

This module reifies those constructs into small classes kept on
``KnowledgeGraph.axioms``. Each axiom carries node/edge **ids** (strings),
not direct references — this keeps axioms JSON-friendly and decoupled from
the in-memory node identities, mirroring how the metamodel keeps edge
endpoints by id when serialised.

Like the rest of the metamodel, the fields are private attributes behind
validating setters (see :class:`~besser.BUML.metamodel.kg.KGEdge`): an id must
be a ``str`` and a list of ids a list of ``str``, and a bad value fails at the
assignment instead of surfacing later in a converter. Each class lists its
fields, in constructor order, in ``FIELDS``; :meth:`KGAxiom.to_dict` and the
serialisers are driven by it.

The records are *informational*: the deterministic KG → BUML class diagram
converter reads them in order to surface preflight issues and to lift
constructs into BUML where appropriate (e.g., ``owl:FunctionalProperty``
becomes ``max=1`` on the corresponding attribute). They never silently
change the BUML output without going through the preflight pipeline.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Tuple


__all__ = [
    "KGAxiom",
    "EquivalentClassesAxiom",
    "EquivalentPropertiesAxiom",
    "DisjointClassesAxiom",
    "DisjointUnionAxiom",
    "SubPropertyOfAxiom",
    "InversePropertiesAxiom",
    "PropertyChainAxiom",
    "HasKeyAxiom",
    "ImportAxiom",
]


def _id_property(name: str, doc: str, *, optional: bool = False) -> property:
    """A ``str`` field stored as ``_<name>``; ``optional`` also accepts ``None``."""
    attr = f"_{name}"

    def getter(self) -> Optional[str]:
        return getattr(self, attr)

    def setter(self, value: Optional[str]) -> None:
        if value is None and optional:
            setattr(self, attr, None)
            return
        if not isinstance(value, str):
            raise ValueError(
                f"{type(self).__name__}.{name} must be a str, got {type(value).__name__}."
            )
        setattr(self, attr, value)

    return property(getter, setter, doc=doc)


def _id_list_property(name: str, doc: str) -> property:
    """A list-of-``str`` field stored as ``_<name>``.

    ``None`` means empty, and the setter stores a copy so a caller's own list
    cannot alias the axiom's state. The getter returns a copy too: change the
    field by assigning it, not by mutating what it returns.
    """
    attr = f"_{name}"

    def getter(self) -> List[str]:
        return list(getattr(self, attr))

    def setter(self, value: Optional[List[str]]) -> None:
        if value is None:
            setattr(self, attr, [])
            return
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{type(self).__name__}.{name} must be a list of str.")
        setattr(self, attr, list(value))

    return property(getter, setter, doc=doc)


class KGAxiom:
    """Base class for all axiom records.

    Subclasses carry the typed payload; this base exists so a single
    ``KnowledgeGraph.axioms: List[KGAxiom]`` field can hold any of them, and
    gives them value semantics (``==``, ``repr``, ``to_dict``).
    """

    #: Field names in constructor order. Declared by each subclass.
    FIELDS: ClassVar[Tuple[str, ...]] = ()

    def to_dict(self) -> Dict[str, Any]:
        """The fields as a plain dict (lists copied), without the axiom's kind."""
        return {name: getattr(self, name) for name in self.FIELDS}

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    # Mutable and compared by value, so deliberately unhashable (as a
    # ``@dataclass`` with ``eq=True`` was).
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        args = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.FIELDS)
        return f"{type(self).__name__}({args})"


class EquivalentClassesAxiom(KGAxiom):
    """``owl:equivalentClass`` between two or more classes.

    All listed class ids are semantically equivalent (sameInstances).
    """

    FIELDS = ("class_ids",)
    class_ids = _id_list_property("class_ids", "list[str]: Ids of the equivalent classes.")

    def __init__(self, class_ids: Optional[List[str]] = None) -> None:
        self.class_ids = class_ids


class EquivalentPropertiesAxiom(KGAxiom):
    """``owl:equivalentProperty`` between two or more properties.

    All listed property ids are semantically equivalent and canonicalize onto a
    single B-UML attribute/association during KG → class-diagram conversion.
    """

    FIELDS = ("property_ids",)
    property_ids = _id_list_property("property_ids", "list[str]: Ids of the equivalent properties.")

    def __init__(self, property_ids: Optional[List[str]] = None) -> None:
        self.property_ids = property_ids


class DisjointClassesAxiom(KGAxiom):
    """``owl:disjointWith`` / ``owl:AllDisjointClasses`` axiom.

    The listed classes are pairwise disjoint (no shared instances).
    """

    FIELDS = ("class_ids",)
    class_ids = _id_list_property("class_ids", "list[str]: Ids of the pairwise-disjoint classes.")

    def __init__(self, class_ids: Optional[List[str]] = None) -> None:
        self.class_ids = class_ids


class DisjointUnionAxiom(KGAxiom):
    """``owl:disjointUnionOf``: the union class is the disjoint union of its parts."""

    FIELDS = ("union_class_id", "part_class_ids")
    union_class_id = _id_property("union_class_id", "str: Id of the union class.")
    part_class_ids = _id_list_property("part_class_ids", "list[str]: Ids of the disjoint parts.")

    def __init__(self, union_class_id: str = "", part_class_ids: Optional[List[str]] = None) -> None:
        self.union_class_id = union_class_id
        self.part_class_ids = part_class_ids


class SubPropertyOfAxiom(KGAxiom):
    """``rdfs:subPropertyOf`` between two properties (sub ⊑ super)."""

    FIELDS = ("sub_property_id", "super_property_id")
    sub_property_id = _id_property("sub_property_id", "str: Id of the sub-property.")
    super_property_id = _id_property("super_property_id", "str: Id of the super-property.")

    def __init__(self, sub_property_id: str = "", super_property_id: str = "") -> None:
        self.sub_property_id = sub_property_id
        self.super_property_id = super_property_id


class InversePropertiesAxiom(KGAxiom):
    """``owl:inverseOf`` between two object properties."""

    FIELDS = ("property_a_id", "property_b_id")
    property_a_id = _id_property("property_a_id", "str: Id of the first property.")
    property_b_id = _id_property("property_b_id", "str: Id of its inverse.")

    def __init__(self, property_a_id: str = "", property_b_id: str = "") -> None:
        self.property_a_id = property_a_id
        self.property_b_id = property_b_id


class PropertyChainAxiom(KGAxiom):
    """``owl:propertyChainAxiom``: a property is the composition of an ordered chain.

    Example: ``hasUncle`` ← (``hasParent``, ``hasBrother``).
    """

    FIELDS = ("property_id", "chain_property_ids")
    property_id = _id_property("property_id", "str: Id of the composed property.")
    chain_property_ids = _id_list_property(
        "chain_property_ids", "list[str]: Ids of the chained properties, in order."
    )

    def __init__(self, property_id: str = "", chain_property_ids: Optional[List[str]] = None) -> None:
        self.property_id = property_id
        self.chain_property_ids = chain_property_ids


class HasKeyAxiom(KGAxiom):
    """``owl:hasKey``: the listed properties uniquely identify instances of the class."""

    FIELDS = ("class_id", "property_ids")
    class_id = _id_property("class_id", "str: Id of the keyed class.")
    property_ids = _id_list_property("property_ids", "list[str]: Ids of the key properties.")

    def __init__(self, class_id: str = "", property_ids: Optional[List[str]] = None) -> None:
        self.class_id = class_id
        self.property_ids = property_ids


class ImportAxiom(KGAxiom):
    """``owl:imports``: the source ontology imports another by IRI.

    The importer does **not** follow these references; this record exists so
    the UI can surface the dependency to the user.
    """

    FIELDS = ("target_iri", "source_ontology_iri")
    target_iri = _id_property("target_iri", "str: IRI of the imported ontology.")
    source_ontology_iri = _id_property(
        "source_ontology_iri", "str | None: IRI of the importing ontology, if known.", optional=True
    )

    def __init__(self, target_iri: str = "", source_ontology_iri: Optional[str] = None) -> None:
        self.target_iri = target_iri
        self.source_ontology_iri = source_ontology_iri
