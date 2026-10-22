"""Knowledge Graph metamodel.

A lightweight, typed in-memory representation of a knowledge graph used by the
Web Modeling Editor's KG diagram and by OWL-to-KG ingestion. The metamodel
distinguishes five node kinds (class, individual, property, literal, blank) and
directed edges between them.

KG diagrams are stored as JSON in projects; the typed metamodel backs the JSON
round-trip converters and the KG-to-ClassDiagram / KG-to-ObjectDiagram
transformations. Exporting a project to Python source goes through
``besser.utilities.buml_code_builder.kg_model_builder``, whose output is read
back by ``kg_diagram_converter.kg_buml_to_json``. Because KG identifiers (IRIs,
blank-node ids, literal values) are not Python-identifier-safe, KGNode inherits
from ``Element`` rather than ``NamedElement``; IRIs and labels are preserved
verbatim, and the builder derives variable names separately.
"""

import copy
from abc import ABC
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set

from besser.BUML.metamodel.structural import Element, Model
from besser.BUML.metamodel.kg.axioms import KGAxiom
from besser.BUML.metamodel.kg.constraint_specs import is_valid_spec


__all__ = [
    "KGNode",
    "KGClass",
    "KGIndividual",
    "KGProperty",
    "KGLiteral",
    "KGBlank",
    "KGNodeConstraint",
    "KGPropertyConstraint",
    "KGEdge",
    "KnowledgeGraph",
]


#: Values handed out as they are: immutable, so a reader cannot change the stored metadata through them.
_IMMUTABLE_VALUE_TYPES = (str, int, float, bool, bytes, type(None))


def _detached(value: Any) -> Any:
    """``value`` itself when it is immutable, otherwise a deep copy of it."""
    return value if isinstance(value, _IMMUTABLE_VALUE_TYPES) else copy.deepcopy(value)


class _MetadataView(Mapping):
    """Read-only view over an element's stored metadata dict.

    Nothing is copied when the view is made, and a scalar entry is returned as
    it is, so reading one field (``node.metadata.get("kind")``) costs a dict
    lookup. A list or dict entry -- a constraint node's ``constraintSpecs`` --
    is deep-copied when it is read, so no reference into the stored metadata
    ever leaves the element. Every write stores a new dict, so a view keeps
    showing the entries it was made over.
    """

    __slots__ = ("__data",)

    def __init__(self, data: Dict[str, Any]):
        self.__data = data

    def __getitem__(self, key: str) -> Any:
        return _detached(self.__data[key])

    def __contains__(self, key: object) -> bool:
        # Mapping's default goes through __getitem__ and would copy the value.
        return key in self.__data

    def get(self, key: str, default: Any = None) -> Any:
        data = self.__data
        return _detached(data[key]) if key in data else default

    def __iter__(self) -> Iterator[str]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__data!r})"


class _KGAnnotatedElement(Element):
    """Label, IRI and metadata, shared by :class:`KGNode` and :class:`KGEdge`.

    ``metadata`` is stored deep-copied and handed out as a read-only view that
    copies a nested list or dict as it is read: nothing reachable from the
    getter writes through to the element, so every change goes through the
    setter (and a subclass's :meth:`_checked_metadata`).
    """

    @property
    def label(self) -> str:
        """str: Get the display label (empty when none was given)."""
        return self.__label

    @label.setter
    def label(self, value: Optional[str]) -> None:
        """str | None: Set the display label; ``None`` becomes the empty string.

        Raises:
            ValueError: If ``value`` is neither a ``str`` nor ``None``.
        """
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(
                f"{type(self).__name__}.label must be a str or None, got {type(value).__name__}."
            )
        self.__label = value

    @property
    def iri(self) -> Optional[str]:
        """str | None: Get the IRI, or ``None`` when there is none."""
        return self.__iri

    @iri.setter
    def iri(self, value: Optional[str]) -> None:
        """str | None: Set the IRI.

        Raises:
            ValueError: If ``value`` is neither a ``str`` nor ``None``.
        """
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"{type(self).__name__}.iri must be a str or None, got {type(value).__name__}."
            )
        self.__iri = value

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Mapping: Get a read-only view of the free-form annotations.

        The view is made without copying and cannot be written to; a list or
        dict read from it is a deep copy of the stored one. So every change
        goes through the setter (and so through a subclass's validation -- a
        constraint node checks its ``constraintSpecs``). Change one entry with
        :meth:`set_metadata_item`, several with :meth:`update_metadata`, or
        replace it all by assignment. ``dict(element.metadata)`` gives a
        mutable copy.
        """
        return _MetadataView(self.__metadata)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Return one metadata entry, or ``default`` when it is absent.

        Like ``element.metadata.get(key, default)``: a list or dict value is
        a deep copy, so changing it does not change the element.
        """
        if key not in self.__metadata:
            return default
        return _detached(self.__metadata[key])

    @metadata.setter
    def metadata(self, value: Optional[Mapping[str, Any]]) -> None:
        """Mapping | None: Replace the metadata with a deep copy of ``value``; ``None`` empties it.

        Raises:
            ValueError: If ``value`` is neither a mapping nor ``None``, or a
                subclass rejects it.
        """
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{type(self).__name__}.metadata must be a dict or None.")
        self.__metadata = copy.deepcopy(self._checked_metadata(dict(value)))

    def _checked_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for subclasses that constrain the metadata; raise ``ValueError`` to reject."""
        return value

    def update_metadata(self, values: Mapping[str, Any]) -> None:
        """Set several metadata entries at once, through the validating setter.

        Raises:
            ValueError: If the resulting metadata is invalid for this element.
        """
        self.metadata = {**self.__metadata, **values}

    def set_metadata_item(self, key: str, value: Any) -> None:
        """Set one metadata entry, through the validating setter.

        Raises:
            ValueError: If the resulting metadata is invalid for this element.
        """
        self.update_metadata({key: value})


class KGNode(_KGAnnotatedElement, ABC):
    """Base class for every node in a Knowledge Graph.

    Args:
        id: Stable identifier unique within the graph.
        label: Human-readable label shown in the editor.
        iri: Optional IRI (populated for OWL-sourced nodes).
        metadata: Free-form dict of structured annotations attached during
            import (e.g., ``{"kind": "restriction", "on_property": iri,
            "restriction_type": "minCardinality", "value": 1}`` on a
            ``KGBlank`` representing an ``owl:Restriction``). Defaults to an
            empty dict; never participates in identity / hashing.
    """

    def __init__(
        self,
        id: str,
        label: str = "",
        iri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.id = id
        self.label = label
        self.iri = iri
        self.metadata = metadata

    @property
    def id(self) -> str:
        """str: Get the id, which is unique within a knowledge graph."""
        return self.__id

    @id.setter
    def id(self, value: str) -> None:
        """str: Set the id.

        Raises:
            ValueError: If the id is not a non-empty string.
        """
        if value is None or not isinstance(value, str) or value == "":
            raise ValueError("KGNode.id must be a non-empty string.")
        self.__id = value

    def __hash__(self):
        return hash((type(self).__name__, self.id))

    def __eq__(self, other):
        return isinstance(other, KGNode) and type(self) is type(other) and self.id == other.id

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id!r}, label={self.label!r})"


class KGClass(KGNode):
    """A class / concept / type (e.g. ``owl:Class``)."""


class KGIndividual(KGNode):
    """An individual / instance (e.g. ``owl:NamedIndividual``)."""


class KGProperty(KGNode):
    """A reified property node — a property that is itself the subject or
    object of further assertions (e.g. ``owl:ObjectProperty``)."""


class KGBlank(KGNode):
    """An anonymous resource (RDF blank node)."""


class KGLiteral(KGNode):
    """A literal value (string, number, typed literal, …).

    Args:
        id: Stable identifier.
        value: Lexical form of the literal.
        datatype: Optional datatype IRI (e.g. ``xsd:integer``).
        label: Optional display label; defaults to ``value`` when omitted.
        metadata: Free-form annotations dict (forwarded to :class:`KGNode`).
    """

    def __init__(
        self,
        id: str,
        value: str,
        datatype: Optional[str] = None,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(id, label=label if label else str(value), iri=None, metadata=metadata)
        self.value = value
        self.datatype = datatype

    @property
    def value(self) -> str:
        """str: Get the lexical form of the literal."""
        return self.__value

    @value.setter
    def value(self, value: str) -> None:
        """str: Set the lexical form (stored as ``str(value)``).

        Raises:
            ValueError: If ``value`` is ``None``.
        """
        if value is None:
            raise ValueError("KGLiteral.value cannot be None.")
        self.__value = str(value)

    @property
    def datatype(self) -> Optional[str]:
        """str | None: Get the datatype IRI (e.g. ``xsd:integer``), if any."""
        return self.__datatype

    @datatype.setter
    def datatype(self, value: Optional[str]) -> None:
        """str | None: Set the datatype IRI."""
        self.__datatype = value


class _ConstraintNodeMixin:
    """Validates ``metadata['constraintSpecs']`` on KGNodeConstraint / KGPropertyConstraint.

    Reuses the shared metadata storage through the :meth:`_checked_metadata`
    hook, so every way of changing the metadata gains the structural check.
    Specs are kept as plain dicts so they round-trip through the existing JSON
    converters unchanged.
    """

    def _checked_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """Reject ``constraintSpecs`` that is not a list of valid specs."""
        specs = value.get("constraintSpecs")
        if specs is not None:
            if not isinstance(specs, list):
                raise ValueError(
                    f"{type(self).__name__}.metadata['constraintSpecs'] must be a list."
                )
            for idx, spec in enumerate(specs):
                if not is_valid_spec(spec):
                    raise ValueError(
                        f"{type(self).__name__}.metadata['constraintSpecs'][{idx}] is invalid: {spec!r}"
                    )
        return super()._checked_metadata(value)

    def add_spec(self, spec) -> None:
        """Append a constraint spec to ``metadata['constraintSpecs']``.

        Args:
            spec: A spec dict accepted by ``is_valid_spec``.

        Raises:
            ValueError: If the spec is invalid.
        """
        if not is_valid_spec(spec):
            raise ValueError(f"Invalid constraint spec: {spec!r}")
        self.set_metadata_item("constraintSpecs", [*self.get_specs(), spec])

    def remove_spec(self, index: int) -> None:
        """Remove the spec at ``index``; an out-of-range index does nothing.

        Args:
            index: Position in ``metadata['constraintSpecs']``.
        """
        specs = self.get_specs()
        if 0 <= index < len(specs):
            specs.pop(index)
            self.set_metadata_item("constraintSpecs", specs)

    def get_specs(self) -> List[Dict[str, Any]]:
        """list[dict]: Get a deep copy of the node's constraint specs."""
        return self.get_metadata("constraintSpecs", [])


class KGNodeConstraint(_ConstraintNodeMixin, KGNode):
    """A reified constraint that applies to a class.

    Materialises an OWL restriction set or a SHACL ``sh:NodeShape`` as a
    first-class visual node. The constraint payload lives in
    ``metadata['constraintSpecs']`` as a list of dicts whose schema is
    documented in :mod:`besser.BUML.metamodel.kg.constraint_specs`.

    Linked to its target class via an edge with iri
    ``http://besser-pearl.org/kg#constraintTargetClass`` (and, for SHACL
    NodeShapes, may also carry the standard ``sh:targetClass`` predicate
    on export).
    """


class KGPropertyConstraint(_ConstraintNodeMixin, KGNode):
    """A reified constraint that applies to a property.

    Materialises an OWL ``owl:Restriction`` or a SHACL ``sh:PropertyShape``
    as a first-class visual node. Linked to a :class:`KGProperty` via the
    ``constraintTargetProperty`` edge and/or grouped under a
    :class:`KGNodeConstraint` via the ``sh:property`` edge.
    """


class KGEdge(_KGAnnotatedElement):
    """A directed relationship between two KG nodes.

    Args:
        id: Stable identifier unique within the graph.
        source: Source node.
        target: Target node.
        label: Optional display label (often the predicate's local name).
        iri: Optional predicate IRI.
        metadata: Free-form dict of structured annotations attached during
            import. Defaults to an empty dict; never participates in identity
            / hashing.
    """

    def __init__(
        self,
        id: str,
        source: KGNode,
        target: KGNode,
        label: str = "",
        iri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.id = id
        self.source = source
        self.target = target
        self.label = label
        self.iri = iri
        self.metadata = metadata

    @property
    def id(self) -> str:
        """str: Get the id, unique among a graph's edges."""
        return self.__id

    @id.setter
    def id(self, value: str) -> None:
        """str: Set the id.

        Raises:
            ValueError: If the id is not a non-empty string.
        """
        if value is None or not isinstance(value, str) or value == "":
            raise ValueError("KGEdge.id must be a non-empty string.")
        self.__id = value

    @property
    def source(self) -> KGNode:
        """KGNode: Get the node the edge starts at."""
        return self.__source

    @source.setter
    def source(self, value: KGNode) -> None:
        """KGNode: Set the source node.

        Raises:
            ValueError: If ``value`` is not a :class:`KGNode`.
        """
        if not isinstance(value, KGNode):
            raise ValueError("KGEdge.source must be a KGNode instance.")
        self.__source = value

    @property
    def target(self) -> KGNode:
        """KGNode: Get the node the edge points to."""
        return self.__target

    @target.setter
    def target(self, value: KGNode) -> None:
        """KGNode: Set the target node.

        Raises:
            ValueError: If ``value`` is not a :class:`KGNode`.
        """
        if not isinstance(value, KGNode):
            raise ValueError("KGEdge.target must be a KGNode instance.")
        self.__target = value

    def __hash__(self):
        return hash(("KGEdge", self.id))

    def __eq__(self, other):
        return isinstance(other, KGEdge) and self.id == other.id

    def __repr__(self):
        return f"KGEdge(id={self.id!r}, {self.source.id!r} -> {self.target.id!r}, label={self.label!r})"


class KnowledgeGraph(Model):
    """Top-level Knowledge Graph container.

    Args:
        name: Python-identifier-safe name for the model (``NamedElement.name`` rules apply).
        nodes: Initial set of nodes.
        edges: Initial set of edges; their source/target must be in ``nodes``.
        axioms: Initial list of decoded OWL axioms (equivalentClass, disjoint,
            inverseOf, subPropertyOf, …) that don't naturally belong on a single
            node or edge. Defaults to an empty list.
    """

    def __init__(
        self,
        name: str = "knowledge_graph",
        nodes: Optional[Set[KGNode]] = None,
        edges: Optional[Set[KGEdge]] = None,
        axioms: Optional[List[KGAxiom]] = None,
    ):
        super().__init__(name)
        self.nodes = nodes if nodes is not None else set()
        self.edges = edges if edges is not None else set()
        self.axioms = axioms

    @property
    def nodes(self) -> Set[KGNode]:
        """set[KGNode]: Get the graph's nodes."""
        return self.__nodes

    @nodes.setter
    def nodes(self, value: Set[KGNode]) -> None:
        """Replace the node set, rejecting duplicate ids.

        ``KGNode.__hash__`` keys on ``(type, id)``, so a plain ``set`` happily
        holds ``KGClass(id="x")`` alongside ``KGIndividual(id="x")`` -- and the
        docs promise ids are unique. When both are present, ``kg_model_to_code``
        reattaches an edge to whichever node it finds first and the other is
        lost. Mirrors ``DomainModel.types``, which rejects duplicate names the
        same way.
        """
        if value is None:
            self.__nodes = set()
            return
        for n in value:
            if not isinstance(n, KGNode):
                raise ValueError("KnowledgeGraph.nodes must contain only KGNode instances.")
        self.__nodes = set(value)
        self._reject_duplicate_node_ids()

    @property
    def edges(self) -> Set[KGEdge]:
        """set[KGEdge]: Get the graph's edges."""
        return self.__edges

    @edges.setter
    def edges(self, value: Set[KGEdge]) -> None:
        """set[KGEdge] | None: Replace the edge set; ``None`` empties it.

        Raises:
            ValueError: If an element is not a :class:`KGEdge`, or an edge's endpoint is not among the graph's nodes.
        """
        if value is None:
            self.__edges = set()
            return
        node_ids = {n.id for n in self.nodes}
        for e in value:
            if not isinstance(e, KGEdge):
                raise ValueError("KnowledgeGraph.edges must contain only KGEdge instances.")
            if e.source.id not in node_ids or e.target.id not in node_ids:
                raise ValueError(
                    f"KGEdge {e.id!r} references node ids not present in the graph "
                    f"(source={e.source.id!r}, target={e.target.id!r})."
                )
        self.__edges = set(value)

    def add_node(self, node: KGNode) -> None:
        """Add one node, rejecting an id another node already uses."""
        if not isinstance(node, KGNode):
            raise ValueError("add_node expects a KGNode instance.")
        clash = next((n for n in self.__nodes if n.id == node.id and n is not node), None)
        if clash is not None:
            raise ValueError(
                f"KnowledgeGraph already holds a node with id {node.id!r} "
                f"({type(clash).__name__}); node ids must be unique across types."
            )
        self.__nodes.add(node)

    def _reject_duplicate_node_ids(self) -> None:
        seen: Dict[str, str] = {}
        duplicates: Set[str] = set()
        for n in self.__nodes:
            if n.id in seen:
                duplicates.add(n.id)
            seen[n.id] = type(n).__name__
        if duplicates:
            raise ValueError(
                "KnowledgeGraph cannot hold nodes with duplicate ids: "
                + ", ".join(sorted(duplicates))
            )

    def add_edge(self, edge: KGEdge) -> None:
        """Add one edge.

        Raises:
            ValueError: If ``edge`` is not a :class:`KGEdge`, or an endpoint is not among the graph's nodes.
        """
        if not isinstance(edge, KGEdge):
            raise ValueError("add_edge expects a KGEdge instance.")
        node_ids = {n.id for n in self.nodes}
        if edge.source.id not in node_ids or edge.target.id not in node_ids:
            raise ValueError(
                f"KGEdge {edge.id!r} references node ids not present in the graph "
                f"(source={edge.source.id!r}, target={edge.target.id!r})."
            )
        self.__edges.add(edge)

    @property
    def axioms(self) -> List[KGAxiom]:
        """list[KGAxiom]: Get a copy of the OWL axioms that relate several nodes at once.

        Change them with :meth:`add_axiom` or by assigning the property, not by
        mutating the list returned.
        """
        return list(self.__axioms)

    @axioms.setter
    def axioms(self, value: Optional[List[KGAxiom]]) -> None:
        """list[KGAxiom] | None: Replace the axioms with a copy of ``value``; ``None`` empties them.

        Raises:
            ValueError: If ``value`` is not a list of :class:`KGAxiom`.
        """
        if value is None:
            self.__axioms = []
            return
        if not isinstance(value, list):
            raise ValueError("KnowledgeGraph.axioms must be a list or None.")
        for a in value:
            if not isinstance(a, KGAxiom):
                raise ValueError("KnowledgeGraph.axioms must contain only KGAxiom instances.")
        self.__axioms = list(value)

    def add_axiom(self, axiom: KGAxiom) -> None:
        """Append one axiom.

        Raises:
            ValueError: If ``axiom`` is not a :class:`KGAxiom`.
        """
        if not isinstance(axiom, KGAxiom):
            raise ValueError("add_axiom expects a KGAxiom instance.")
        self.__axioms.append(axiom)

    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Return the node with id ``node_id``.

        Returns:
            KGNode | None: The node, or ``None`` when no node has that id.
        """
        for n in self.__nodes:
            if n.id == node_id:
                return n
        return None

    def validate(self, raise_exception: bool = True) -> dict:
        """Validate the graph, collecting every problem rather than the first.

        ``KnowledgeGraph`` was the only ``Model`` without this, so a graph
        assembled by assigning ``kg.nodes`` / ``kg.edges`` directly (which the
        setters allow, and which skips ``add_edge``'s endpoint check in the
        order they are assigned) could carry dangling edges unnoticed.

        Args:
            raise_exception: raise ``ValueError`` when validation fails.

        Returns:
            dict: ``{"success": bool, "errors": list, "warnings": list}``,
            matching ``DomainModel.validate`` and ``StateMachine.validate``.
        """
        errors: List[str] = []
        warnings: List[str] = []

        self._validate_unique_node_ids(errors)
        self._validate_edge_endpoints(errors)
        self._validate_constraint_specs(errors)
        self._validate_axiom_types(errors)
        self._validate_axiom_references(warnings)

        result = {"success": len(errors) == 0, "errors": errors, "warnings": warnings}
        if errors and raise_exception:
            raise ValueError("\n".join(errors))
        return result

    def _validate_unique_node_ids(self, errors: List[str]) -> None:
        seen: Set[str] = set()
        for n in sorted(self.__nodes, key=lambda x: x.id):
            if n.id in seen:
                errors.append(f"Duplicate node id {n.id!r} in the knowledge graph.")
            seen.add(n.id)

    def _validate_edge_endpoints(self, errors: List[str]) -> None:
        node_ids = {n.id for n in self.__nodes}
        for e in sorted(self.__edges, key=lambda x: x.id):
            for role, endpoint in (("source", e.source), ("target", e.target)):
                if endpoint.id not in node_ids:
                    errors.append(
                        f"KGEdge {e.id!r} has a dangling {role}: node "
                        f"{endpoint.id!r} is not in the graph."
                    )

    def _validate_constraint_specs(self, errors: List[str]) -> None:
        """Constraint nodes validate their specs on assignment; this catches state that got
        around that (an unpickled graph, or one assembled through private attributes)."""
        for n in sorted(self.__nodes, key=lambda x: x.id):
            if not isinstance(n, _ConstraintNodeMixin):
                continue
            for idx, spec in enumerate(n.get_specs()):
                if not is_valid_spec(spec):
                    errors.append(f"{type(n).__name__} {n.id!r} has an invalid constraint spec at index {idx}: {spec!r}")

    def _validate_axiom_types(self, errors: List[str]) -> None:
        """The setter checks this; catch state that got around it."""
        for idx, axiom in enumerate(self.__axioms):
            if not isinstance(axiom, KGAxiom):
                errors.append(
                    f"KnowledgeGraph.axioms[{idx}] is a {type(axiom).__name__}, not a KGAxiom."
                )

    def _validate_axiom_references(self, warnings: List[str]) -> None:
        """Flag every axiom id that names no node of the graph.

        A warning, not an error: an imported ontology may legitimately refer to
        an IRI it does not define (a ``rdfs:subPropertyOf foaf:knows`` without
        ``foaf:knows`` itself), and that should not block a conversion.
        """
        node_ids = {n.id for n in self.__nodes}
        for axiom in self.__axioms:
            if not isinstance(axiom, KGAxiom):
                continue
            for field_name, refs in axiom.referenced_ids().items():
                for ref in refs:
                    if ref not in node_ids:
                        warnings.append(
                            f"{type(axiom).__name__}.{field_name} references "
                            f"{ref!r}, which is not a node in the graph."
                        )

    def __repr__(self):
        return f"KnowledgeGraph(name={self.name!r}, nodes={len(self.nodes)}, edges={len(self.edges)})"
