"""Knowledge Graph metamodel.

A lightweight, typed in-memory representation of a knowledge graph used by the
Web Modeling Editor's KG diagram and by OWL-to-KG ingestion. The metamodel
distinguishes five node kinds (class, individual, property, literal, blank) and
directed edges between them.

There is intentionally no Python code builder for this metamodel: KG diagrams
are stored as JSON in projects, and the typed metamodel exists to back the
JSON round-trip converters and future KG-to-ClassDiagram / KG-to-ObjectDiagram
transformations. Because KG identifiers (IRIs, blank-node ids, literal values)
are not Python-identifier-safe, KGNode inherits from ``Element`` rather than
``NamedElement``; IRIs and labels are preserved verbatim.
"""

from abc import ABC
from typing import Optional, Set

from besser.BUML.metamodel.structural import Element, Model


__all__ = [
    "KGNode",
    "KGClass",
    "KGIndividual",
    "KGProperty",
    "KGLiteral",
    "KGBlank",
    "KGEdge",
    "KnowledgeGraph",
]


class KGNode(Element, ABC):
    """Base class for every node in a Knowledge Graph.

    Args:
        id: Stable identifier unique within the graph.
        label: Human-readable label shown in the editor.
        iri: Optional IRI (populated for OWL-sourced nodes).
    """

    def __init__(self, id: str, label: str = "", iri: Optional[str] = None):
        super().__init__()
        self.id = id
        self.label = label
        self.iri = iri

    @property
    def id(self) -> str:
        return self.__id

    @id.setter
    def id(self, value: str):
        if value is None or not isinstance(value, str) or value == "":
            raise ValueError("KGNode.id must be a non-empty string.")
        self.__id = value

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, value: str):
        self.__label = value if value is not None else ""

    @property
    def iri(self) -> Optional[str]:
        return self.__iri

    @iri.setter
    def iri(self, value: Optional[str]):
        self.__iri = value

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
    """

    def __init__(self, id: str, value: str, datatype: Optional[str] = None, label: str = ""):
        super().__init__(id, label=label if label else str(value), iri=None)
        self.value = value
        self.datatype = datatype

    @property
    def value(self) -> str:
        return self.__value

    @value.setter
    def value(self, value: str):
        if value is None:
            raise ValueError("KGLiteral.value cannot be None.")
        self.__value = str(value)

    @property
    def datatype(self) -> Optional[str]:
        return self.__datatype

    @datatype.setter
    def datatype(self, value: Optional[str]):
        self.__datatype = value


class KGEdge(Element):
    """A directed relationship between two KG nodes.

    Args:
        id: Stable identifier unique within the graph.
        source: Source node.
        target: Target node.
        label: Optional display label (often the predicate's local name).
        iri: Optional predicate IRI.
    """

    def __init__(self, id: str, source: KGNode, target: KGNode, label: str = "", iri: Optional[str] = None):
        super().__init__()
        self.id = id
        self.source = source
        self.target = target
        self.label = label
        self.iri = iri

    @property
    def id(self) -> str:
        return self.__id

    @id.setter
    def id(self, value: str):
        if value is None or not isinstance(value, str) or value == "":
            raise ValueError("KGEdge.id must be a non-empty string.")
        self.__id = value

    @property
    def source(self) -> KGNode:
        return self.__source

    @source.setter
    def source(self, value: KGNode):
        if not isinstance(value, KGNode):
            raise ValueError("KGEdge.source must be a KGNode instance.")
        self.__source = value

    @property
    def target(self) -> KGNode:
        return self.__target

    @target.setter
    def target(self, value: KGNode):
        if not isinstance(value, KGNode):
            raise ValueError("KGEdge.target must be a KGNode instance.")
        self.__target = value

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, value: str):
        self.__label = value if value is not None else ""

    @property
    def iri(self) -> Optional[str]:
        return self.__iri

    @iri.setter
    def iri(self, value: Optional[str]):
        self.__iri = value

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
    """

    def __init__(self, name: str = "knowledge_graph", nodes: Optional[Set[KGNode]] = None,
                 edges: Optional[Set[KGEdge]] = None):
        super().__init__(name)
        self.nodes = nodes if nodes is not None else set()
        self.edges = edges if edges is not None else set()

    @property
    def nodes(self) -> Set[KGNode]:
        return self.__nodes

    @nodes.setter
    def nodes(self, value: Set[KGNode]):
        if value is None:
            self.__nodes = set()
            return
        for n in value:
            if not isinstance(n, KGNode):
                raise ValueError("KnowledgeGraph.nodes must contain only KGNode instances.")
        self.__nodes = set(value)

    @property
    def edges(self) -> Set[KGEdge]:
        return self.__edges

    @edges.setter
    def edges(self, value: Set[KGEdge]):
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
        if not isinstance(node, KGNode):
            raise ValueError("add_node expects a KGNode instance.")
        self.__nodes.add(node)

    def add_edge(self, edge: KGEdge) -> None:
        if not isinstance(edge, KGEdge):
            raise ValueError("add_edge expects a KGEdge instance.")
        node_ids = {n.id for n in self.nodes}
        if edge.source.id not in node_ids or edge.target.id not in node_ids:
            raise ValueError(
                f"KGEdge {edge.id!r} references node ids not present in the graph "
                f"(source={edge.source.id!r}, target={edge.target.id!r})."
            )
        self.__edges.add(edge)

    def get_node(self, node_id: str) -> Optional[KGNode]:
        for n in self.__nodes:
            if n.id == node_id:
                return n
        return None

    def __repr__(self):
        return f"KnowledgeGraph(name={self.name!r}, nodes={len(self.nodes)}, edges={len(self.edges)})"
