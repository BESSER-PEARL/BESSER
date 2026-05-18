"""UML Deployment metamodel for B-UML.

A first-class B-UML model for UML Deployment diagrams, alongside ``structural``
/ ``state_machine`` / ``gui`` / ``bpmn`` / ``uml_component``. Implements the
design in ``.claude/component-deployment/01-component-deployment-design.md``
(reviewed and locked 2026-05-18).

Distinct from the existing ``besser/BUML/metamodel/deployment/`` package which
models cloud-infrastructure concepts (K8s ``Cluster``, ``Deployment``,
``Node`` with public/private IPs, ``Region``, …) and feeds the Terraform
generator. That package stays untouched; this one models UML 2.5 Deployment
notation (``Node``, ``Artifact``, ``DeploymentRelation``, ``CommunicationPath``)
with an agentic stereotype profile aligned with ``uml_component``.

Hierarchy::

    Element -> NamedElement -> DeploymentElement -> {Node, Artifact, Interface,
                                                     DeploymentRelationship}
                            -> Model -> DeploymentModel

    DeploymentRelationship <|-- DeploymentRelation     # Artifact -> Node
                          <|-- CommunicationPath      # Node <-> Node
                          <|-- DeploymentDependency
                          <|-- InterfaceProvided
                          <|-- InterfaceRequired

Naming-clash notes (recorded in 00- §8):

* UML 2.5 ``Node`` collides with ``besser.BUML.metamodel.deployment.Node``
  (a compute node with public/private IPs). Same class name, different
  packages -- Python namespacing resolves cleanly; reader-facing
  ambiguity bounded.
* UML 2.5 calls the artifact-on-node relationship ``Deployment``. The
  existing cloud-infra package already has a class named ``Deployment``
  (a K8s Deployment object), so this metamodel uses ``DeploymentRelation``
  to disambiguate without losing meaning.

The agentic stereotype profile (D3 hybrid) carries swarm intent via typed
discriminator attributes (``NodeKind``, ``Locality``) plus a free-form
``stereotypes: List[str]`` passthrough on every element. ``Artifact.manifests:
List[str]`` is the cross-diagram link to ``uml_component.Component`` IDs
(D2: artifact manifests component; D13 stable-ID cross-ref).

Cross-diagram links to ``uml_component.Component`` are *not* a Python
import (decision D13 -- string IDs only). This metamodel reuses
``structural.Multiplicity`` on ``DeploymentRelation`` for ``[3]`` / ``[1..*]``
artifact counts on a node (R-06, R-20).

Growth path (01- §3.5):

* UML ``Manifestation`` as a distinct relationship class -- currently
  absorbed into ``Artifact.manifests``.
* UML ``DeploymentSpecification`` -- not modelled.
* ``«autoscale»`` policy attributes on ``DeploymentRelation``.
* AgentGroup as a resource grouping (NR-6) -- via the Component-side
  ``Subsystem`` plus an Artifact that manifests it.
"""

from enum import Enum
from typing import List, Optional

from besser.BUML.metamodel.structural import Model, Multiplicity, NamedElement


# ---------------------------------------------------------------------------
# Enumerations (decisions D4 / D5 -- plain enum.Enum; .value strings match the
# WME / UML 2.5 well-known stereotype strings so converters can map by value)
# ---------------------------------------------------------------------------

class NodeKind(Enum):
    """The kind of a UML Node (UML 2.5 §19.2).

    Defaults to ``GENERIC`` to match Apollon's default ``«node»`` stereotype.
    ``DEVICE`` and ``EXECUTION_ENVIRONMENT`` are UML's well-known
    specialisations.
    """
    GENERIC = "node"
    DEVICE = "device"
    EXECUTION_ENVIRONMENT = "executionEnvironment"


class Locality(Enum):
    """Where a Node / Artifact is hosted, generalised from a binary ``«external»``
    stereotype per NR-5 of the requirements review.

    Mirrors the same-named enum in ``uml_component`` -- the locality
    classification is meaningful at any of the three levels (Component,
    Artifact, Node). Kept as a separate class so each metamodel is
    independently usable.

    * ``LOCAL`` -- owned and hosted by us.
    * ``EXTERNAL`` -- third-party API or service we don't own.
    * ``HYBRID`` -- on-prem with cloud fallback / mirror.
    """
    LOCAL = "local"
    EXTERNAL = "external"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _checked_set(values, expected_type, label: str) -> set:
    """Coerce ``values`` to a set, raising TypeError if any element is not ``expected_type``."""
    result = set(values)
    for value in result:
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{label} must contain {expected_type.__name__} instances, "
                f"got {type(value).__name__}"
            )
    return result


def _checked_str_list(values, label: str) -> List[str]:
    """Coerce ``values`` to a list of str, raising TypeError on a non-str entry."""
    result = list(values)
    for value in result:
        if not isinstance(value, str):
            raise TypeError(
                f"{label} must contain str entries, got {type(value).__name__}"
            )
    return result


# ---------------------------------------------------------------------------
# Base element (decisions D9 / D10 -- relaxed name, opaque layout passthrough)
# ---------------------------------------------------------------------------

class DeploymentElement(NamedElement):
    """Base class for every UML Deployment-diagram abstract-syntax element.

    Relaxes ``NamedElement.name`` (decision D9): a Deployment label is free
    text and may contain spaces (``"UNP sandbox VM"``), colons, brackets
    (``"Code Tester [3]"``), or be empty. Carries ``layout`` -- the opaque
    diagram-interchange passthrough the metamodel never interprets (D10).

    Args:
        name (str): The element label. Empty allowed; ``None`` -> ``""``.
        stereotypes (List[str]): Free-form stereotype passthrough.
        layout (dict): Opaque DI data.
        metadata (Metadata): Inherited from NamedElement.
        timestamp (datetime): Inherited from NamedElement.

    Attributes:
        name (str): The element label.
        stereotypes (List[str]): Free-form stereotype passthrough.
        layout (dict): Opaque DI passthrough, or None.
    """

    def __init__(self, name: str = "", stereotypes: List[str] = None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(name=name, metadata=metadata, timestamp=timestamp)
        self.stereotypes = stereotypes if stereotypes is not None else []
        self.layout = layout

    @NamedElement.name.setter
    def name(self, name: str):
        """str: Set the Deployment-element label. Permits empty / whitespace /
        spaces / punctuation; ``None`` -> ``""``.

        Raises:
            TypeError: if name is neither a str nor None.
        """
        if name is None:
            name = ""
        if not isinstance(name, str):
            raise TypeError(
                f"DeploymentElement name must be a str or None, "
                f"got {type(name).__name__}"
            )
        self._NamedElement__name = name

    @property
    def stereotypes(self) -> List[str]:
        """List[str]: Get the free-form stereotype strings."""
        return self.__stereotypes

    @stereotypes.setter
    def stereotypes(self, stereotypes: List[str]):
        """List[str]: Set the free-form stereotype strings.

        Raises:
            TypeError: if not a list of str.
        """
        self.__stereotypes = _checked_str_list(stereotypes, "stereotypes")

    @property
    def layout(self) -> Optional[dict]:
        """dict: Get the opaque DI passthrough, or None."""
        return self.__layout

    @layout.setter
    def layout(self, layout: Optional[dict]):
        """dict: Set the opaque DI passthrough.

        Raises:
            TypeError: if layout is neither a dict nor None.
        """
        if layout is not None and not isinstance(layout, dict):
            raise TypeError(f"layout must be a dict or None, got {type(layout).__name__}")
        self.__layout = layout

    def __repr__(self):
        return f"{type(self).__name__}(name='{self.name}')"


# ---------------------------------------------------------------------------
# Nodes and Artifacts (decision D12)
# ---------------------------------------------------------------------------

class Node(DeploymentElement):
    """A UML Node -- a deployment target (UML 2.5 §19.2).

    Apollon emits a single ``DeploymentNode`` element with a free-form
    ``stereotype`` string. We map the WME stereotype to ``NodeKind`` (D4):
    ``«device»`` / ``«executionEnvironment»`` / fall-through to ``GENERIC``.

    Args:
        name (str): The node label.
        kind (NodeKind): The node's kind. Defaults to ``GENERIC``.
        locality (Locality): Where the node is hosted. Defaults to ``LOCAL``.
        nested_artifacts (set[Artifact]): Child artifacts inside this node.
            Each child's ``parent`` back-reference is maintained automatically.
        nested_nodes (set[Node]): Child nodes inside this node (e.g.
            ExecutionEnvironment inside a Device).
        stereotypes, layout, metadata, timestamp: Inherited.

    Attributes:
        kind (NodeKind): The node's kind.
        locality (Locality): Hosting locality.
        nested_artifacts (set[Artifact]): Child artifacts.
        nested_nodes (set[Node]): Child nodes.
        parent (Node | None): The containing node, or None at model root.
    """

    def __init__(self, name: str = "", kind: NodeKind = None,
                 locality: Locality = None,
                 nested_artifacts: set = None, nested_nodes: set = None,
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.kind = kind if kind is not None else NodeKind.GENERIC
        self.locality = locality if locality is not None else Locality.LOCAL
        self.__nested_artifacts: set = set()
        self.__nested_nodes: set = set()
        self.__parent: Optional["Node"] = None
        self.nested_artifacts = (nested_artifacts
                                 if nested_artifacts is not None else set())
        self.nested_nodes = nested_nodes if nested_nodes is not None else set()

    @property
    def kind(self) -> NodeKind:
        """NodeKind: Get the node's kind."""
        return self.__kind

    @kind.setter
    def kind(self, kind: NodeKind):
        """NodeKind: Set the node's kind.

        Raises:
            TypeError: if not a NodeKind.
        """
        if not isinstance(kind, NodeKind):
            raise TypeError(f"kind must be a NodeKind, got {type(kind).__name__}")
        self.__kind = kind

    @property
    def locality(self) -> Locality:
        """Locality: Get the hosting locality."""
        return self.__locality

    @locality.setter
    def locality(self, locality: Locality):
        """Locality: Set the hosting locality.

        Raises:
            TypeError: if not a Locality.
        """
        if not isinstance(locality, Locality):
            raise TypeError(f"locality must be a Locality, got {type(locality).__name__}")
        self.__locality = locality

    @property
    def nested_artifacts(self) -> set:
        """set[Artifact]: Get the child artifacts."""
        return self.__nested_artifacts

    @nested_artifacts.setter
    def nested_artifacts(self, nested_artifacts: set):
        """set[Artifact]: Set the child artifacts; maintains each child's
        ``parent`` back-reference.

        Raises:
            TypeError: if any element is not an Artifact.
        """
        nested_artifacts = set(nested_artifacts)
        for child in nested_artifacts:
            if not isinstance(child, Artifact):
                raise TypeError(
                    f"nested_artifacts must contain Artifact instances, "
                    f"got {type(child).__name__}"
                )
        for existing in self.__nested_artifacts:
            if existing.parent is self:
                existing.parent = None
        for child in nested_artifacts:
            child.parent = self
        self.__nested_artifacts = nested_artifacts

    def add_artifact(self, artifact: "Artifact"):
        """Add a nested artifact; sets its ``parent``.

        Raises:
            TypeError: if artifact is not an Artifact.
        """
        if not isinstance(artifact, Artifact):
            raise TypeError(
                f"artifact must be an Artifact, got {type(artifact).__name__}"
            )
        artifact.parent = self
        self.__nested_artifacts.add(artifact)

    def remove_artifact(self, artifact: "Artifact"):
        """Remove a nested artifact; clears its ``parent``."""
        if artifact in self.__nested_artifacts:
            if artifact.parent is self:
                artifact.parent = None
            self.__nested_artifacts.discard(artifact)

    @property
    def nested_nodes(self) -> set:
        """set[Node]: Get the child nodes."""
        return self.__nested_nodes

    @nested_nodes.setter
    def nested_nodes(self, nested_nodes: set):
        """set[Node]: Set the child nodes; maintains each child's ``parent``.

        Raises:
            TypeError: if any element is not a Node.
            ValueError: if self is in the children (self-containment).
        """
        nested_nodes = set(nested_nodes)
        for child in nested_nodes:
            if not isinstance(child, Node):
                raise TypeError(
                    f"nested_nodes must contain Node instances, "
                    f"got {type(child).__name__}"
                )
            if child is self:
                raise ValueError("A Node cannot be nested in itself.")
        for existing in self.__nested_nodes:
            if existing.parent is self:
                existing.parent = None
        for child in nested_nodes:
            child.parent = self
        self.__nested_nodes = nested_nodes

    def add_nested_node(self, node: "Node"):
        """Add a nested node; sets its ``parent``.

        Raises:
            TypeError: if node is not a Node.
            ValueError: if node is self.
        """
        if not isinstance(node, Node):
            raise TypeError(f"node must be a Node, got {type(node).__name__}")
        if node is self:
            raise ValueError("A Node cannot be nested in itself.")
        node.parent = self
        self.__nested_nodes.add(node)

    def remove_nested_node(self, node: "Node"):
        """Remove a nested node; clears its ``parent``."""
        if node in self.__nested_nodes:
            if node.parent is self:
                node.parent = None
            self.__nested_nodes.discard(node)

    @property
    def parent(self) -> Optional["Node"]:
        """Node: Get the containing node, or None."""
        return self.__parent

    @parent.setter
    def parent(self, parent: Optional["Node"]):
        """Node: Set the containing node.

        Raises:
            TypeError: if parent is neither a Node nor None.
        """
        if parent is not None and not isinstance(parent, Node):
            raise TypeError(
                f"parent must be a Node or None, got {type(parent).__name__}"
            )
        self.__parent = parent


class Artifact(DeploymentElement):
    """A UML Artifact -- the deployable unit (UML 2.5 §19.2).

    Manifests one or more ``uml_component.Component`` s (D2 / D13: cross-diagram
    link by stable Component IDs).

    Args:
        name (str): The artifact label.
        locality (Locality): Where the artifact is hosted. Defaults to ``LOCAL``.
        manifests (List[str]): Cross-diagram IDs of Components this artifact
            manifests. Default ``[]``.
        stereotypes, layout, metadata, timestamp: Inherited.

    Attributes:
        locality (Locality): Hosting locality.
        manifests (List[str]): Cross-diagram Component IDs.
        parent (Node | None): The containing node, or None at model root.
    """

    def __init__(self, name: str = "", locality: Locality = None,
                 manifests: List[str] = None,
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.locality = locality if locality is not None else Locality.LOCAL
        self.manifests = manifests if manifests is not None else []
        self.__parent: Optional[Node] = None

    @property
    def locality(self) -> Locality:
        """Locality: Get the hosting locality."""
        return self.__locality

    @locality.setter
    def locality(self, locality: Locality):
        """Locality: Set the hosting locality.

        Raises:
            TypeError: if not a Locality.
        """
        if not isinstance(locality, Locality):
            raise TypeError(f"locality must be a Locality, got {type(locality).__name__}")
        self.__locality = locality

    @property
    def manifests(self) -> List[str]:
        """List[str]: Cross-diagram IDs of Components this artifact manifests."""
        return self.__manifests

    @manifests.setter
    def manifests(self, manifests: List[str]):
        """List[str]: Set the cross-diagram Component IDs.

        Raises:
            TypeError: if not a list of str.
        """
        self.__manifests = _checked_str_list(manifests, "manifests")

    @property
    def parent(self) -> Optional[Node]:
        """Node: The containing node, or None if at model root."""
        return self.__parent

    @parent.setter
    def parent(self, parent: Optional[Node]):
        """Node: Set the containing node.

        Raises:
            TypeError: if parent is neither a Node nor None.
        """
        if parent is not None and not isinstance(parent, Node):
            raise TypeError(
                f"parent must be a Node or None, got {type(parent).__name__}"
            )
        self.__parent = parent


class Interface(DeploymentElement):
    """A UML Interface provided / required on a Node or Artifact (UML 2.5).

    Mirrors ``uml_component.Interface`` shape; kept as a separate class so
    each metamodel is independently usable (decision D11 / Q10=a).
    """


# ---------------------------------------------------------------------------
# Relationships (decision D11 -- abstract base + endpoint type-check seam;
# D6 -- Multiplicity reused on DeploymentRelation)
# ---------------------------------------------------------------------------

class DeploymentRelationship(DeploymentElement):
    """Abstract base for the Deployment-diagram relationship kinds.

    Each concrete subclass overrides ``_check_endpoint`` with its own endpoint
    type rule; model-wide checks live in ``DeploymentModel.validate()``.

    Attributes:
        source (DeploymentElement): The relationship's source element.
        target (DeploymentElement): The relationship's target element.
    """

    def __init__(self, source, target, name: str = "",
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        if type(self) is DeploymentRelationship:
            raise TypeError(
                "DeploymentRelationship is abstract and cannot be instantiated directly"
            )
        super().__init__(name=name, stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.__source = None
        self.__target = None
        self.source = source
        self.target = target

    @property
    def source(self):
        """DeploymentElement: Get the source element."""
        return self.__source

    @source.setter
    def source(self, source):
        """DeploymentElement: Set the source element.

        Raises:
            TypeError: if source is not acceptable for this relationship.
        """
        self._check_endpoint(source, "source")
        self.__source = source

    @property
    def target(self):
        """DeploymentElement: Get the target element."""
        return self.__target

    @target.setter
    def target(self, target):
        """DeploymentElement: Set the target element.

        Raises:
            TypeError: if target is not acceptable for this relationship.
        """
        self._check_endpoint(target, "target")
        self.__target = target

    def _check_endpoint(self, endpoint, role: str):
        """Type-check a single endpoint. Base requires any DeploymentElement;
        subclasses tighten.

        Raises:
            TypeError: if endpoint is not acceptable.
        """
        if not isinstance(endpoint, DeploymentElement):
            raise TypeError(
                f"{type(self).__name__} {role} must be a DeploymentElement, "
                f"got {type(endpoint).__name__}"
            )

    def __repr__(self):
        source_name = self.source.name if self.source is not None else None
        target_name = self.target.name if self.target is not None else None
        return (f"{type(self).__name__}(name='{self.name}', "
                f"source='{source_name}', target='{target_name}')")


class DeploymentRelation(DeploymentRelationship):
    """The UML 2.5 ``Deployment`` relationship -- artifact deployed on a node.

    Renamed from UML's ``Deployment`` to avoid clash with the cloud-infra
    ``besser.BUML.metamodel.deployment.Deployment`` (a K8s Deployment object).
    Carries the multiplicity of the artifact-instance count on this node
    (decision D6: reuse ``structural.Multiplicity``; placement on the
    relation, not on the artifact, matches conventional UML).

    Endpoint rule: ``source`` is ``Artifact``, ``target`` is ``Node``.

    Args:
        source (Artifact): The artifact being deployed.
        target (Node): The node it deploys on.
        multiplicity (Multiplicity): Instance count on this node.
            Defaults to ``Multiplicity(1, 1)``.
        name, stereotypes, layout, metadata, timestamp: Inherited.

    Attributes:
        multiplicity (Multiplicity): Instance count of source on target.
    """

    def __init__(self, source, target, multiplicity: Multiplicity = None,
                 name: str = "", stereotypes: List[str] = None,
                 layout: dict = None, metadata=None, timestamp=None):
        super().__init__(source=source, target=target, name=name,
                         stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.multiplicity = (multiplicity if multiplicity is not None
                             else Multiplicity(1, 1))

    @property
    def multiplicity(self) -> Multiplicity:
        """Multiplicity: Get the multiplicity (instance count on the target node)."""
        return self.__multiplicity

    @multiplicity.setter
    def multiplicity(self, multiplicity: Multiplicity):
        """Multiplicity: Set the multiplicity.

        Raises:
            TypeError: if not a Multiplicity.
        """
        if not isinstance(multiplicity, Multiplicity):
            raise TypeError(
                f"multiplicity must be a Multiplicity, "
                f"got {type(multiplicity).__name__}"
            )
        self.__multiplicity = multiplicity

    def _check_endpoint(self, endpoint, role: str):
        if role == "source":
            if not isinstance(endpoint, Artifact):
                raise TypeError(
                    f"DeploymentRelation source must be an Artifact, "
                    f"got {type(endpoint).__name__}"
                )
        else:  # target
            if not isinstance(endpoint, Node):
                raise TypeError(
                    f"DeploymentRelation target must be a Node, "
                    f"got {type(endpoint).__name__}"
                )


class CommunicationPath(DeploymentRelationship):
    """A communication path between two Nodes (UML 2.5 §19.2).

    Endpoint rule: both ``source`` and ``target`` are ``Node`` s.
    """

    def _check_endpoint(self, endpoint, role: str):
        if not isinstance(endpoint, Node):
            raise TypeError(
                f"CommunicationPath {role} must be a Node, "
                f"got {type(endpoint).__name__}"
            )


class DeploymentDependency(DeploymentRelationship):
    """A plain UML dependency between two Deployment elements (carries an
    arbitrary stereotype string on ``stereotypes``).
    """


class InterfaceProvided(DeploymentRelationship):
    """A Node-or-Artifact -> Interface edge: the source *provides* the
    Interface.

    Endpoint rule: ``source`` is ``Node`` or ``Artifact``, ``target`` is
    ``Interface``.
    """

    def _check_endpoint(self, endpoint, role: str):
        if role == "source":
            if not isinstance(endpoint, (Node, Artifact)):
                raise TypeError(
                    f"InterfaceProvided source must be a Node or Artifact, "
                    f"got {type(endpoint).__name__}"
                )
        else:  # target
            if not isinstance(endpoint, Interface):
                raise TypeError(
                    f"InterfaceProvided target must be an Interface, "
                    f"got {type(endpoint).__name__}"
                )


class InterfaceRequired(DeploymentRelationship):
    """A Node-or-Artifact -> Interface edge: the source *requires* the
    Interface.

    Endpoint rule: ``source`` is ``Node`` or ``Artifact``, ``target`` is
    ``Interface``.
    """

    def _check_endpoint(self, endpoint, role: str):
        if role == "source":
            if not isinstance(endpoint, (Node, Artifact)):
                raise TypeError(
                    f"InterfaceRequired source must be a Node or Artifact, "
                    f"got {type(endpoint).__name__}"
                )
        else:  # target
            if not isinstance(endpoint, Interface):
                raise TypeError(
                    f"InterfaceRequired target must be an Interface, "
                    f"got {type(endpoint).__name__}"
                )


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class DeploymentModel(Model):
    """The root of a UML Deployment model -- a first-class B-UML model.

    Like ``BPMNModel`` / ``ComponentModel``, this does not populate the
    inherited ``Model.elements`` set; it exposes its own typed accessors.

    Note: shares the bare class name ``DeploymentModel`` with the cloud-infra
    ``besser.BUML.metamodel.deployment.DeploymentModel``. Python namespacing
    resolves cleanly via the qualified import path; reader-facing collision is
    accepted (00- §8 #1).

    Args:
        name (str): The model name (free text; relaxed like ``DeploymentElement``).
        nodes (set[Node]): All nodes in this model (root-level + nested via
            ``Node.nested_nodes``).
        artifacts (set[Artifact]): Root-level artifacts (not nested inside any
            node). Nested artifacts are reachable via ``Node.nested_artifacts``.
        interfaces (set[Interface]): Interfaces in this model.
        relationships (set[DeploymentRelationship]): All relationships in this
            model.

    Attributes:
        nodes, artifacts, interfaces, relationships: as above.
    """

    def __init__(self, name: str, nodes: set = None, artifacts: set = None,
                 interfaces: set = None, relationships: set = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, metadata=metadata, timestamp=timestamp)
        self.__nodes: set = set()
        self.__artifacts: set = set()
        self.__interfaces: set = set()
        self.__relationships: set = set()
        self.nodes = nodes if nodes is not None else set()
        self.artifacts = artifacts if artifacts is not None else set()
        self.interfaces = interfaces if interfaces is not None else set()
        self.relationships = relationships if relationships is not None else set()

    @NamedElement.name.setter
    def name(self, name: str):
        """str: Set the model name. Relaxed like ``DeploymentElement``.

        Raises:
            TypeError: if name is neither a str nor None.
        """
        if name is None:
            name = ""
        if not isinstance(name, str):
            raise TypeError(
                f"DeploymentModel name must be a str or None, "
                f"got {type(name).__name__}"
            )
        self._NamedElement__name = name

    @property
    def nodes(self) -> set:
        """set[Node]: Get the (root-level) nodes."""
        return self.__nodes

    @nodes.setter
    def nodes(self, nodes: set):
        """set[Node]: Set the nodes.

        Raises:
            TypeError: if any element is not a Node.
        """
        self.__nodes = _checked_set(nodes, Node, "nodes")

    def add_node(self, node: "Node"):
        """Add a node.

        Raises:
            TypeError: if node is not a Node.
        """
        if not isinstance(node, Node):
            raise TypeError(f"node must be a Node, got {type(node).__name__}")
        self.__nodes.add(node)

    def remove_node(self, node: "Node"):
        """Remove a node."""
        self.__nodes.discard(node)

    @property
    def artifacts(self) -> set:
        """set[Artifact]: Get the root-level artifacts."""
        return self.__artifacts

    @artifacts.setter
    def artifacts(self, artifacts: set):
        """set[Artifact]: Set the root-level artifacts.

        Raises:
            TypeError: if any element is not an Artifact.
        """
        self.__artifacts = _checked_set(artifacts, Artifact, "artifacts")

    def add_artifact(self, artifact: "Artifact"):
        """Add a root-level artifact.

        Raises:
            TypeError: if artifact is not an Artifact.
        """
        if not isinstance(artifact, Artifact):
            raise TypeError(
                f"artifact must be an Artifact, got {type(artifact).__name__}"
            )
        self.__artifacts.add(artifact)

    def remove_artifact(self, artifact: "Artifact"):
        """Remove a root-level artifact."""
        self.__artifacts.discard(artifact)

    @property
    def interfaces(self) -> set:
        """set[Interface]: Get the interfaces."""
        return self.__interfaces

    @interfaces.setter
    def interfaces(self, interfaces: set):
        """set[Interface]: Set the interfaces.

        Raises:
            TypeError: if any element is not an Interface.
        """
        self.__interfaces = _checked_set(interfaces, Interface, "interfaces")

    def add_interface(self, interface: "Interface"):
        """Add an interface.

        Raises:
            TypeError: if interface is not an Interface.
        """
        if not isinstance(interface, Interface):
            raise TypeError(
                f"interface must be an Interface, got {type(interface).__name__}"
            )
        self.__interfaces.add(interface)

    def remove_interface(self, interface: "Interface"):
        """Remove an interface."""
        self.__interfaces.discard(interface)

    @property
    def relationships(self) -> set:
        """set[DeploymentRelationship]: Get the relationships."""
        return self.__relationships

    @relationships.setter
    def relationships(self, relationships: set):
        """set[DeploymentRelationship]: Set the relationships.

        Raises:
            TypeError: if any element is not a DeploymentRelationship.
        """
        self.__relationships = _checked_set(
            relationships, DeploymentRelationship, "relationships"
        )

    def add_relationship(self, relationship: "DeploymentRelationship"):
        """Add a relationship.

        Raises:
            TypeError: if relationship is not a DeploymentRelationship.
        """
        if not isinstance(relationship, DeploymentRelationship):
            raise TypeError(
                f"relationship must be a DeploymentRelationship, "
                f"got {type(relationship).__name__}"
            )
        self.__relationships.add(relationship)

    def remove_relationship(self, relationship: "DeploymentRelationship"):
        """Remove a relationship."""
        self.__relationships.discard(relationship)

    # --- derived accessors -------------------------------------------------

    def all_nodes(self) -> set:
        """set[Node]: Every node, expanded through ``Node.nested_nodes``."""
        result: set = set()

        def _collect(node: Node):
            result.add(node)
            for child in node.nested_nodes:
                _collect(child)

        for node in self.__nodes:
            _collect(node)
        return result

    def all_artifacts(self) -> set:
        """set[Artifact]: Every artifact, including those nested in any node."""
        result: set = set(self.__artifacts)
        for node in self.all_nodes():
            result |= node.nested_artifacts
        return result

    # --- validation --------------------------------------------------------

    def validate(self, raise_exception: bool = True) -> dict:
        """Validate the Deployment model according to its structural rules.

        Args:
            raise_exception (bool): If True, raise ValueError on errors.

        Returns:
            dict: ``{"success": bool, "errors": list[str], "warnings": list[str]}``.
        """
        errors: list = []
        warnings: list = []

        self._validate_endpoint_references(errors)
        self._validate_relationship_endpoint_types(errors)
        self._validate_multiplicity_bounds(errors)
        self._validate_parent_membership(errors)
        self._validate_unique_node_names(errors)
        self._warn_structural_smells(warnings)

        result = {"success": len(errors) == 0, "errors": errors, "warnings": warnings}
        if errors and raise_exception:
            raise ValueError("\n".join(errors))
        return result

    def _all_elements(self) -> set:
        """set[DeploymentElement]: Every node-like element a relationship may
        reference."""
        return self.all_nodes() | self.all_artifacts() | self.__interfaces

    def _validate_endpoint_references(self, errors: list):
        """E1: every relationship's source / target is reachable in this model."""
        universe = self._all_elements()
        for rel in self.__relationships:
            for role, endpoint in (("source", rel.source), ("target", rel.target)):
                if endpoint is None:
                    errors.append(
                        f"{type(rel).__name__} '{rel.name}' has no {role}."
                    )
                elif endpoint not in universe:
                    errors.append(
                        f"{type(rel).__name__} '{rel.name}' {role} "
                        f"'{endpoint.name}' ({type(endpoint).__name__}) is "
                        f"not present in the model."
                    )

    def _validate_relationship_endpoint_types(self, errors: list):
        """E2-E5: per-class endpoint rules (second line of defence over setters)."""
        for rel in self.__relationships:
            if isinstance(rel, DeploymentRelation):
                if not (isinstance(rel.source, Artifact)
                        and isinstance(rel.target, Node)):
                    errors.append(
                        f"DeploymentRelation '{rel.name}' must be "
                        f"Artifact -> Node."
                    )
            elif isinstance(rel, CommunicationPath):
                if not (isinstance(rel.source, Node)
                        and isinstance(rel.target, Node)):
                    errors.append(
                        f"CommunicationPath '{rel.name}' must connect two Nodes."
                    )
            elif isinstance(rel, InterfaceProvided):
                if not (isinstance(rel.source, (Node, Artifact))
                        and isinstance(rel.target, Interface)):
                    errors.append(
                        f"InterfaceProvided '{rel.name}' must be "
                        f"Node|Artifact -> Interface."
                    )
            elif isinstance(rel, InterfaceRequired):
                if not (isinstance(rel.source, (Node, Artifact))
                        and isinstance(rel.target, Interface)):
                    errors.append(
                        f"InterfaceRequired '{rel.name}' must be "
                        f"Node|Artifact -> Interface."
                    )

    def _validate_multiplicity_bounds(self, errors: list):
        """E6: DeploymentRelation.multiplicity must have min <= max (or max
        is UNLIMITED). Multiplicity's own setters already enforce min >= 0
        and max > 0 construction-time, but a manual mutation could break the
        invariant; double-check here."""
        for rel in self.__relationships:
            if isinstance(rel, DeploymentRelation):
                mult = rel.multiplicity
                if mult.min > mult.max:
                    errors.append(
                        f"DeploymentRelation '{rel.name}' has invalid "
                        f"multiplicity: min ({mult.min}) > max ({mult.max})."
                    )

    def _validate_parent_membership(self, errors: list):
        """E7: ``Artifact.parent`` / ``Node.parent``, if set, are Nodes in
        the model (or the model's transitive node closure)."""
        node_universe = self.all_nodes()
        for artifact in self.all_artifacts():
            parent = artifact.parent
            if parent is not None and parent not in node_universe:
                errors.append(
                    f"Artifact '{artifact.name}' has parent Node "
                    f"'{parent.name}' which is not in the model."
                )
        for node in self.all_nodes():
            parent = node.parent
            if parent is not None and parent not in node_universe:
                errors.append(
                    f"Node '{node.name}' has parent Node "
                    f"'{parent.name}' which is not in the model."
                )

    def _validate_unique_node_names(self, errors: list):
        """E8: Node names within a model are unique (skipping empty names)."""
        seen: dict = {}
        for node in self.all_nodes():
            name = node.name
            if name == "":
                continue
            if name in seen:
                errors.append(
                    f"Duplicate node name '{name}'."
                )
            else:
                seen[name] = node

    def _warn_structural_smells(self, warnings: list):
        """W1-W4: non-blocking structural smells."""
        # Build incoming-DeploymentRelation index for nodes.
        nodes_with_artifacts: set = set()
        for rel in self.__relationships:
            if isinstance(rel, DeploymentRelation) and rel.target is not None:
                nodes_with_artifacts.add(rel.target)

        for node in self.all_nodes():
            # W1 -- node with no nested artifacts and no incoming DeploymentRelation.
            if not node.nested_artifacts and node not in nodes_with_artifacts:
                warnings.append(
                    f"Node '{node.name}' is empty (no nested artifacts and "
                    f"no incoming DeploymentRelation)."
                )
            # W4 -- GENERIC node with no stereotypes (under-typed).
            if node.kind is NodeKind.GENERIC and not node.stereotypes:
                warnings.append(
                    f"Node '{node.name}' has kind=GENERIC and no stereotypes; "
                    f"consider DEVICE or EXECUTION_ENVIRONMENT."
                )

        # W2 -- artifact with empty `manifests`.
        for artifact in self.all_artifacts():
            if not artifact.manifests:
                warnings.append(
                    f"Artifact '{artifact.name}' has empty manifests "
                    f"(manifests no Component)."
                )

    def __repr__(self):
        return (f"DeploymentModel(name='{self.name}', "
                f"nodes={len(self.nodes)}, artifacts={len(self.artifacts)}, "
                f"interfaces={len(self.interfaces)}, "
                f"relationships={len(self.relationships)})")
