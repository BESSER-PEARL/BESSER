"""UML Component metamodel for B-UML -- vanilla UML 2.5 base.

A first-class B-UML model for UML Component diagrams, alongside ``structural`` /
``state_machine`` / ``gui`` / ``bpmn``. Implements the design in
``.claude/component-deployment/01-component-deployment-design.md`` (reviewed and
locked 2026-05-18; the base/agentic split is ``04-...``, 2026-05-20).

This module is **pure UML 2.5 Component modelling**. The agentic-swarm
stereotype profile is a separate extension in ``agentic.py`` (decision D11,
revised by ``04-``: the profile was originally baked in here; it now lives in
its own module, mirroring ``bpmn/bpmn.py`` + ``bpmn/agentic.py``). The package
``__init__.py`` re-exports both, so ``from besser.BUML.metamodel.uml_component
import *`` is unchanged for downstream callers.

Hierarchy::

    Element -> NamedElement -> ComponentElement -> {Component, Interface,
                                                    ComponentRelationship}
                            -> Model -> ComponentModel

    Component <|-- Subsystem (a Component that is also a container)

    ComponentRelationship <|-- InterfaceProvided
                          <|-- InterfaceRequired
                          <|-- ComponentDependency

Growth path (01-... §3.5) -- constructs designed but intentionally NOT implemented yet:

* UML ``Realization`` as a distinct relationship class (currently absorbed into
  ``Component.realizes: List[str]`` of structural ``Class`` IDs).
* ``«autoscale»`` policy attributes (sibling to NR-6 in the requirements review).

``Locality`` (below) is a BESSER general profile addition (NR-5) -- see its
docstring. Everything else in this module is vanilla UML 2.5.
"""

from enum import Enum
from typing import List, Optional

from besser.BUML.metamodel.structural import Model, NamedElement


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Locality(Enum):
    """Where a Component / Artifact / Node is hosted.

    A BESSER **general profile addition** (NR-5 of the requirements review) --
    *not* part of UML 2.5.1: the spec defines no ``«external»`` standard
    stereotype and no locality concept (verified against uml-2-5-1-formal-
    17-12-05 -- Clause 22 Standard Profile, Clause 19 Deployments). UML 2.5.1
    Clause 19 explicitly anticipates profiles adding deployment stereotypes
    (e.g. ``«concurrencyMode»``); ``Locality`` is exactly such a profile
    stereotype. It is **non-agentic** -- it classifies any deployment -- so it
    stays in the base metamodel rather than the ``agentic.py`` extension, and
    is shared (defined once per metamodel) with ``uml_deployment``.

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
# Base element (decision D9 -- relaxed name, opaque layout passthrough)
# ---------------------------------------------------------------------------

class ComponentElement(NamedElement):
    """Base class for every UML Component-diagram abstract-syntax element.

    Relaxes ``NamedElement.name`` (decision D9, mirrors BPMN D5): a Component-diagram
    label is free text and may contain spaces (``"Code Tester"``), colons
    (``"merge:approve"``), or be empty (a freshly-dropped element). Also carries
    ``layout`` -- an opaque diagram-interchange passthrough the metamodel never
    interprets (decision D10); only the converters read it. The opaque ``layout``
    dict is also where converters stash the stable WME element ID so cross-diagram
    string-ID references survive round-trips.

    Args:
        name (str): The element label. Empty allowed; ``None`` is coerced to ``""``.
        stereotypes (List[str]): Free-form stereotype strings (decision D3 long
            tail). Defaults to ``[]``.
        layout (dict): Opaque DI data. Stored untouched.
        metadata (Metadata): Inherited from NamedElement.
        timestamp (datetime): Inherited from NamedElement.

    Attributes:
        name (str): The element label (free text; may be empty).
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
        """str: Set the Component-element label. Permits empty / whitespace /
        spaces / punctuation; ``None`` -> ``""``.

        Raises:
            TypeError: if name is neither a str nor None.
        """
        if name is None:
            name = ""
        if not isinstance(name, str):
            raise TypeError(
                f"ComponentElement name must be a str or None, got {type(name).__name__}"
            )
        # Bypass NamedElement's strict identifier validation by writing its private slot.
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
# Components (Subsystem subclasses Component; the agentic subclasses
# AgenticComponent / Skill / Tool live in agentic.py)
# ---------------------------------------------------------------------------

class Component(ComponentElement):
    """A UML Component -- the canonical node in a Component diagram.

    The base UML 2.5 Component. ``Subsystem`` subclasses it (a Component that is
    also a container). The agentic-swarm subclasses ``AgenticComponent`` /
    ``Skill`` / ``Tool`` live in ``agentic.py``.

    Args:
        name (str): The component label.
        locality (Locality): Where the component is hosted (NR-5). ``LOCAL`` by
            default.
        realizes (List[str]): Cross-diagram IDs of structural ``Class`` es this
            component realizes (UML Realization). Default ``[]``.
        stereotypes, layout, metadata, timestamp: Inherited from ComponentElement.

    Attributes:
        locality (Locality): Hosting locality.
        realizes (List[str]): Cross-diagram Class IDs.
        parent (Subsystem | None): The containing Subsystem (set by the
            ``Subsystem`` container, or None if at model root).
    """

    def __init__(self, name: str = "", locality: Locality = None,
                 realizes: List[str] = None,
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.locality = locality if locality is not None else Locality.LOCAL
        self.realizes = realizes if realizes is not None else []
        self.__parent: Optional["Subsystem"] = None

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
    def realizes(self) -> List[str]:
        """List[str]: Cross-diagram IDs of structural Classes this component realizes."""
        return self.__realizes

    @realizes.setter
    def realizes(self, realizes: List[str]):
        """List[str]: Set the cross-diagram Class IDs.

        Raises:
            TypeError: if not a list of str.
        """
        self.__realizes = _checked_str_list(realizes, "realizes")

    @property
    def parent(self) -> Optional["Subsystem"]:
        """Subsystem: The containing Subsystem, or None if at model root."""
        return self.__parent

    @parent.setter
    def parent(self, parent: Optional["Subsystem"]):
        """Subsystem: Set the containing Subsystem.

        Raises:
            TypeError: if parent is neither a Subsystem nor None.
        """
        if parent is not None and not isinstance(parent, Subsystem):
            raise TypeError(
                f"parent must be a Subsystem or None, got {type(parent).__name__}"
            )
        self.__parent = parent


class Subsystem(Component):
    """A UML Subsystem -- a Component that is *also* a flow-element container.

    Apollon emits Subsystem as a separate element type (with stereotype
    ``«subsystem»``). UML 2.5 deprecated the dedicated Subsystem metaclass to a
    stereotype on Component, but we keep the subclass for round-trip clarity
    with WME's wire shape.

    Attributes:
        children (set[Component]): Child components nested in this subsystem.
            Each child's ``parent`` back-reference is maintained automatically.
    """

    def __init__(self, name: str = "", children: set = None,
                 locality: Locality = None, realizes: List[str] = None,
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        super().__init__(name=name, locality=locality, realizes=realizes,
                         stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.__children: set = set()
        self.children = children if children is not None else set()

    @property
    def children(self) -> set:
        """set[Component]: Get the child components."""
        return self.__children

    @children.setter
    def children(self, children: set):
        """set[Component]: Set the child components; maintains each child's
        ``parent`` back-reference.

        Raises:
            TypeError: if any element is not a Component.
        """
        children = set(children)
        for child in children:
            if not isinstance(child, Component):
                raise TypeError(
                    f"children must contain Component instances, "
                    f"got {type(child).__name__}"
                )
        for existing in self.__children:
            if existing.parent is self:
                existing.parent = None
        for child in children:
            child.parent = self
        self.__children = children

    def add_child(self, child: "Component"):
        """Add a child component; sets its ``parent``.

        Raises:
            TypeError: if child is not a Component.
        """
        if not isinstance(child, Component):
            raise TypeError(f"child must be a Component, got {type(child).__name__}")
        child.parent = self
        self.__children.add(child)

    def remove_child(self, child: "Component"):
        """Remove a child component; clears its ``parent``."""
        if child in self.__children:
            if child.parent is self:
                child.parent = None
            self.__children.discard(child)


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class Interface(ComponentElement):
    """A UML provided / required Interface -- a standalone named element
    referenced from ``InterfaceProvided`` / ``InterfaceRequired`` relationships
    (decision D11 -- separate class, not inlined on Component, matching UML 2.5
    and Apollon's wire shape).
    """


# ---------------------------------------------------------------------------
# Relationships (decision D11 -- abstract base with endpoint type-check seam)
# ---------------------------------------------------------------------------

class ComponentRelationship(ComponentElement):
    """Abstract base for the four Component-diagram relationship kinds.

    ``source`` and ``target`` are required (non-None). Each concrete subclass
    overrides ``_check_endpoint`` with its own endpoint type rule; the model-wide
    validator (``ComponentModel.validate()``) re-checks agentic-state rules at
    validate-time.

    Attributes:
        source (ComponentElement): The relationship's source element.
        target (ComponentElement): The relationship's target element.
    """

    def __init__(self, source, target, name: str = "",
                 stereotypes: List[str] = None, layout: dict = None,
                 metadata=None, timestamp=None):
        if type(self) is ComponentRelationship:
            raise TypeError(
                "ComponentRelationship is abstract and cannot be instantiated directly"
            )
        super().__init__(name=name, stereotypes=stereotypes, layout=layout,
                         metadata=metadata, timestamp=timestamp)
        self.__source = None
        self.__target = None
        self.source = source
        self.target = target

    @property
    def source(self):
        """ComponentElement: Get the source element."""
        return self.__source

    @source.setter
    def source(self, source):
        """ComponentElement: Set the source element.

        Raises:
            TypeError: if source is not an acceptable endpoint for this relationship.
        """
        self._check_endpoint(source, "source")
        self.__source = source

    @property
    def target(self):
        """ComponentElement: Get the target element."""
        return self.__target

    @target.setter
    def target(self, target):
        """ComponentElement: Set the target element.

        Raises:
            TypeError: if target is not an acceptable endpoint for this relationship.
        """
        self._check_endpoint(target, "target")
        self.__target = target

    def _check_endpoint(self, endpoint, role: str):
        """Type-check a single endpoint. Base requires any ``ComponentElement``;
        subclasses tighten.

        Raises:
            TypeError: if endpoint is not acceptable.
        """
        if not isinstance(endpoint, ComponentElement):
            raise TypeError(
                f"{type(self).__name__} {role} must be a ComponentElement, "
                f"got {type(endpoint).__name__}"
            )

    def __repr__(self):
        source_name = self.source.name if self.source is not None else None
        target_name = self.target.name if self.target is not None else None
        return (f"{type(self).__name__}(name='{self.name}', "
                f"source='{source_name}', target='{target_name}')")


class InterfaceProvided(ComponentRelationship):
    """A Component -> Interface edge: the Component *provides* the Interface
    (UML 2.5 BehavioredClassifier -> realizedInterface, lollipop notation)."""

    def _check_endpoint(self, endpoint, role: str):
        if role == "source":
            if not isinstance(endpoint, Component):
                raise TypeError(
                    f"InterfaceProvided source must be a Component, "
                    f"got {type(endpoint).__name__}"
                )
        else:  # target
            if not isinstance(endpoint, Interface):
                raise TypeError(
                    f"InterfaceProvided target must be an Interface, "
                    f"got {type(endpoint).__name__}"
                )


class InterfaceRequired(ComponentRelationship):
    """A Component -> Interface edge: the Component *requires* the Interface
    (UML 2.5 Component -> requiredInterface, socket notation)."""

    def _check_endpoint(self, endpoint, role: str):
        if role == "source":
            if not isinstance(endpoint, Component):
                raise TypeError(
                    f"InterfaceRequired source must be a Component, "
                    f"got {type(endpoint).__name__}"
                )
        else:  # target
            if not isinstance(endpoint, Interface):
                raise TypeError(
                    f"InterfaceRequired target must be an Interface, "
                    f"got {type(endpoint).__name__}"
                )


class ComponentDependency(ComponentRelationship):
    """A plain UML dependency between two Components (e.g. ``«use»``,
    ``«import»``). Carries an arbitrary ``stereotypes`` list inherited from
    ``ComponentElement``.

    Endpoint rule: both source and target are ``Component`` s. Tighter than the
    abstract base's ``ComponentElement`` to exclude Interfaces and Permissions.
    """

    def _check_endpoint(self, endpoint, role: str):
        if not isinstance(endpoint, Component):
            raise TypeError(
                f"{type(self).__name__} {role} must be a Component, "
                f"got {type(endpoint).__name__}"
            )


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class ComponentModel(Model):
    """The root of a UML Component model -- a first-class B-UML model.

    Pure UML 2.5: holds components, interfaces, and relationships. The agentic
    extension subclass ``AgenticComponentModel`` (in ``agentic.py``) adds the
    ``permissions`` collection and the agentic validation rules.

    Like ``BPMNModel``, ``ComponentModel`` does not populate the inherited
    ``Model.elements`` set; it exposes its own typed accessors.

    Args:
        name (str): The model name (free text; relaxed like ``ComponentElement``).
        components (set[Component]): All components (incl. Subsystems) in this
            model. Subsystem children also count here.
        interfaces (set[Interface]): Interfaces in this model.
        relationships (set[ComponentRelationship]): All relationships in this
            model (``InterfaceProvided`` / ``InterfaceRequired`` / plain
            ``ComponentDependency``).

    Attributes:
        components, interfaces, relationships: as above.
    """

    def __init__(self, name: str, components: set = None,
                 interfaces: set = None,
                 relationships: set = None, metadata=None, timestamp=None):
        super().__init__(name=name, metadata=metadata, timestamp=timestamp)
        self.__components: set = set()
        self.__interfaces: set = set()
        self.__relationships: set = set()
        self.components = components if components is not None else set()
        self.interfaces = interfaces if interfaces is not None else set()
        self.relationships = relationships if relationships is not None else set()

    @NamedElement.name.setter
    def name(self, name: str):
        """str: Set the model name. Relaxed like ``ComponentElement``.

        Raises:
            TypeError: if name is neither a str nor None.
        """
        if name is None:
            name = ""
        if not isinstance(name, str):
            raise TypeError(
                f"ComponentModel name must be a str or None, got {type(name).__name__}"
            )
        self._NamedElement__name = name

    @property
    def components(self) -> set:
        """set[Component]: Get the components in this model."""
        return self.__components

    @components.setter
    def components(self, components: set):
        """set[Component]: Set the components in this model.

        Raises:
            TypeError: if any element is not a Component.
        """
        self.__components = _checked_set(components, Component, "components")

    def add_component(self, component: "Component"):
        """Add a component.

        Raises:
            TypeError: if component is not a Component.
        """
        if not isinstance(component, Component):
            raise TypeError(
                f"component must be a Component, got {type(component).__name__}"
            )
        self.__components.add(component)

    def remove_component(self, component: "Component"):
        """Remove a component."""
        self.__components.discard(component)

    @property
    def interfaces(self) -> set:
        """set[Interface]: Get the interfaces in this model."""
        return self.__interfaces

    @interfaces.setter
    def interfaces(self, interfaces: set):
        """set[Interface]: Set the interfaces in this model.

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
        """set[ComponentRelationship]: Get the relationships in this model."""
        return self.__relationships

    @relationships.setter
    def relationships(self, relationships: set):
        """set[ComponentRelationship]: Set the relationships in this model.

        Raises:
            TypeError: if any element is not a ComponentRelationship.
        """
        self.__relationships = _checked_set(
            relationships, ComponentRelationship, "relationships"
        )

    def add_relationship(self, relationship: "ComponentRelationship"):
        """Add a relationship.

        Raises:
            TypeError: if relationship is not a ComponentRelationship.
        """
        if not isinstance(relationship, ComponentRelationship):
            raise TypeError(
                f"relationship must be a ComponentRelationship, "
                f"got {type(relationship).__name__}"
            )
        self.__relationships.add(relationship)

    def remove_relationship(self, relationship: "ComponentRelationship"):
        """Remove a relationship."""
        self.__relationships.discard(relationship)

    # --- derived accessors -------------------------------------------------

    def all_components(self) -> set:
        """set[Component]: Every component, expanded through Subsystem nesting."""
        result: set = set()

        def _collect(component: Component):
            result.add(component)
            if isinstance(component, Subsystem):
                for child in component.children:
                    _collect(child)

        for component in self.__components:
            _collect(component)
        return result

    # --- validation --------------------------------------------------------

    def validate(self, raise_exception: bool = True) -> dict:
        """Validate the Component model according to its structural rules.

        Args:
            raise_exception (bool): If True, raise ValueError when validation fails.

        Returns:
            dict: ``{"success": bool, "errors": list[str], "warnings": list[str]}``.
        """
        errors: list = []
        warnings: list = []

        self._validate_endpoint_references(errors)
        self._validate_relationship_endpoint_types(errors)
        self._validate_unique_component_names(errors)
        self._validate_subsystem_membership(errors)
        self._warn_structural_smells(warnings)

        result = {"success": len(errors) == 0, "errors": errors, "warnings": warnings}
        if errors and raise_exception:
            raise ValueError("\n".join(errors))
        return result

    def _all_elements(self) -> set:
        """set[ComponentElement]: Every element a relationship may reference.

        ``AgenticComponentModel`` overrides this to also include permissions.
        """
        return self.all_components() | self.__interfaces

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
        """E2-E4: per-kind endpoint type rules for the base relationship kinds.

        Construction-time setters already enforce these; validate() is the
        second line of defence against a model assembled by wiring ``source`` /
        ``target`` through the private slots. ``AgenticEdge`` (a
        ``ComponentDependency`` subclass) is skipped here -- ``type(rel) is
        ComponentDependency`` excludes it -- and re-checked by
        ``AgenticComponentModel``.
        """
        for rel in self.__relationships:
            if isinstance(rel, InterfaceProvided):
                if not (isinstance(rel.source, Component)
                        and isinstance(rel.target, Interface)):
                    errors.append(
                        f"InterfaceProvided '{rel.name}' must be "
                        f"Component -> Interface."
                    )
            elif isinstance(rel, InterfaceRequired):
                if not (isinstance(rel.source, Component)
                        and isinstance(rel.target, Interface)):
                    errors.append(
                        f"InterfaceRequired '{rel.name}' must be "
                        f"Component -> Interface."
                    )
            elif type(rel) is ComponentDependency:
                if not (isinstance(rel.source, Component)
                        and isinstance(rel.target, Component)):
                    errors.append(
                        f"ComponentDependency '{rel.name}' must connect "
                        f"two Components."
                    )

    def _validate_unique_component_names(self, errors: list):
        """E11: Component names within a model are unique."""
        seen: dict = {}
        for component in self.all_components():
            name = component.name
            if name == "":
                continue
            if name in seen:
                errors.append(
                    f"Duplicate component name '{name}': {type(seen[name]).__name__} "
                    f"and {type(component).__name__}."
                )
            else:
                seen[name] = component

    def _validate_subsystem_membership(self, errors: list):
        """E13: ``Component.parent``, if set, points at a Subsystem that is in
        the model's components."""
        for component in self.all_components():
            parent = component.parent
            if parent is not None and parent not in self.all_components():
                errors.append(
                    f"Component '{component.name}' has parent Subsystem "
                    f"'{parent.name}' which is not in the model."
                )

    def _warn_structural_smells(self, warnings: list):
        """W7: non-blocking structural smell (base UML).

        W7: Interface with no incoming InterfaceProvided / InterfaceRequired.

        The agentic smells (W1 agent with no «has»/«uses», W2 agent in no BPMN
        process, W3 permissions on a non-agent-to-agent edge) live in
        ``AgenticComponentModel`` in ``agentic.py``.
        """
        # W7 -- orphan interface.
        used_interfaces: set = set()
        for rel in self.__relationships:
            if isinstance(rel, (InterfaceProvided, InterfaceRequired)):
                used_interfaces.add(rel.target)
        for interface in self.__interfaces:
            if interface not in used_interfaces:
                warnings.append(
                    f"Interface '{interface.name}' has no incoming "
                    f"InterfaceProvided / InterfaceRequired (orphan interface)."
                )

    def __repr__(self):
        return (f"ComponentModel(name='{self.name}', "
                f"components={len(self.components)}, "
                f"interfaces={len(self.interfaces)}, "
                f"relationships={len(self.relationships)})")
