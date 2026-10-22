"""Metaclasses for the structural events of incremental integrity checking.

Each concrete class is one event type from section 4.1 of Cabot's thesis.
``Object`` replaces the thesis name ``ET`` and ``Link`` replaces ``RT``.
An object event is linked to the :class:`~besser.BUML.metamodel.structural.structural.Class`
whose instances change. A link event is linked to the
:class:`~besser.BUML.metamodel.structural.structural.Association` whose links
change. An attribute update is linked to the
:class:`~besser.BUML.metamodel.structural.structural.Property` whose value changes.

:class:`ViolatingEvent` is the association class between a
:class:`~besser.BUML.metamodel.structural.structural.Constraint` and a
:class:`StructuralEvent`. Each instance links exactly one constraint to
exactly one structural event.
"""

from __future__ import annotations

from typing import Any, Optional
from weakref import WeakKeyDictionary

from besser.BUML.metamodel.structural.structural import (
    Association,
    Class,
    Constraint,
    DomainModel,
    Element,
    Property,
)


class StructuralEvent(Element):
    """An elementary change in the population of a class or an association.

    Subclasses name the kind of change and the B-UML element the change
    applies to. They are event types: two insertions of different objects
    into the same class are the same event type.
    """

    def key(self) -> tuple:
        """Identity of this event type, independent of the Python object id."""
        raise NotImplementedError

    def kind(self) -> str:
        """Name of the event type, for example ``InsertObject``."""
        return type(self).__name__


class ObjectEvent(StructuralEvent):
    """A structural event linked to the class whose objects change.

    Args:
        class_ (Class): The class. Inserting, deleting or reclassifying an
            instance of this class is the event.
    """

    def __init__(self, class_: Class, timestamp=None, is_derived: bool = False, uncertainty: float = 0.0):
        super().__init__(timestamp, is_derived, uncertainty)
        self.class_: Class = class_

    @property
    def class_(self) -> Class:
        """Class: The class this object event is about."""
        return self.__class_ref

    @class_.setter
    def class_(self, class_: Class):
        """Class: Set the class this object event is about.

        Raises:
            TypeError: if ``class_`` is not a :class:`Class`.
        """
        if not isinstance(class_, Class):
            raise TypeError(f"class_ must be a Class, got {type(class_).__name__}")
        self.__class_ref = class_

    def key(self) -> tuple:
        return (self.kind(), self.class_)

    def __repr__(self):
        return f"{self.kind()}({self.class_.name})"


class InsertObject(ObjectEvent):
    """Insertion of a new object into ``class_``.

    The new object may have its attributes initialized. It does not participate
    in any association yet. Creating an object of a subclass also inserts it
    into each superclass; this event records the class named by ``class_``.
    """


class DeleteObject(ObjectEvent):
    """Deletion of an object from ``class_``.

    Deleting an object also deletes it from its superclasses and subclasses.
    This event records the class named by ``class_``.
    """


class SpecializeObject(ObjectEvent):
    """Reclassification of an existing object into ``class_``.

    The object already belongs to a superclass of ``class_`` and gains
    ``class_`` as its new class. This is the thesis event ``SpecializeET``.
    """


class GeneralizeObject(ObjectEvent):
    """Reclassification of an object away from a subclass, up to ``class_``.

    The object leaves a direct subclass of ``class_`` and is afterwards only
    an instance of ``class_`` (and its superclasses). This is the thesis event
    ``GeneralizeET``.
    """


class UpdateAttribute(StructuralEvent):
    """Replacement of the value of one attribute of an object.

    The event is linked to the attribute. The class of the object is the
    attribute's owner.

    Args:
        attribute (Property): The attribute whose value changes. It is owned
            by a class, not by an association.
    """

    def __init__(self, attribute: Property, timestamp=None, is_derived: bool = False, uncertainty: float = 0.0):
        super().__init__(timestamp, is_derived, uncertainty)
        self.attribute: Property = attribute

    @property
    def attribute(self) -> Property:
        """Property: The attribute whose value changes."""
        return self.__attribute

    @attribute.setter
    def attribute(self, attribute: Property):
        """Property: Set the attribute whose value changes.

        Raises:
            TypeError: if ``attribute`` is not a :class:`Property`.
        """
        if not isinstance(attribute, Property):
            raise TypeError(f"attribute must be a Property, got {type(attribute).__name__}")
        self.__attribute = attribute

    def key(self) -> tuple:
        return (self.kind(), self.attribute)

    def __repr__(self):
        owner = self.attribute.owner
        owner_name = owner.name if owner is not None else None
        return f"UpdateAttribute({self.attribute.name}, {owner_name})"


class LinkEvent(StructuralEvent):
    """A structural event linked to the association whose links change.

    Args:
        association (Association): The association. Inserting or deleting a
            link of this association is the event.
    """

    def __init__(self, association: Association, timestamp=None, is_derived: bool = False,
                 uncertainty: float = 0.0):
        super().__init__(timestamp, is_derived, uncertainty)
        self.association: Association = association

    @property
    def association(self) -> Association:
        """Association: The association this link event is about."""
        return self.__association

    @association.setter
    def association(self, association: Association):
        """Association: Set the association this link event is about.

        Raises:
            TypeError: if ``association`` is not an :class:`Association`.
        """
        if not isinstance(association, Association):
            raise TypeError(f"association must be an Association, got {type(association).__name__}")
        self.__association = association

    def key(self) -> tuple:
        return (self.kind(), self.association)

    def __repr__(self):
        return f"{self.kind()}({self.association.name})"


class InsertLink(LinkEvent):
    """Insertion of a new link into ``association``.

    For an association class, the link is the association facet of the new
    link object. The object facet is a separate :class:`InsertObject`.
    """


class DeleteLink(LinkEvent):
    """Deletion of a link from ``association``."""


# Association-class instances currently attached to each constraint.
# The constraint class itself is unchanged; this is the inverse of the
# ``constraint`` end, held here so a constraint can be asked for its links.
_LINKS: WeakKeyDictionary = WeakKeyDictionary()


class ViolatingEvent(Element):
    """Association class between a constraint and one structural event.

    Each instance is one link and therefore has exactly one
    :attr:`constraint` and exactly one :attr:`structural_event`.
    ``approximated`` is true when the event was added because part of the
    expression is outside the fragment the analysis handles exactly. Such an
    event may be a false positive. Dropping it could hide a real violation,
    so it is kept.

    The constraint text those events were computed from is
    ``constraint.expression``. It is not stored again on the link.

    Creating a link registers it. :func:`violating_events` returns the links
    registered for a constraint, and :func:`set_violating_events` replaces
    that set. The ends of an existing link stay in place: a link is never
    left without its constraint or its event.

    Args:
        constraint (Constraint): The invariant this event may violate.
        structural_event (StructuralEvent): The event type.
        approximated (bool): True when the event comes from over-approximation.
    """

    def __init__(self, constraint: Constraint, structural_event: StructuralEvent,
                 approximated: bool = False, timestamp=None, is_derived: bool = False,
                 uncertainty: float = 0.0):
        super().__init__(timestamp, is_derived, uncertainty)
        self.__constraint: Optional[Constraint] = None
        self.structural_event: StructuralEvent = structural_event
        self.approximated: bool = approximated
        self.constraint = constraint

    @property
    def constraint(self) -> Constraint:
        """Constraint: The one constraint this link connects."""
        return self.__constraint

    @constraint.setter
    def constraint(self, constraint: Constraint):
        """Constraint: Point this link at ``constraint``.

        Raises:
            TypeError: if ``constraint`` is not a :class:`Constraint`.
        """
        if not isinstance(constraint, Constraint):
            raise TypeError(f"constraint must be a Constraint, got {type(constraint).__name__}")
        current = self.__constraint
        if current is constraint:
            _register(self)
            return
        if current is not None:
            _unregister(self, current)
        self.__constraint = constraint
        _register(self)

    @property
    def structural_event(self) -> StructuralEvent:
        """StructuralEvent: The one event type this link connects."""
        return self.__structural_event

    @structural_event.setter
    def structural_event(self, structural_event: StructuralEvent):
        """StructuralEvent: Set the event type.

        Raises:
            TypeError: if ``structural_event`` is not a :class:`StructuralEvent`.
        """
        if not isinstance(structural_event, StructuralEvent):
            raise TypeError(
                f"structural_event must be a StructuralEvent, got {type(structural_event).__name__}"
            )
        self.__structural_event = structural_event

    @property
    def approximated(self) -> bool:
        """bool: True when this link was produced by over-approximation."""
        return self.__approximated

    @approximated.setter
    def approximated(self, approximated: bool):
        """bool: Set whether this link was produced by over-approximation."""
        if not isinstance(approximated, bool):
            raise TypeError("approximated must be a bool")
        self.__approximated = approximated

    def __repr__(self):
        name = self.constraint.name if self.constraint is not None else None
        return f"ViolatingEvent({name}, {self.structural_event}, approximated={self.approximated})"


def violating_events(constraint: Constraint) -> set[ViolatingEvent]:
    """The :class:`ViolatingEvent` links registered for ``constraint``."""
    if constraint is None:
        return set()
    return set(_LINKS.get(constraint, ()))


def set_violating_events(constraint: Constraint, links: set[ViolatingEvent]) -> None:
    """Replace the links registered for ``constraint``.

    Every link must already connect ``constraint`` to one structural event.
    Links that were registered and are absent from ``links`` leave the set.
    Their own ends are unchanged.

    Raises:
        TypeError: if ``links`` is not a set, or an item is not a
            :class:`ViolatingEvent`.
        ValueError: if a link connects a different constraint.
    """
    if not isinstance(links, set):
        raise TypeError("links must be a set")
    for item in links:
        if not isinstance(item, ViolatingEvent):
            raise TypeError(f"links must contain ViolatingEvent instances, got {type(item).__name__}")
        if item.constraint is not constraint:
            raise ValueError("each ViolatingEvent must link this constraint")
    for old in list(violating_events(constraint)):
        if old not in links:
            _unregister(old, constraint)
    for item in links:
        _register(item)


def _register(link: ViolatingEvent) -> None:
    constraint = link.constraint
    if constraint is None:
        return
    bucket = _LINKS.get(constraint)
    if bucket is None:
        bucket = set()
        _LINKS[constraint] = bucket
    bucket.add(link)


def _unregister(link: ViolatingEvent, constraint: Constraint) -> None:
    bucket = _LINKS.get(constraint)
    if bucket is None:
        return
    bucket.discard(link)
    if not bucket:
        _LINKS.pop(constraint, None)


_EVENT_CLASSES = {
    "InsertObject": InsertObject,
    "DeleteObject": DeleteObject,
    "SpecializeObject": SpecializeObject,
    "GeneralizeObject": GeneralizeObject,
    "UpdateAttribute": UpdateAttribute,
    "InsertLink": InsertLink,
    "DeleteLink": DeleteLink,
}


def event_payloads(constraint: Constraint) -> list[dict[str, Any]]:
    """JSON-ready events linked to ``constraint``, in a stable order."""
    links = sorted(violating_events(constraint), key=_link_sort_key)
    return [event_to_dict(link) for link in links]


def _link_sort_key(link: ViolatingEvent) -> tuple:
    event = link.structural_event
    class_ = getattr(event, "class_", None)
    association = getattr(event, "association", None)
    attribute = getattr(event, "attribute", None)
    owner = getattr(attribute, "owner", None) if attribute is not None else None
    return (
        event.kind(),
        getattr(class_, "name", "") or getattr(association, "name", "") or getattr(owner, "name", "") or "",
        getattr(attribute, "name", "") or "",
        link.approximated,
    )


def event_to_dict(link: ViolatingEvent) -> dict[str, Any]:
    """Serialize one violating-event link to a JSON-ready dictionary.

    Class, attribute and association ends are stored by name. The dictionary
    is what the editor keeps on an OCL constraint box.
    """
    event = link.structural_event
    payload: dict[str, Any] = {
        "constraint": link.constraint.name if link.constraint is not None else None,
        "kind": event.kind(),
        "approximated": link.approximated,
    }
    if isinstance(event, UpdateAttribute):
        payload["attribute"] = event.attribute.name
        owner = event.attribute.owner
        if owner is not None:
            payload["class"] = owner.name
    elif isinstance(event, ObjectEvent):
        payload["class"] = event.class_.name
    elif isinstance(event, LinkEvent):
        payload["association"] = event.association.name
    return payload


def event_from_dict(payload: dict[str, Any], constraint: Constraint,
                    model: DomainModel) -> Optional[ViolatingEvent]:
    """Rebuild one :class:`ViolatingEvent` from :func:`event_to_dict`.

    Names are resolved against ``model``. Returns None when the payload names
    an element that is no longer there; the caller records that as a warning
    rather than failing the whole diagram load.

    Args:
        payload: One serialized event.
        constraint: The invariant the link belongs to.
        model: The domain model that owns the classes and associations.
    """
    kind = payload.get("kind")
    cls = _EVENT_CLASSES.get(kind)
    if cls is None:
        return None
    approximated = bool(payload.get("approximated", False))
    if cls is UpdateAttribute:
        target = _find_class(model, payload.get("class"))
        attribute = _find_attribute(target, payload.get("attribute")) if target is not None else None
        if attribute is None:
            return None
        return ViolatingEvent(constraint, UpdateAttribute(attribute), approximated=approximated)
    if issubclass(cls, ObjectEvent):
        target = _find_class(model, payload.get("class"))
        if target is None:
            return None
        return ViolatingEvent(constraint, cls(target), approximated=approximated)
    association = _find_association(model, payload.get("association"))
    if association is None:
        return None
    return ViolatingEvent(constraint, cls(association), approximated=approximated)


def _find_class(model, name: Optional[str]) -> Optional[Class]:
    if not name or model is None:
        return None
    for typ in model.types:
        if isinstance(typ, Class) and typ.name == name:
            return typ
    return None


def _find_association(model, name: Optional[str]) -> Optional[Association]:
    if not name or model is None:
        return None
    for association in model.associations:
        if association.name == name:
            return association
    return None


def _find_attribute(target: Optional[Class], name: Optional[str]) -> Optional[Property]:
    if target is None or not name:
        return None
    for attribute in target.all_attributes():
        if attribute.name == name and not isinstance(attribute.owner, Association):
            return attribute
    return None
