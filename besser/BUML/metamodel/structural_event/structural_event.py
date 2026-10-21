"""Metaclasses for the structural events of incremental integrity checking.

Each class is one event type from section 4.1 of Cabot's thesis. A
:class:`ViolatingEvent` stores, on a :class:`~besser.BUML.metamodel.structural.Constraint`,
the fact that one of those events may violate it.
"""

from __future__ import annotations

from typing import Any, Optional

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

    Subclasses name the kind of change. Instances point at the model element
    the change applies to. They are event *types*: two insertions of different
    objects into the same class are the same event type.
    """

    def key(self) -> tuple:
        """Identity of this event type, independent of the Python object id."""
        raise NotImplementedError

    def kind(self) -> str:
        """Name of the event type, for example ``InsertClass``."""
        return type(self).__name__


class ClassEvent(StructuralEvent):
    """A structural event whose parameter is a class.

    Args:
        target (Class): The class the event changes the population of.
    """

    def __init__(self, target: Class, timestamp=None, is_derived: bool = False, uncertainty: float = 0.0):
        super().__init__(timestamp, is_derived, uncertainty)
        self.target: Class = target

    @property
    def target(self) -> Class:
        """Class: The class this event changes."""
        return self.__target

    @target.setter
    def target(self, target: Class):
        """Class: Set the class this event changes.

        Raises:
            TypeError: if ``target`` is not a :class:`Class`.
        """
        if not isinstance(target, Class):
            raise TypeError(f"target must be a Class, got {type(target).__name__}")
        self.__target = target

    def key(self) -> tuple:
        return (self.kind(), self.target)

    def __repr__(self):
        return f"{self.kind()}({self.target.name})"


class InsertClass(ClassEvent):
    """Insertion of a new object into a class.

    The new object may have its attributes initialized. It does not participate
    in any association yet. Creating an object of a subclass also inserts it
    into each superclass; this event records the class named by ``target``.
    """


class DeleteClass(ClassEvent):
    """Deletion of an object from a class.

    Deleting an object also deletes it from its superclasses and subclasses.
    This event records the class named by ``target``.
    """


class SpecializeClass(ClassEvent):
    """Reclassification of an existing object into ``target``.

    The object already belongs to a superclass of ``target`` and gains
    ``target`` as its new class. This is the thesis event ``SpecializeET``.
    """


class GeneralizeClass(ClassEvent):
    """Reclassification of an object away from a subclass, up to ``target``.

    The object leaves a direct subclass of ``target`` and is afterwards only
    an instance of ``target`` (and its superclasses). This is the thesis event
    ``GeneralizeET``.
    """


class UpdateAttribute(StructuralEvent):
    """Replacement of the value of one attribute of an object.

    Args:
        attribute (Property): The attribute whose value changes. It is owned
            by a class, not by an association.
        target (Class): The class of the object whose attribute changes.
    """

    def __init__(self, attribute: Property, target: Class, timestamp=None,
                 is_derived: bool = False, uncertainty: float = 0.0):
        super().__init__(timestamp, is_derived, uncertainty)
        self.attribute: Property = attribute
        self.target: Class = target

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

    @property
    def target(self) -> Class:
        """Class: The class of the object being updated."""
        return self.__target

    @target.setter
    def target(self, target: Class):
        """Class: Set the class of the object being updated.

        Raises:
            TypeError: if ``target`` is not a :class:`Class`.
        """
        if not isinstance(target, Class):
            raise TypeError(f"target must be a Class, got {type(target).__name__}")
        self.__target = target

    def key(self) -> tuple:
        return (self.kind(), self.target, self.attribute)

    def __repr__(self):
        return f"UpdateAttribute({self.attribute.name}, {self.target.name})"


class AssociationEvent(StructuralEvent):
    """A structural event whose parameter is an association.

    Args:
        target (Association): The association the event inserts a link into
            or deletes a link from.
    """

    def __init__(self, target: Association, timestamp=None, is_derived: bool = False,
                 uncertainty: float = 0.0):
        super().__init__(timestamp, is_derived, uncertainty)
        self.target: Association = target

    @property
    def target(self) -> Association:
        """Association: The association this event changes."""
        return self.__target

    @target.setter
    def target(self, target: Association):
        """Association: Set the association this event changes.

        Raises:
            TypeError: if ``target`` is not an :class:`Association`.
        """
        if not isinstance(target, Association):
            raise TypeError(f"target must be an Association, got {type(target).__name__}")
        self.__target = target

    def key(self) -> tuple:
        return (self.kind(), self.target)

    def __repr__(self):
        return f"{self.kind()}({self.target.name})"


class InsertAssociation(AssociationEvent):
    """Insertion of a new link into an association.

    For an association class, the link is the association facet of the new
    link object. The object facet is a separate :class:`InsertClass`.
    """


class DeleteAssociation(AssociationEvent):
    """Deletion of a link from an association."""


class ViolatingEvent(Element):
    """Link between a constraint and one structural event that may violate it.

    The link is stored on the constraint (``constraint.violating_events``).
    Creating a :class:`ViolatingEvent` registers it there. Setting
    ``constraint`` to ``None`` unregisters it.

    ``approximated`` is true when the event was added because part of the
    expression is outside the fragment the analysis handles exactly. Such an
    event may be a false positive. Dropping it could hide a real violation,
    so it is kept.

    Args:
        constraint (Constraint): The invariant this event may violate.
        event (StructuralEvent): The event type.
        approximated (bool): True when the event comes from over-approximation.
    """

    def __init__(self, constraint: Optional[Constraint], event: StructuralEvent,
                 approximated: bool = False, timestamp=None, is_derived: bool = False,
                 uncertainty: float = 0.0):
        super().__init__(timestamp, is_derived, uncertainty)
        self.__constraint: Optional[Constraint] = None
        self.event: StructuralEvent = event
        self.approximated: bool = approximated
        self.constraint = constraint

    @property
    def constraint(self) -> Optional[Constraint]:
        """Constraint: The constraint this event may violate, or None if detached."""
        return self.__constraint

    @constraint.setter
    def constraint(self, constraint: Optional[Constraint]):
        """Constraint: Attach this link to ``constraint``, or detach it when None.

        Raises:
            TypeError: if ``constraint`` is neither a :class:`Constraint` nor None.
        """
        if constraint is not None and not isinstance(constraint, Constraint):
            raise TypeError(f"constraint must be a Constraint or None, got {type(constraint).__name__}")
        current = self.__constraint
        if current is constraint:
            return
        if current is not None:
            current.violating_events.discard(self)
        self.__constraint = constraint
        if constraint is not None:
            constraint.violating_events.add(self)

    @property
    def event(self) -> StructuralEvent:
        """StructuralEvent: The event type that may violate the constraint."""
        return self.__event

    @event.setter
    def event(self, event: StructuralEvent):
        """StructuralEvent: Set the event type.

        Raises:
            TypeError: if ``event`` is not a :class:`StructuralEvent`.
        """
        if not isinstance(event, StructuralEvent):
            raise TypeError(f"event must be a StructuralEvent, got {type(event).__name__}")
        self.__event = event

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
        return f"ViolatingEvent({name}, {self.event}, approximated={self.approximated})"


_EVENT_CLASSES = {
    "InsertClass": InsertClass,
    "DeleteClass": DeleteClass,
    "SpecializeClass": SpecializeClass,
    "GeneralizeClass": GeneralizeClass,
    "UpdateAttribute": UpdateAttribute,
    "InsertAssociation": InsertAssociation,
    "DeleteAssociation": DeleteAssociation,
}


def event_payloads(constraint: Constraint) -> list[dict[str, Any]]:
    """JSON-ready events stored on ``constraint``, in a stable order."""
    links = sorted(getattr(constraint, "violating_events", ()) or (), key=_link_sort_key)
    return [event_to_dict(link) for link in links]


def _link_sort_key(link: ViolatingEvent) -> tuple:
    event = link.event
    target = getattr(event, "target", None)
    attribute = getattr(event, "attribute", None)
    return (
        event.kind(),
        getattr(target, "name", "") or "",
        getattr(attribute, "name", "") or "",
        link.approximated,
    )


def event_to_dict(link: ViolatingEvent) -> dict[str, Any]:
    """Serialize one violating-event link to a JSON-ready dictionary.

    Class and association parameters are stored by name. The dictionary is
    what the editor keeps on an OCL constraint box.
    """
    event = link.event
    payload: dict[str, Any] = {
        "constraint": link.constraint.name if link.constraint is not None else None,
        "kind": event.kind(),
        "approximated": link.approximated,
    }
    if isinstance(event, UpdateAttribute):
        payload["class"] = event.target.name
        payload["attribute"] = event.attribute.name
    elif isinstance(event, ClassEvent):
        payload["class"] = event.target.name
    elif isinstance(event, AssociationEvent):
        payload["association"] = event.target.name
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
        if target is None or attribute is None:
            return None
        return ViolatingEvent(constraint, UpdateAttribute(attribute, target), approximated=approximated)
    if issubclass(cls, ClassEvent):
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
