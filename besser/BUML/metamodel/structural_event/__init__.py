"""Structural events that can violate an OCL invariant.

The event types follow the classification in Cabot and Teniente,
*Incremental Integrity Checking in UML/OCL Conceptual Schemas*
(chapter 4). ``Object`` replaces the thesis name ``ET`` and ``Link``
replaces ``RT``. An object event is linked to a class, a link event to an
association, and an attribute update to a property.

:class:`ViolatingEvent` is the association class between a constraint and
one structural event. Each instance links exactly one of each. The
constraint expression is the text those events were computed from.
"""

from besser.BUML.metamodel.structural_event.structural_event import (
    DeleteLink,
    DeleteObject,
    GeneralizeObject,
    InsertLink,
    InsertObject,
    LinkEvent,
    ObjectEvent,
    SpecializeObject,
    StructuralEvent,
    UpdateAttribute,
    ViolatingEvent,
    event_from_dict,
    event_payloads,
    event_to_dict,
    set_violating_events,
    violating_events,
)

__all__ = [
    "DeleteLink",
    "DeleteObject",
    "GeneralizeObject",
    "InsertLink",
    "InsertObject",
    "LinkEvent",
    "ObjectEvent",
    "SpecializeObject",
    "StructuralEvent",
    "UpdateAttribute",
    "ViolatingEvent",
    "event_from_dict",
    "event_payloads",
    "event_to_dict",
    "set_violating_events",
    "violating_events",
]
