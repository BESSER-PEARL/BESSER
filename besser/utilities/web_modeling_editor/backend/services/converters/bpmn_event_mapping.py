"""Shared WME ↔ B-UML BPMN event mapping.

WME flattens BPMN events into a single string per element ("default", "message-catch",
"timer", "terminate", …). The B-UML BPMN metamodel splits an event along two orthogonal
axes (decision D2 in ``.claude/bpmn/01-bpmn-metamodel-design.md``):

* ``EventDirection`` — ``CATCH`` or ``THROW``
* ``EventDefinitionType`` — ``NONE`` / ``MESSAGE`` / ``TIMER`` / …

This module is the *single* source of truth for that translation. Both the
``json_to_buml`` processor and the ``buml_to_json`` converter import from here so the two
directions stay symmetric (per the §7 round-trip guarantee).

The full mapping table lives in Appendix A of
``.claude/bpmn/03-bpmn-converters-guide.md``.
"""

from besser.BUML.metamodel.bpmn import (
    EndEvent,
    Event,
    EventDefinitionType,
    EventDirection,
    IntermediateEvent,
    StartEvent,
)


# WME's per-class enums use ``"default"`` for the no-trigger event; the metamodel uses
# ``EventDefinitionType.NONE``.
_WME_DEFAULT = "default"


def parse_event_type(event_class: type, event_type: str) -> tuple[EventDirection, EventDefinitionType]:
    """WME ``eventType`` → ``(EventDirection, EventDefinitionType)``.

    Args:
        event_class: ``StartEvent`` / ``IntermediateEvent`` / ``EndEvent`` (the metamodel
            class to construct).
        event_type: The WME ``eventType`` string from the JSON node.

    Returns:
        Tuple ``(EventDirection, EventDefinitionType)``.

    Raises:
        ValueError: if ``event_type`` is not a recognised WME string for ``event_class``.
    """
    if event_class is StartEvent:
        # Start events always catch; only the trigger varies.
        if event_type == _WME_DEFAULT:
            return EventDirection.CATCH, EventDefinitionType.NONE
        return EventDirection.CATCH, EventDefinitionType(event_type)

    if event_class is EndEvent:
        # End events always throw; only the trigger varies.
        if event_type == _WME_DEFAULT:
            return EventDirection.THROW, EventDefinitionType.NONE
        return EventDirection.THROW, EventDefinitionType(event_type)

    if event_class is IntermediateEvent:
        # Intermediate events carry direction in the suffix: "message-catch", "timer-throw"…
        if event_type == _WME_DEFAULT:
            return EventDirection.CATCH, EventDefinitionType.NONE
        if "-" not in event_type:
            raise ValueError(
                f"IntermediateEvent eventType '{event_type}' must be 'default' or "
                f"'<defn>-<catch|throw>'."
            )
        defn_str, dir_str = event_type.rsplit("-", 1)
        return EventDirection(dir_str), EventDefinitionType(defn_str)

    raise ValueError(
        f"parse_event_type: unsupported event class {event_class.__name__}."
    )


def serialise_event_type(event: Event) -> str:
    """``Event`` (StartEvent / IntermediateEvent / EndEvent) → WME ``eventType`` string.

    The inverse of :func:`parse_event_type`, by construction (round-trip is identity for
    every legal triple — see the metamodel's ``_LEGAL_EVENT_DEFINITIONS`` table).
    """
    if isinstance(event, (StartEvent, EndEvent)):
        if event.event_definition is EventDefinitionType.NONE:
            return _WME_DEFAULT
        return event.event_definition.value

    if isinstance(event, IntermediateEvent):
        if (event.direction is EventDirection.CATCH
                and event.event_definition is EventDefinitionType.NONE):
            return _WME_DEFAULT
        return f"{event.event_definition.value}-{event.direction.value}"

    raise ValueError(
        f"serialise_event_type: unsupported event {type(event).__name__}."
    )
