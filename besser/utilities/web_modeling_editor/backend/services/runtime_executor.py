"""Runtime executor for the web modeling editor.

Lets the user invoke a method on a specific Object instance from the
editor's object diagram, without having to generate a platform first.
Stateless: each request carries the current class + object diagrams,
the executor materializes Python classes on the fly, runs the method,
and returns the mutated attributes for the frontend to merge.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import copy
import re

from besser.runtime.engine import Engine
from besser.runtime.materialize import build_associations_index, materialize_classes
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.object_diagram_processor import (
    process_object_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


# ---------------------------------------------------------------------------
# Pre-processing — tolerate unfilled attribute slots
# ---------------------------------------------------------------------------
# The shared ``process_object_diagram`` strictly coerces every attribute it
# encounters (float("") raises, datetime parsing of "" warns, enumeration
# lookup of "" fails). For our runtime use case this is wrong: a method
# invocation only needs the attributes it actually touches, and an object
# half-filled by the user (or partially loaded from a saved diagram) should
# still be runnable. We rewrite ObjectAttribute strings of the form
# ``"+ name: type = "`` (empty value) to just ``"+ name: type"`` so the
# converter treats those slots as "no value" and skips coercion. The
# materialized Python class will then expose them as ``None`` — and a
# method body referencing them can use the standard ``(self.x or 0)``
# pattern users are already writing.
_TRAILING_EQUALS_BLANK_RE = re.compile(r"\s*=\s*$")


def _strip_blank_attribute_values(object_diagram_json):
    """Return a deep-copied object diagram JSON with blank ``= `` suffixes
    stripped from ObjectAttribute name strings. Original payload is left
    untouched so the caller's data isn't mutated under their feet."""
    cleaned = copy.deepcopy(object_diagram_json)
    elements = (
        cleaned.get("model", {}).get("elements", {})
        if isinstance(cleaned, dict)
        else {}
    )
    if not isinstance(elements, dict):
        return cleaned
    for element in elements.values():
        if not isinstance(element, dict):
            continue
        if element.get("type") != "ObjectAttribute":
            continue
        name = element.get("name")
        if not isinstance(name, str):
            continue
        # If the name carries a trailing ``= `` with nothing after it, drop
        # the equals so the converter doesn't try to coerce empty string.
        element["name"] = _TRAILING_EQUALS_BLANK_RE.sub("", name)
    return cleaned


# ---------------------------------------------------------------------------
# In-memory instance store — shaped like the generated platform's
# ``instance_manager`` so the same ``Engine`` can drive both.
# ---------------------------------------------------------------------------
class _InMemoryInstanceStore:
    """Holds Object state as a flat dict keyed by (class_name, instance_name).

    The web editor identifies instances by name (the user-facing label,
    e.g. ``StorageTank_A``) since that's what shows up in the diagram.
    The generated-platform store uses UUIDs — both interfaces look the
    same to ``Engine``, so the swap is invisible.

    Links from ``ObjectModel.links`` are projected onto the *first*
    connection's owning object as an outgoing link. The engine's
    reverse-link index handles bidirectional navigation, so storing
    each link on a single side avoids double-counting on materialize.
    """

    def __init__(self, object_model: Any):
        self._rows: Dict[tuple, Dict[str, Any]] = {}
        for obj in object_model.objects:
            class_name = obj.classifier.name
            instance_name = obj.name
            attributes = {
                slot.attribute.name: slot.value.value for slot in obj.slots
            }
            self._rows[(class_name, instance_name)] = {
                "id": instance_name,
                "class_name": class_name,
                "instance_name": instance_name,
                "attributes": attributes,
                "links": [],
            }

        # Project Links onto the first connection's owner. The engine's
        # reverse-link index will reach the other end on demand.
        for link in getattr(object_model, "links", []) or []:
            connections = list(getattr(link, "connections", []) or [])
            if len(connections) < 2:
                continue
            source_end, target_end = connections[0], connections[1]
            source_obj = getattr(source_end, "object", None)
            target_obj = getattr(target_end, "object", None)
            association = getattr(link, "association", None)
            assoc_name = getattr(association, "name", None)
            if not (source_obj and target_obj and assoc_name):
                continue
            source_key = (
                source_obj.classifier.name,
                source_obj.name,
            )
            row = self._rows.get(source_key)
            if row is None:
                continue
            row["links"].append({
                "association_name": assoc_name,
                "target_class": target_obj.classifier.name,
                "target_id": target_obj.name,
            })

    def get_instance(self, class_name: str, instance_id: str) -> Optional[Dict[str, Any]]:
        return self._rows.get((class_name, instance_id))

    def get_all_instances(self) -> List[Dict[str, Any]]:
        """Required by ``Engine._build_reverse_index`` so the kernel can
        scan inbound links across the whole object diagram."""
        return list(self._rows.values())

    def update_instance(
        self,
        class_name: str,
        instance_id: str,
        attributes: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        row = self._rows.get((class_name, instance_id))
        if row is None:
            return None
        row["attributes"].update(attributes)
        return row


# ---------------------------------------------------------------------------
# Public entry point used by the router.
# ---------------------------------------------------------------------------
def invoke_method_on_diagram(
    class_diagram_json: Dict[str, Any],
    object_diagram_json: Dict[str, Any],
    class_name: str,
    instance_name: str,
    method_name: str,
    args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Materialize, invoke, return mutated state.

    Args:
        class_diagram_json: The editor's class-diagram JSON payload
            ({"title": ..., "model": ...}).
        object_diagram_json: The editor's object-diagram JSON payload.
            ``referenceDiagramData`` inside is honoured by the converter.
        class_name: Domain class of the target instance (e.g. ``"StorageTank"``).
        instance_name: User-facing instance label as it appears on the
            diagram (e.g. ``"StorageTank_A"``).
        method_name: Method to dispatch.
        args: Kwargs forwarded to the method. ``None`` means call with no args.

    Returns:
        ``{instance_name, class_name, method_name, args, result,
        updated_attributes}``. ``updated_attributes`` is the dict the
        frontend overlays onto its in-diagram representation.

    Raises:
        ConversionError: Diagram payload couldn't be parsed (or missing
            reference data on the object diagram).
        KeyError: The target instance doesn't exist in the object diagram.
        ValueError: The class isn't in the domain model.
        AttributeError: The class doesn't declare a callable by that name.
        TypeError: ``args`` doesn't match the method signature.
    """
    domain_model = process_class_diagram(class_diagram_json)

    # ``process_object_diagram`` needs the same shape ``conversion_router``
    # uses — title + model — and refuses if no reference class diagram is
    # supplied. We pass the just-built DomainModel directly so the user
    # doesn't need to nest ``referenceDiagramData`` inside the JSON.
    # Strip blank ``= `` suffixes first so unfilled slots don't trip the
    # converter's strict coercion (float("") etc.).
    cleaned_object_diagram = _strip_blank_attribute_values(object_diagram_json)
    object_model = process_object_diagram(cleaned_object_diagram, domain_model)

    class_map = materialize_classes(domain_model)
    associations_index = build_associations_index(domain_model)
    store = _InMemoryInstanceStore(object_model)
    engine = Engine(
        instance_manager=store,
        class_map=class_map,
        associations_index=associations_index,
    )

    result = engine.invoke(
        class_name=class_name,
        instance_id=instance_name,
        method_name=method_name,
        args=args,
    )

    return {
        "instance_name": instance_name,
        "class_name": class_name,
        "method_name": method_name,
        "args": result["args"],
        "result": result["result"],
        "updated_attributes": result["instance"]["attributes"],
    }
