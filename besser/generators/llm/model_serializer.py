"""
Serialize BUML metamodel objects into compact structured dicts for LLM context.

The LLM never sees Python code — it receives a clean JSON representation
of the domain model that's unambiguous, compact, and contains everything
needed to generate correct code.

A typical 10-class model serializes to ~2-3 KB of JSON.
"""

import json
from typing import Any

from besser.BUML.metamodel.structural import (
    UNLIMITED_MAX_MULTIPLICITY,
    AssociationClass,
    Class,
    DomainModel,
    Enumeration,
    PrimitiveDataType,
)


def _multiplicity_str(mult) -> str:
    """Format a Multiplicity as 'min..max' (e.g. '1..1', '0..*')."""
    max_val = "*" if mult.max == UNLIMITED_MAX_MULTIPLICITY else str(mult.max)
    return f"{mult.min}..{max_val}"


def _type_name(t) -> str:
    """Get a human-readable type name."""
    if t is None:
        return "None"
    return t.name


def _attribute_entry(attr) -> dict[str, Any]:
    """Produce the compact dict representation of a single ``Property``."""
    entry: dict[str, Any] = {
        "name": attr.name,
        "type": _type_name(attr.type),
    }
    if getattr(attr, "is_id", False):
        entry["is_id"] = True
    if getattr(attr, "is_optional", False):
        entry["is_optional"] = True
    mult = getattr(attr, "multiplicity", None)
    if mult and (mult.min != 1 or mult.max != 1):
        entry["multiplicity"] = _multiplicity_str(mult)
    if getattr(attr, "default_value", None) is not None:
        entry["default"] = attr.default_value
    visibility = getattr(attr, "visibility", None)
    if visibility and visibility != "public":
        entry["visibility"] = visibility
    return entry


def _method_entry(method) -> dict[str, Any]:
    """Produce the compact dict representation of a single ``Method``."""
    entry: dict[str, Any] = {"name": method.name}
    if method.type:
        entry["return_type"] = _type_name(method.type)
    if method.parameters:
        entry["parameters"] = [
            {"name": p.name, "type": _type_name(p.type)}
            for p in method.parameters
        ]
    visibility = getattr(method, "visibility", None)
    if visibility and visibility != "public":
        entry["visibility"] = visibility
    impl_type = getattr(method, "implementation_type", None)
    if impl_type:
        entry["implementation"] = str(impl_type.name).lower()
    # Where available, include the raw body — lets the LLM preserve or
    # translate behaviour instead of re-inventing it.
    code = getattr(method, "code", None) or getattr(method, "body", None)
    if isinstance(code, str) and code.strip():
        entry["body"] = code
    return entry


def _find_attribute_owner(attr, cls) -> str:
    """Walk the class's ancestors to find which one declares ``attr``."""
    try:
        ancestors = cls.all_parents()
    except Exception:
        return ""
    for parent in ancestors:
        if attr in getattr(parent, "attributes", set()):
            return parent.name
    return ""


def _collect_inherited_methods(cls) -> list:
    """Return ``[(method, owner_name), …]`` for methods from ancestors.

    A method is considered inherited if an ancestor declares it and the
    class itself does not declare a same-named method (i.e. not
    overridden). Overridden methods are still present in ``cls.methods``
    and will appear under the class's own ``methods`` entry, so we skip
    them here to avoid duplication.
    """
    try:
        ancestors = cls.all_parents()
    except Exception:
        return []
    own_names = {m.name for m in getattr(cls, "methods", [])}
    # Sort ancestors so the output is deterministic.
    sorted_ancestors = sorted(ancestors, key=lambda p: p.name)
    seen: set[str] = set()
    out = []
    for parent in sorted_ancestors:
        for m in sorted(getattr(parent, "methods", set()), key=lambda x: x.name):
            if m.name in own_names or m.name in seen:
                continue
            seen.add(m.name)
            out.append((m, parent.name))
    return out


def serialize_domain_model(model: DomainModel) -> dict[str, Any]:
    """
    Convert a DomainModel into a structured dict for LLM context.

    The output is designed to be:
    - **Compact**: minimizes tokens in the LLM context window
    - **Unambiguous**: clear structure, no parsing needed
    - **Complete**: all info the LLM needs to generate correct code

    Returns:
        Dict ready for ``json.dumps()``.
    """
    classes = []
    association_classes = []

    for cls in sorted(model.get_classes(), key=lambda c: c.name):
        cls_data: dict[str, Any] = {
            "name": cls.name,
        }
        if cls.is_abstract:
            cls_data["is_abstract"] = True

        # Attributes (declared directly on the class)
        attrs = [_attribute_entry(a) for a in sorted(cls.attributes, key=lambda a: a.name)]
        if attrs:
            cls_data["attributes"] = attrs

        # Methods (declared directly on the class)
        methods = [_method_entry(m) for m in sorted(cls.methods, key=lambda m: m.name)]
        if methods:
            cls_data["methods"] = methods

        # Parents (inheritance) — direct parents only.
        parents = list(cls.parents())
        if parents:
            cls_data["parents"] = [p.name for p in sorted(parents, key=lambda p: p.name)]

        # Flattened inherited members — saves the LLM from walking the MRO.
        # Each entry records the name of the parent the member came from so
        # the LLM can decide whether to override or call super().
        try:
            inherited_attrs = cls.inherited_attributes()
        except Exception:
            inherited_attrs = set()
        if inherited_attrs:
            cls_data["inherited_attributes"] = [
                {**_attribute_entry(a), "from": _find_attribute_owner(a, cls)}
                for a in sorted(inherited_attrs, key=lambda a: a.name)
            ]

        inherited_methods = _collect_inherited_methods(cls)
        if inherited_methods:
            cls_data["inherited_methods"] = [
                {**_method_entry(m), "from": owner_name}
                for m, owner_name in inherited_methods
            ]

        # Metadata
        if hasattr(cls, "metadata") and cls.metadata:
            meta = {}
            if cls.metadata.description:
                meta["description"] = cls.metadata.description
            if meta:
                cls_data["metadata"] = meta

        if isinstance(cls, AssociationClass):
            cls_data["association"] = cls.association.name
            association_classes.append(cls_data)
        else:
            classes.append(cls_data)

    # Enumerations
    enumerations = []
    for enum in sorted(model.get_enumerations(), key=lambda e: e.name):
        enumerations.append({
            "name": enum.name,
            "literals": [lit.name for lit in sorted(enum.literals, key=lambda l: l.name)],
        })

    # Associations
    associations = []
    for assoc in sorted(model.associations, key=lambda a: a.name):
        ends = []
        for end in assoc.ends:
            end_data: dict[str, Any] = {
                "role": end.name,
                "class": _type_name(end.type),
                "multiplicity": _multiplicity_str(end.multiplicity),
            }
            if not end.is_navigable:
                end_data["navigable"] = False
            if end.is_composite:
                end_data["composite"] = True
            ends.append(end_data)
        associations.append({
            "name": assoc.name,
            "ends": ends,
        })

    # Generalizations (inheritance)
    generalizations = []
    for gen in sorted(model.generalizations, key=lambda g: f"{g.general.name}-{g.specific.name}"):
        generalizations.append({
            "parent": gen.general.name,
            "child": gen.specific.name,
        })

    # Constraints
    constraints = []
    for c in sorted(model.constraints, key=lambda c: c.name):
        c_data: dict[str, Any] = {"name": c.name}
        if c.context:
            c_data["context"] = c.context.name
        if c.expression:
            c_data["expression"] = c.expression
        constraints.append(c_data)

    # Build output — only include non-empty sections
    result: dict[str, Any] = {"name": model.name}
    if classes:
        result["classes"] = classes
    if association_classes:
        result["association_classes"] = association_classes
    if enumerations:
        result["enumerations"] = enumerations
    if associations:
        result["associations"] = associations
    if generalizations:
        result["generalizations"] = generalizations
    if constraints:
        result["constraints"] = constraints
    return result


def serialize_gui_model(gui_model) -> dict[str, Any] | None:
    """
    Convert a GUIModel into a structured dict for LLM context.

    Returns None if gui_model is None or empty.
    """
    if gui_model is None:
        return None

    modules = []
    for module in gui_model.modules:
        screens = []
        for screen in module.screens:
            # Screen uses view_elements (from ViewContainer parent class)
            elements = getattr(screen, "view_elements", None) or getattr(screen, "view_components", None) or set()
            screen_data: dict[str, Any] = {
                "name": screen.name,
                "components": _serialize_components(elements),
            }
            if hasattr(screen, "is_main_page") and screen.is_main_page:
                screen_data["is_main_page"] = True
            screens.append(screen_data)
        modules.append({
            "name": module.name,
            "screens": screens,
        })

    return {"modules": modules} if modules else None


def _serialize_components(components) -> list[dict]:
    """Recursively serialize UI components."""
    result = []
    for comp in components:
        comp_data: dict[str, Any] = {
            "type": type(comp).__name__,
            "name": getattr(comp, "name", ""),
        }
        # Add type-specific fields
        if hasattr(comp, "label") and comp.label:
            comp_data["label"] = comp.label
        if hasattr(comp, "placeholder") and comp.placeholder:
            comp_data["placeholder"] = comp.placeholder
        if hasattr(comp, "data_source") and comp.data_source:
            ds = comp.data_source
            comp_data["data_source"] = {
                "class": _type_name(ds.source_class) if hasattr(ds, "source_class") and ds.source_class else None,
            }
        # Nested components (containers use view_elements)
        children = getattr(comp, "view_elements", None) or getattr(comp, "view_components", None)
        if children:
            comp_data["children"] = _serialize_components(children)
        result.append(comp_data)
    return result


def serialize_agent_model(agent_model) -> dict[str, Any] | None:
    """
    Convert an Agent model into a structured dict for LLM context.

    Returns None if agent_model is None.
    """
    if agent_model is None:
        return None

    states = []
    for state in agent_model.states:
        state_data: dict[str, Any] = {"name": state.name}
        if hasattr(state, "initial") and state.initial:
            state_data["initial"] = True
        if hasattr(state, "final") and state.final:
            state_data["final"] = True
        states.append(state_data)

    intents = []
    if hasattr(agent_model, "intents"):
        for intent in agent_model.intents:
            intent_data: dict[str, Any] = {"name": intent.name}
            if hasattr(intent, "training_sentences") and intent.training_sentences:
                intent_data["examples"] = list(intent.training_sentences)[:3]
            intents.append(intent_data)

    result: dict[str, Any] = {"name": agent_model.name}
    if states:
        result["states"] = states
    if intents:
        result["intents"] = intents
    return result


def serialize_object_model(object_model) -> dict[str, Any] | None:
    """Convert an ObjectModel into a structured dict for LLM context.

    Each object is dumped with its classifier name, attribute slots, and the
    link ends that connect it to other objects. The LLM can use these as seed
    data, database fixtures, or example request/response payloads.

    Returns None if object_model is None or empty.
    """
    if object_model is None:
        return None

    objects_out: list[dict[str, Any]] = []
    for obj in sorted(object_model.objects, key=lambda o: getattr(o, "name_", getattr(o, "name", ""))):
        obj_name = getattr(obj, "name_", None) or getattr(obj, "name", "")
        classifier = getattr(obj, "classifier", None)
        entry: dict[str, Any] = {
            "name": obj_name,
            "class": _type_name(classifier) if classifier else "Unknown",
        }
        slots_data: list[dict[str, Any]] = []
        for slot in getattr(obj, "slots", []) or []:
            attr = getattr(slot, "attribute", None)
            value = getattr(slot, "value", None)
            raw_value = getattr(value, "value", value) if value is not None else None
            # The downstream prompt builder calls plain ``json.dumps`` without
            # a ``default=`` callback, so anything not natively JSON-serialisable
            # must be stringified here.
            try:
                json.dumps(raw_value)
                safe_value = raw_value
            except TypeError:
                safe_value = str(raw_value)
            slots_data.append({
                "attribute": attr.name if attr else "?",
                "value": safe_value,
            })
        if slots_data:
            entry["slots"] = slots_data
        objects_out.append(entry)

    links_out: list[dict[str, Any]] = []
    seen_link_ids: set[int] = set()
    for link in getattr(object_model, "links", None) or []:
        if id(link) in seen_link_ids:
            continue
        seen_link_ids.add(id(link))
        assoc = getattr(link, "association", None)
        ends = []
        for end in getattr(link, "connections", []) or []:
            end_obj = getattr(end, "object", None)
            end_obj_name = getattr(end_obj, "name_", None) or getattr(end_obj, "name", "") if end_obj else ""
            assoc_end = getattr(end, "association_end", None)
            ends.append({
                "role": assoc_end.name if assoc_end else getattr(end, "name", ""),
                "object": end_obj_name,
            })
        links_out.append({
            "name": getattr(link, "name", ""),
            "association": assoc.name if assoc else "",
            "ends": ends,
        })

    result: dict[str, Any] = {"name": getattr(object_model, "name", "ObjectModel")}
    if objects_out:
        result["objects"] = objects_out
    if links_out:
        result["links"] = links_out

    # Return None rather than an empty shell — the prompt builder uses
    # truthiness to decide whether to include the section.
    if not objects_out and not links_out:
        return None
    return result


def serialize_state_machines(state_machines) -> list[dict[str, Any]] | None:
    """Convert a list (or single instance) of StateMachine into structured dicts.

    Each state machine is serialised with its states (name / initial / final)
    and the outgoing transitions from each state (source → dest, event,
    conditions). The LLM uses this to wire state fields + transition guards
    in the generated code.

    Returns None if the argument is None or empty.
    """
    if state_machines is None:
        return None

    if not isinstance(state_machines, (list, tuple, set)):
        state_machines = [state_machines]

    result: list[dict[str, Any]] = []
    for sm in state_machines:
        if sm is None:
            continue
        states_data: list[dict[str, Any]] = []
        for state in getattr(sm, "states", []) or []:
            state_entry: dict[str, Any] = {"name": state.name}
            if getattr(state, "initial", False):
                state_entry["initial"] = True
            if getattr(state, "final", False):
                state_entry["final"] = True
            transitions_data: list[dict[str, Any]] = []
            for transition in getattr(state, "transitions", []) or []:
                event = getattr(transition, "event", None)
                conditions = getattr(transition, "conditions", []) or []
                transitions_data.append({
                    "to": transition.dest.name if transition.dest else "",
                    "event": event.name if event else None,
                    "conditions": [c.name for c in conditions] if conditions else [],
                })
            if transitions_data:
                state_entry["transitions"] = transitions_data
            states_data.append(state_entry)

        sm_entry: dict[str, Any] = {"name": getattr(sm, "name", "StateMachine")}
        if states_data:
            sm_entry["states"] = states_data
        properties = getattr(sm, "properties", None)
        if properties:
            sm_entry["properties"] = [
                {
                    "section": getattr(p, "section", ""),
                    "name": getattr(p, "name", ""),
                    "value": getattr(p, "value", None),
                }
                for p in properties
            ]
        result.append(sm_entry)

    return result or None


def serialize_quantum_circuit(circuit) -> dict[str, Any] | None:
    """Convert a QuantumCircuit into a structured dict for LLM context.

    Captures register sizes and the ordered operation list. Gate parameters
    are kept shallow — the LLM only needs enough to re-emit Qiskit / Cirq code.

    Returns None if circuit is None.
    """
    if circuit is None:
        return None

    qregs = [
        {"name": getattr(q, "name", ""), "size": getattr(q, "size", 0)}
        for q in getattr(circuit, "qregs", []) or []
    ]
    cregs = [
        {"name": getattr(c, "name", ""), "size": getattr(c, "size", 0)}
        for c in getattr(circuit, "cregs", []) or []
    ]

    operations = []
    for op in getattr(circuit, "operations", []) or []:
        op_entry: dict[str, Any] = {
            "type": type(op).__name__,
            "name": getattr(op, "name", ""),
        }
        targets = getattr(op, "target_qubits", None)
        if targets:
            op_entry["targets"] = list(targets)
        controls = getattr(op, "control_qubits", None)
        if controls:
            op_entry["controls"] = list(controls)
        # Measurement-specific
        if hasattr(op, "output_bit") and op.output_bit is not None:
            op_entry["output_bit"] = op.output_bit
        if hasattr(op, "basis") and op.basis:
            op_entry["basis"] = op.basis
        operations.append(op_entry)

    result: dict[str, Any] = {
        "name": getattr(circuit, "name", "QuantumCircuit"),
        "num_qubits": getattr(circuit, "num_qubits", 0),
        "num_clbits": getattr(circuit, "num_clbits", 0),
    }
    if qregs:
        result["qregs"] = qregs
    if cregs:
        result["cregs"] = cregs
    if operations:
        result["operations"] = operations
    return result
