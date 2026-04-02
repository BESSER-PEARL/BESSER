"""
Serialize BUML metamodel objects into compact structured dicts for LLM context.

The LLM never sees Python code — it receives a clean JSON representation
of the domain model that's unambiguous, compact, and contains everything
needed to generate correct code.

A typical 10-class model serializes to ~2-3 KB of JSON.
"""

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

        # Attributes
        attrs = []
        for attr in sorted(cls.attributes, key=lambda a: a.name):
            attr_data: dict[str, Any] = {
                "name": attr.name,
                "type": _type_name(attr.type),
            }
            if attr.is_id:
                attr_data["is_id"] = True
            if attr.is_optional:
                attr_data["is_optional"] = True
            if attr.multiplicity and (attr.multiplicity.min != 1 or attr.multiplicity.max != 1):
                attr_data["multiplicity"] = _multiplicity_str(attr.multiplicity)
            if attr.default_value is not None:
                attr_data["default"] = attr.default_value
            if attr.visibility and attr.visibility != "public":
                attr_data["visibility"] = attr.visibility
            attrs.append(attr_data)
        if attrs:
            cls_data["attributes"] = attrs

        # Methods
        methods = []
        for m in sorted(cls.methods, key=lambda m: m.name):
            m_data: dict[str, Any] = {"name": m.name}
            if m.type:
                m_data["return_type"] = _type_name(m.type)
            if m.parameters:
                m_data["parameters"] = [
                    {"name": p.name, "type": _type_name(p.type)}
                    for p in m.parameters
                ]
            if m.visibility and m.visibility != "public":
                m_data["visibility"] = m.visibility
            if hasattr(m, "implementation_type") and m.implementation_type:
                m_data["implementation"] = str(m.implementation_type.name).lower()
            methods.append(m_data)
        if methods:
            cls_data["methods"] = methods

        # Parents (inheritance)
        parents = list(cls.parents())
        if parents:
            cls_data["parents"] = [p.name for p in parents]

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
            screen_data: dict[str, Any] = {
                "name": screen.name,
                "components": _serialize_components(screen.view_components),
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
        # Nested components (containers)
        if hasattr(comp, "view_components") and comp.view_components:
            comp_data["children"] = _serialize_components(comp.view_components)
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
