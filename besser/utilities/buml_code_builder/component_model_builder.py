"""Component model code builder.

Generates a self-contained Python module from a ``ComponentModel`` that
re-creates the model on ``exec()``. Implements 03-... §2 / §4.1 / §5
(option-b emission, sort_by_timestamp determinism, layout passthrough,
cross-diagram-ref dot-assignment per Q1=a).
"""

from typing import Optional

from besser.BUML.metamodel.uml_component import (
    AgenticComponent,
    AgenticEdge,
    Component,
    ComponentDependency,
    ComponentModel,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    Permission,
    Skill,
    Subsystem,
    Tool,
)
from besser.utilities.buml_code_builder.common import (
    _escape_python_string,
    safe_var_name,
)
from besser.utilities.utils import sort_by_timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NameDispenser:
    """Mint unique Python variable names keyed by metamodel object identity.

    Repro of the BPMN-04- pattern: keep one map per ``component_model_to_code``
    call, so collisions stay local and `<prefix>_<safe-name>` is reused
    across the call (essential for ``add_*`` references to resolve).
    """

    def __init__(self):
        self._used: set = set()
        self._for_obj: dict = {}

    def name_for(self, obj) -> str:
        if obj in self._for_obj:
            return self._for_obj[obj]
        prefix = type(obj).__name__.lower()
        base = safe_var_name(obj.name, lowercase=True) or prefix
        candidate = f"{prefix}_{base}" if base != prefix else prefix
        suffix = 1
        while candidate in self._used:
            candidate = f"{prefix}_{base}_{suffix}"
            suffix += 1
        self._used.add(candidate)
        self._for_obj[obj] = candidate
        return candidate


def _emit_str_list(values) -> str:
    """Emit a Python list of strings, each escaped for safe interpolation."""
    if not values:
        return "[]"
    parts = [f"'{_escape_python_string(v)}'" for v in values]
    return f"[{', '.join(parts)}]"


def _emit_layout_line(var_name: str, layout: Optional[dict]) -> Optional[str]:
    """Emit a ``<var>.layout = {...}`` line, or ``None`` to skip.

    Layout dicts arrive opaque from the converter side; ``repr()`` is the
    natural emission (dicts of primitives + nested dicts). When a value is
    a str, ``repr`` already escapes via Python's default repr — sufficient
    for the round-trip contract.
    """
    if not layout:
        return None
    return f"{var_name}.layout = {layout!r}"


def _collect_imports(model: ComponentModel) -> list:
    """Return the metamodel symbols this model actually needs imported.

    Always include the model class (``ComponentModel`` or, for an agentic
    model, ``AgenticComponentModel``). Pull in concrete classes only when
    used so the emitted ``.py`` doesn't carry dead imports.
    """
    needed = {type(model).__name__}
    has_agent_category = False
    has_locality_nondefault = False
    has_agentic_edge_kind = False

    for c in model.components:
        if isinstance(c, Subsystem):
            needed.add("Subsystem")
        elif isinstance(c, Skill):
            needed.add("Skill")
        elif isinstance(c, Tool):
            needed.add("Tool")
        elif isinstance(c, AgenticComponent):
            needed.add("AgenticComponent")
        else:
            needed.add("Component")
        if isinstance(c, AgenticComponent) and c.agent_category.name != "NONE":
            has_agent_category = True
        if c.locality.name != "LOCAL":
            has_locality_nondefault = True

    if model.interfaces:
        needed.add("Interface")
    if getattr(model, "permissions", None):
        needed.add("Permission")

    for rel in model.relationships:
        if isinstance(rel, AgenticEdge):
            needed.add("AgenticEdge")
            has_agentic_edge_kind = True
        elif isinstance(rel, ComponentDependency):
            needed.add("ComponentDependency")
        elif isinstance(rel, InterfaceProvided):
            needed.add("InterfaceProvided")
        elif isinstance(rel, InterfaceRequired):
            needed.add("InterfaceRequired")

    enum_imports = []
    if has_agent_category:
        enum_imports.append("AgentCategory")
    if has_locality_nondefault or has_agentic_edge_kind:
        enum_imports.append("Locality")
    if has_agentic_edge_kind:
        enum_imports.append("AgenticEdgeKind")

    ordered_classes = [
        c for c in (
            "ComponentModel", "AgenticComponentModel",
            "Component", "Subsystem", "AgenticComponent", "Skill", "Tool",
            "Interface", "Permission",
            "InterfaceProvided", "InterfaceRequired",
            "ComponentDependency", "AgenticEdge",
        )
        if c in needed
    ]
    return ordered_classes + enum_imports


# ---------------------------------------------------------------------------
# Component constructor emission
# ---------------------------------------------------------------------------

def _emit_component(component: Component, var_name: str, dispenser: _NameDispenser) -> list:
    """Emit the constructor + post-construction assignment lines for a
    Component / Subsystem / Skill / Tool. Returns a list of source lines.
    """
    lines = []
    class_name = type(component).__name__
    args = [f"name='{_escape_python_string(component.name)}'"]
    if isinstance(component, AgenticComponent):
        if component.agent_category.name != "NONE":
            args.append(
                f"agent_category=AgentCategory.{component.agent_category.name}"
            )
        if component.is_human:
            args.append("is_human=True")
    if component.locality.name != "LOCAL":
        args.append(f"locality=Locality.{component.locality.name}")
    if component.stereotypes:
        args.append(f"stereotypes={_emit_str_list(component.stereotypes)}")
    lines.append(f"{var_name} = {class_name}({', '.join(args)})")

    # Cross-diagram refs (Q1=a — dot-assignment, omitted when empty).
    if component.realizes:
        lines.append(f"{var_name}.realizes = {_emit_str_list(component.realizes)}")
    if isinstance(component, AgenticComponent) and component.process_model_refs:
        lines.append(
            f"{var_name}.process_model_refs = "
            f"{_emit_str_list(component.process_model_refs)}"
        )

    layout_line = _emit_layout_line(var_name, component.layout)
    if layout_line is not None:
        lines.append(layout_line)
    return lines


def _emit_permission(permission: Permission, var_name: str) -> list:
    """Emit a Permission constructor + optional layout line."""
    lines = []
    args = [f"name='{_escape_python_string(permission.name)}'"]
    args.append(f"scope='{_escape_python_string(permission.scope)}'")
    if permission.stereotypes:
        args.append(f"stereotypes={_emit_str_list(permission.stereotypes)}")
    lines.append(f"{var_name} = Permission({', '.join(args)})")
    layout_line = _emit_layout_line(var_name, permission.layout)
    if layout_line is not None:
        lines.append(layout_line)
    return lines


def _emit_interface(interface: Interface, var_name: str) -> list:
    """Emit an Interface constructor + optional layout line."""
    lines = []
    args = [f"name='{_escape_python_string(interface.name)}'"]
    if interface.stereotypes:
        args.append(f"stereotypes={_emit_str_list(interface.stereotypes)}")
    lines.append(f"{var_name} = Interface({', '.join(args)})")
    layout_line = _emit_layout_line(var_name, interface.layout)
    if layout_line is not None:
        lines.append(layout_line)
    return lines


def _emit_relationship(rel, dispenser: _NameDispenser) -> list:
    """Emit a relationship — inline or via a temp variable when it has a
    layout to round-trip. Returns the source lines (the
    ``component_model.add_relationship(...)`` line is the caller's job)."""
    rel_class = type(rel).__name__
    source_var = dispenser.name_for(rel.source)
    target_var = dispenser.name_for(rel.target)

    args = [f"source={source_var}", f"target={target_var}"]
    if isinstance(rel, AgenticEdge):
        args.append(f"kind=AgenticEdgeKind.{rel.kind.name}")
        if rel.permissions:
            perm_vars = ", ".join(dispenser.name_for(p) for p in rel.permissions)
            args.append(f"permissions=[{perm_vars}]")
    if rel.name:
        args.append(f"name='{_escape_python_string(rel.name)}'")
    if rel.stereotypes:
        args.append(f"stereotypes={_emit_str_list(rel.stereotypes)}")

    constructor = f"{rel_class}({', '.join(args)})"

    # If the relationship carries a layout we need to round-trip, bind it
    # to a variable so we can attach .layout before adding to the model.
    if rel.layout:
        rel_var = dispenser.name_for(rel)
        return [
            f"{rel_var} = {constructor}",
            _emit_layout_line(rel_var, rel.layout),
            f"component_model.add_relationship({rel_var})",
        ]
    return [f"component_model.add_relationship({constructor})"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def component_model_to_code(model: ComponentModel,
                            file_path: Optional[str] = None,
                            model_var_name: str = "component_model") -> str:
    """Generate Python code that re-creates ``model`` on ``exec()``.

    Args:
        model: The ``ComponentModel`` to serialise.
        file_path: Optional path to write the generated code to.
        model_var_name: Variable name for the model in the generated code.

    Returns:
        The generated Python source as a string.
    """
    if not isinstance(model, ComponentModel):
        raise TypeError(
            f"component_model_to_code expects a ComponentModel, "
            f"got {type(model).__name__}."
        )

    dispenser = _NameDispenser()
    lines: list = []

    # ----- Header banner + imports -----
    lines.append("####################")
    lines.append("# COMPONENT MODEL  #")
    lines.append("####################")
    lines.append("")

    needed_imports = _collect_imports(model)
    # Wrap the import line at ~80 chars (PEP 8 friendly when short, multi-line
    # parenthesised import when long).
    if len(needed_imports) <= 4:
        joined = ", ".join(needed_imports)
        lines.append(
            f"from besser.BUML.metamodel.uml_component import {joined}"
        )
    else:
        lines.append("from besser.BUML.metamodel.uml_component import (")
        for symbol in needed_imports:
            lines.append(f"    {symbol},")
        lines.append(")")
    lines.append("")

    # ----- Model constructor -----
    model_class = type(model).__name__
    lines.append(
        f"{model_var_name} = {model_class}("
        f"name='{_escape_python_string(model.name)}')"
    )
    lines.append("")

    # ----- Components (Step 3) — sorted by Element.timestamp for determinism.
    components_sorted = sort_by_timestamp(model.components)
    if components_sorted:
        lines.append("# --- Components ---")
        for component in components_sorted:
            var = dispenser.name_for(component)
            lines.extend(_emit_component(component, var, dispenser))
            lines.append(f"{model_var_name}.add_component({var})")
            lines.append("")

    # ----- Subsystem.children (Step 4) — deferred; all components exist now.
    subsystems = [c for c in components_sorted if isinstance(c, Subsystem)]
    has_children = any(s.children for s in subsystems)
    if has_children:
        lines.append("# --- Subsystem containment ---")
        for subsystem in subsystems:
            if not subsystem.children:
                continue
            s_var = dispenser.name_for(subsystem)
            for child in sort_by_timestamp(subsystem.children):
                c_var = dispenser.name_for(child)
                lines.append(f"{s_var}.add_child({c_var})")
        lines.append("")

    # ----- Interfaces (Step 5).
    interfaces_sorted = sort_by_timestamp(model.interfaces)
    if interfaces_sorted:
        lines.append("# --- Interfaces ---")
        for interface in interfaces_sorted:
            var = dispenser.name_for(interface)
            lines.extend(_emit_interface(interface, var))
            lines.append(f"{model_var_name}.add_interface({var})")
            lines.append("")

    # ----- Permissions (Step 6) -- AgenticComponentModel only.
    permissions_sorted = sort_by_timestamp(getattr(model, "permissions", set()))
    if permissions_sorted:
        lines.append("# --- Permissions ---")
        for permission in permissions_sorted:
            var = dispenser.name_for(permission)
            lines.extend(_emit_permission(permission, var))
            lines.append(f"{model_var_name}.add_permission({var})")
            lines.append("")

    # ----- Relationships (Step 7).
    relationships_sorted = sort_by_timestamp(model.relationships)
    if relationships_sorted:
        lines.append("# --- Relationships ---")
        for rel in relationships_sorted:
            lines.extend(_emit_relationship(rel, dispenser))
            lines.append("")

    source = "\n".join(line for line in lines if line is not None) + "\n"

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(source)
    return source
