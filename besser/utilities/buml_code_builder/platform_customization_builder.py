"""
Platform Customization Builder

Generates Python code from a `PlatformCustomizationModel` so the model can be
re-created via `exec()`. Mirrors the other top-level builders
(`gui_model_to_code`, `agent_model_to_code`, etc.).
"""

from enum import Enum
from typing import Iterable, List, Optional

from besser.BUML.metamodel.platform_customization import (
    AssociationCustomization,
    ClassCustomization,
    DiagramCustomization,
    PlatformCustomizationModel,
)
from besser.utilities.buml_code_builder.common import _escape_python_string


_CLASS_FIELDS = (
    ("is_container", False),
    ("default_width", None),
    ("default_height", None),
    ("node_shape", None),
    ("fill_color", None),
    ("border_color", None),
    ("border_width", None),
    ("border_style", None),
    ("border_radius", None),
    ("font_size", None),
    ("font_weight", None),
    ("font_color", None),
    ("label_position", None),
)

_ASSOC_FIELDS = (
    ("edge_color", None),
    ("line_width", None),
    ("line_style", None),
    ("source_arrow_style", None),
    ("target_arrow_style", None),
    ("label_visible", None),
    ("label_font_size", None),
    ("label_font_color", None),
    ("is_container_association", False),
)

_DIAGRAM_FIELDS = (
    ("background_color", None),
    ("grid_visible", None),
    ("grid_size", None),
    ("snap_to_grid", None),
    ("theme", None),
)


def _format_value(value) -> str:
    """Render a Python literal for one customization field."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, str):
        return f'"{_escape_python_string(value)}"'
    return repr(value)


def _format_dataclass(class_name: str, fields: Iterable[tuple], obj) -> str:
    """Render `ClassName(field=value, ...)` skipping defaults to keep diffs lean."""
    parts: List[str] = []
    for field_name, default in fields:
        value = getattr(obj, field_name)
        if value == default:
            continue
        parts.append(f"{field_name}={_format_value(value)}")
    if not parts:
        return f"{class_name}()"
    return f"{class_name}({', '.join(parts)})"


def _collect_used_enums(model: PlatformCustomizationModel) -> List[str]:
    """Return the enum class names that the emitted code will reference."""
    enums = set()
    for cust in model.class_overrides.values():
        for value in (
            cust.node_shape, cust.border_style, cust.font_weight, cust.label_position,
        ):
            if value is not None:
                enums.add(type(value).__name__)
    for cust in model.association_overrides.values():
        for value in (
            cust.line_style, cust.source_arrow_style, cust.target_arrow_style,
        ):
            if value is not None:
                enums.add(type(value).__name__)
    if model.diagram_customization is not None:
        if model.diagram_customization.theme is not None:
            enums.add(type(model.diagram_customization.theme).__name__)
    return sorted(enums)


def platform_customization_to_code(
    model: PlatformCustomizationModel,
    file_path: Optional[str] = None,
    model_var_name: str = "platform_customization",
) -> str:
    """Generate Python code that reconstructs `model`.

    Args:
        model: The PlatformCustomizationModel to serialize.
        file_path: Optional path to write the code to.
        model_var_name: Variable name for the emitted model (default
            `platform_customization`).

    Returns:
        The generated Python source as a string.
    """
    base_imports = [
        "PlatformCustomizationModel",
        "ClassCustomization",
        "AssociationCustomization",
    ]
    if model.diagram_customization is not None:
        base_imports.append("DiagramCustomization")
    used_enums = _collect_used_enums(model)
    all_imports = base_imports + used_enums

    lines: List[str] = []
    lines.append("##################################")
    lines.append("# PLATFORM CUSTOMIZATION MODEL #")
    lines.append("##################################")
    lines.append("")
    lines.append(
        "from besser.BUML.metamodel.platform_customization import (\n    "
        + ",\n    ".join(all_imports)
        + ",\n)"
    )
    lines.append("")

    name_safe = _escape_python_string(model.name)

    has_class_overrides = bool(model.class_overrides)
    has_assoc_overrides = bool(model.association_overrides)
    has_diagram = model.diagram_customization is not None

    if has_class_overrides:
        lines.append("class_overrides = {")
        for cls_name, cust in model.class_overrides.items():
            key = _escape_python_string(cls_name)
            lines.append(f'    "{key}": {_format_dataclass("ClassCustomization", _CLASS_FIELDS, cust)},')
        lines.append("}")
        lines.append("")

    if has_assoc_overrides:
        lines.append("association_overrides = {")
        for assoc_name, cust in model.association_overrides.items():
            key = _escape_python_string(assoc_name)
            lines.append(
                f'    "{key}": {_format_dataclass("AssociationCustomization", _ASSOC_FIELDS, cust)},'
            )
        lines.append("}")
        lines.append("")

    if has_diagram:
        lines.append(
            "diagram_customization = "
            + _format_dataclass("DiagramCustomization", _DIAGRAM_FIELDS, model.diagram_customization)
        )
        lines.append("")

    ctor_args = [f'name="{name_safe}"']
    if has_class_overrides:
        ctor_args.append("class_overrides=class_overrides")
    if has_assoc_overrides:
        ctor_args.append("association_overrides=association_overrides")
    if has_diagram:
        ctor_args.append("diagram_customization=diagram_customization")

    lines.append(f"{model_var_name} = PlatformCustomizationModel(")
    for i, arg in enumerate(ctor_args):
        suffix = "," if i < len(ctor_args) - 1 else ""
        lines.append(f"    {arg}{suffix}")
    lines.append(")")
    lines.append("")

    code = "\n".join(lines)

    if file_path:
        if not file_path.endswith(".py"):
            file_path += ".py"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

    return code
