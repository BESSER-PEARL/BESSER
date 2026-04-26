"""JSON → PlatformCustomizationModel processor.

Expected JSON shape (from the "Platform" tab in the web editor):

    {
        "classOverrides": {
            "Region": {
                "isContainer": true, "defaultWidth": 400, "defaultHeight": 300,
                "nodeShape": "rounded_rect", "fillColor": "#fff", "borderColor": "#000",
                "borderWidth": 2, "borderStyle": "dashed", "borderRadius": 12,
                "fontSize": 14, "fontWeight": "bold", "fontColor": "#111",
                "labelPosition": "inside"
            }
        },
        "associationOverrides": {
            "has": {
                "edgeColor": "#22c55e", "lineWidth": 2, "lineStyle": "solid",
                "sourceArrowStyle": "none", "targetArrowStyle": "filled_triangle",
                "labelVisible": true, "labelFontSize": 11, "labelFontColor": "#111"
            }
        },
        "diagramCustomization": {
            "backgroundColor": "#fafafa", "gridVisible": true, "gridSize": 24,
            "snapToGrid": false, "theme": "auto"
        }
    }

All fields are optional. Missing or invalid fields fall back to None / defaults
so a v1-shape payload (only isContainer/defaultWidth/defaultHeight/edgeColor)
parses unchanged. Unknown extra keys are ignored.
"""

from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar

from besser.BUML.metamodel.platform_customization import (
    ArrowStyle,
    AssociationCustomization,
    ClassCustomization,
    DiagramCustomization,
    FontWeight,
    LabelPosition,
    LineStyle,
    NodeShape,
    PlatformCustomizationModel,
    Theme,
)


_E = TypeVar("_E", bound=Enum)


def _str_or_none(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _int_in_range_or_none(value: Any, lo: int, hi: int) -> Optional[int]:
    """Coerce to int and clamp acceptance to [lo, hi]; otherwise None."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < lo or parsed > hi:
        return None
    return parsed


def _enum_or_none(enum_cls: Type[_E], value: Any) -> Optional[_E]:
    if value is None or value == "":
        return None
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.strip())
        except ValueError:
            return None
    return None


def _positive_int_or_none(value: Any) -> Optional[int]:
    """Existing v1 helper kept for default_width / default_height."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_class_customization(data: Dict[str, Any]) -> ClassCustomization:
    return ClassCustomization(
        is_container=bool(data.get("isContainer", False)),
        default_width=_positive_int_or_none(data.get("defaultWidth")),
        default_height=_positive_int_or_none(data.get("defaultHeight")),
        node_shape=_enum_or_none(NodeShape, data.get("nodeShape")),
        fill_color=_str_or_none(data.get("fillColor")),
        border_color=_str_or_none(data.get("borderColor")),
        border_width=_int_in_range_or_none(data.get("borderWidth"), 0, 8),
        border_style=_enum_or_none(LineStyle, data.get("borderStyle")),
        border_radius=_int_in_range_or_none(data.get("borderRadius"), 0, 32),
        font_size=_int_in_range_or_none(data.get("fontSize"), 8, 24),
        font_weight=_enum_or_none(FontWeight, data.get("fontWeight")),
        font_color=_str_or_none(data.get("fontColor")),
        label_position=_enum_or_none(LabelPosition, data.get("labelPosition")),
    )


def _parse_association_customization(data: Dict[str, Any]) -> AssociationCustomization:
    return AssociationCustomization(
        edge_color=_str_or_none(data.get("edgeColor")),
        line_width=_int_in_range_or_none(data.get("lineWidth"), 1, 6),
        line_style=_enum_or_none(LineStyle, data.get("lineStyle")),
        source_arrow_style=_enum_or_none(ArrowStyle, data.get("sourceArrowStyle")),
        target_arrow_style=_enum_or_none(ArrowStyle, data.get("targetArrowStyle")),
        label_visible=_bool_or_none(data.get("labelVisible")),
        label_font_size=_int_in_range_or_none(data.get("labelFontSize"), 8, 18),
        label_font_color=_str_or_none(data.get("labelFontColor")),
        is_container_association=bool(data.get("isContainerAssociation", False)),
    )


def _parse_diagram_customization(data: Optional[Dict[str, Any]]) -> Optional[DiagramCustomization]:
    if not isinstance(data, dict):
        return None
    diagram = DiagramCustomization(
        background_color=_str_or_none(data.get("backgroundColor")),
        grid_visible=_bool_or_none(data.get("gridVisible")),
        grid_size=_int_in_range_or_none(data.get("gridSize"), 8, 64),
        snap_to_grid=_bool_or_none(data.get("snapToGrid")),
        theme=_enum_or_none(Theme, data.get("theme")),
    )
    # Skip the all-defaults case so round-tripping an empty diagram block
    # doesn't materialise a non-None DiagramCustomization on the model.
    if (
        diagram.background_color is None
        and diagram.grid_visible is None
        and diagram.grid_size is None
        and diagram.snap_to_grid is None
        and diagram.theme is None
    ):
        return None
    return diagram


def process_platform_customization_diagram(
    customization_diagram: Optional[Dict[str, Any]],
    name: str = "PlatformCustomization",
) -> PlatformCustomizationModel:
    """Convert the frontend JSON into a `PlatformCustomizationModel`.

    Args:
        customization_diagram: The JSON payload from the Platform tab, or
            `None` / `{}` to produce an empty (no-override) customization.
        name: Optional model name. NamedElement forbids spaces/hyphens; the
            default is a safe fallback.
    """

    if not customization_diagram:
        return PlatformCustomizationModel(name=name)

    model_data = customization_diagram.get("model", customization_diagram)
    raw_class_overrides = model_data.get("classOverrides") or {}
    raw_assoc_overrides = model_data.get("associationOverrides") or {}
    raw_diagram = model_data.get("diagramCustomization")

    class_overrides: Dict[str, ClassCustomization] = {}
    for class_name, data in raw_class_overrides.items():
        if not isinstance(class_name, str) or not class_name.strip():
            continue
        if not isinstance(data, dict):
            continue
        class_overrides[class_name] = _parse_class_customization(data)

    association_overrides: Dict[str, AssociationCustomization] = {}
    for assoc_name, data in raw_assoc_overrides.items():
        if not isinstance(assoc_name, str) or not assoc_name.strip():
            continue
        if not isinstance(data, dict):
            continue
        association_overrides[assoc_name] = _parse_association_customization(data)

    diagram_customization = _parse_diagram_customization(raw_diagram)

    return PlatformCustomizationModel(
        name=name,
        class_overrides=class_overrides,
        association_overrides=association_overrides,
        diagram_customization=diagram_customization,
    )
