"""PlatformCustomizationModel → JSON converter.

Mirrors `platform_customization_processor.py`. Round-trips are expected to be
identity modulo default-value elision (we skip omitted fields instead of
emitting explicit `null`s so diffs stay clean).
"""

from enum import Enum
from typing import Any, Dict, Optional

from besser.BUML.metamodel.platform_customization import (
    AssociationCustomization,
    ClassCustomization,
    DiagramCustomization,
    PlatformCustomizationModel,
)


def _enum_value(value: Optional[Enum]) -> Optional[str]:
    return value.value if value is not None else None


def _set_if(out: Dict[str, Any], key: str, value: Any) -> None:
    """Add `value` to `out[key]` only if it's not None."""
    if value is not None:
        out[key] = value


def _class_customization_to_json(cust: ClassCustomization) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if cust.is_container:
        out["isContainer"] = True
    _set_if(out, "defaultWidth", cust.default_width)
    _set_if(out, "defaultHeight", cust.default_height)
    _set_if(out, "nodeShape", _enum_value(cust.node_shape))
    _set_if(out, "fillColor", cust.fill_color)
    _set_if(out, "borderColor", cust.border_color)
    _set_if(out, "borderWidth", cust.border_width)
    _set_if(out, "borderStyle", _enum_value(cust.border_style))
    _set_if(out, "borderRadius", cust.border_radius)
    _set_if(out, "fontSize", cust.font_size)
    _set_if(out, "fontWeight", _enum_value(cust.font_weight))
    _set_if(out, "fontColor", cust.font_color)
    _set_if(out, "labelPosition", _enum_value(cust.label_position))
    return out


def _association_customization_to_json(cust: AssociationCustomization) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    _set_if(out, "edgeColor", cust.edge_color)
    _set_if(out, "lineWidth", cust.line_width)
    _set_if(out, "lineStyle", _enum_value(cust.line_style))
    _set_if(out, "sourceArrowStyle", _enum_value(cust.source_arrow_style))
    _set_if(out, "targetArrowStyle", _enum_value(cust.target_arrow_style))
    _set_if(out, "labelVisible", cust.label_visible)
    _set_if(out, "labelFontSize", cust.label_font_size)
    _set_if(out, "labelFontColor", cust.label_font_color)
    return out


def _diagram_customization_to_json(diagram: DiagramCustomization) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    _set_if(out, "backgroundColor", diagram.background_color)
    _set_if(out, "gridVisible", diagram.grid_visible)
    _set_if(out, "gridSize", diagram.grid_size)
    _set_if(out, "snapToGrid", diagram.snap_to_grid)
    _set_if(out, "theme", _enum_value(diagram.theme))
    return out


def platform_customization_to_json(model: PlatformCustomizationModel) -> Dict[str, Any]:
    """Convert a `PlatformCustomizationModel` to the Platform-tab JSON shape."""
    class_overrides: Dict[str, Dict[str, Any]] = {}
    for class_name, cust in model.class_overrides.items():
        payload = _class_customization_to_json(cust)
        if payload:
            class_overrides[class_name] = payload

    association_overrides: Dict[str, Dict[str, Any]] = {}
    for assoc_name, cust in model.association_overrides.items():
        payload = _association_customization_to_json(cust)
        if payload:
            association_overrides[assoc_name] = payload

    result: Dict[str, Any] = {
        "classOverrides": class_overrides,
        "associationOverrides": association_overrides,
    }

    if model.diagram_customization is not None:
        diagram_payload = _diagram_customization_to_json(model.diagram_customization)
        if diagram_payload:
            result["diagramCustomization"] = diagram_payload

    return result
