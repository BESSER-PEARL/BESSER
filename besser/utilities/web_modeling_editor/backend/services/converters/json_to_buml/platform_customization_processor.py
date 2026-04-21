"""JSON → PlatformCustomizationModel processor.

Expected JSON shape (from the "Platform" tab in the web editor):

    {
        "classOverrides": {
            "Region": { "isContainer": true, "defaultWidth": 400, "defaultHeight": 300 },
            "Sensor": { "defaultWidth": 80, "defaultHeight": 80 }
        },
        "associationOverrides": {
            "has": { "edgeColor": "#22c55e" }
        }
    }

Missing fields default to their `ClassCustomization` / `AssociationCustomization`
defaults. Unknown extra fields are ignored so the converter tolerates frontend
shapes that add new knobs before the backend catches up.
"""

from typing import Any, Dict, Optional

from besser.BUML.metamodel.platform_customization import (
    AssociationCustomization,
    ClassCustomization,
    PlatformCustomizationModel,
)


def _parse_class_customization(data: Dict[str, Any]) -> ClassCustomization:
    def _int_or_none(value):
        if value is None or value == "":
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    return ClassCustomization(
        is_container=bool(data.get("isContainer", False)),
        default_width=_int_or_none(data.get("defaultWidth")),
        default_height=_int_or_none(data.get("defaultHeight")),
    )


def _parse_association_customization(data: Dict[str, Any]) -> AssociationCustomization:
    raw_color = data.get("edgeColor")
    color: Optional[str] = None
    if isinstance(raw_color, str) and raw_color.strip():
        color = raw_color.strip()
    return AssociationCustomization(edge_color=color)


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

    return PlatformCustomizationModel(
        name=name,
        class_overrides=class_overrides,
        association_overrides=association_overrides,
    )
