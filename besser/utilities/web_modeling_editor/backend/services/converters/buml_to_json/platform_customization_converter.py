"""PlatformCustomizationModel → JSON converter.

Mirrors `platform_customization_processor.py`. Round-trips are expected to be
identity modulo default-value elision (we skip omitted fields instead of
emitting explicit `null`s so diffs stay clean).
"""

from typing import Any, Dict

from besser.BUML.metamodel.platform_customization import (
    AssociationCustomization,
    ClassCustomization,
    PlatformCustomizationModel,
)


def _class_customization_to_json(cust: ClassCustomization) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if cust.is_container:
        out["isContainer"] = True
    if cust.default_width is not None:
        out["defaultWidth"] = cust.default_width
    if cust.default_height is not None:
        out["defaultHeight"] = cust.default_height
    return out


def _association_customization_to_json(cust: AssociationCustomization) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if cust.edge_color is not None:
        out["edgeColor"] = cust.edge_color
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

    return {
        "classOverrides": class_overrides,
        "associationOverrides": association_overrides,
    }
