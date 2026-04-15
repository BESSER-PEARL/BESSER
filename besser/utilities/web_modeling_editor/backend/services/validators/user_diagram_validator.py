from typing import Any, Dict, List, Optional, Set, Tuple


def _extract_elements_and_relationships(model_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    elements = model_data.get("elements", {}) if isinstance(model_data, dict) else {}
    relationships = model_data.get("relationships", {}) if isinstance(model_data, dict) else {}

    if not elements and isinstance(model_data.get("model"), dict):
        nested_model = model_data.get("model", {})
        elements = nested_model.get("elements", {})
        relationships = nested_model.get("relationships", {})

    if not isinstance(elements, dict):
        elements = {}
    if not isinstance(relationships, dict):
        relationships = {}

    return elements, relationships


def _resolve_element_class_name(
    element: Dict[str, Any],
    reference_elements: Dict[str, Any],
) -> Optional[str]:
    class_name = element.get("className")
    if isinstance(class_name, str) and class_name.strip():
        return class_name.strip()

    class_id = element.get("classId")
    if isinstance(class_id, str) and class_id:
        class_element = reference_elements.get(class_id, {})
        if isinstance(class_element, dict):
            referenced_name = class_element.get("name")
            if isinstance(referenced_name, str) and referenced_name.strip():
                return referenced_name.strip()

    return None


def validate_user_diagram_specific_rules(model_data: Dict[str, Any]) -> List[str]:
    elements, relationships = _extract_elements_and_relationships(model_data)

    reference_data = model_data.get("referenceDiagramData", {}) if isinstance(model_data, dict) else {}
    if not reference_data and isinstance(model_data.get("model"), dict):
        reference_data = model_data.get("model", {}).get("referenceDiagramData", {})
    reference_elements = reference_data.get("elements", {}) if isinstance(reference_data, dict) else {}
    if not isinstance(reference_elements, dict):
        reference_elements = {}

    object_elements: Dict[str, Dict[str, Any]] = {
        element_id: element
        for element_id, element in elements.items()
        if isinstance(element, dict) and element.get("type") in {"UserModelName", "ObjectName"}
    }

    if not object_elements:
        return ["User element is missing from the main model."]

    user_object_ids: Set[str] = set()
    for element_id, element in object_elements.items():
        class_name = _resolve_element_class_name(element, reference_elements)
        if class_name == "User":
            user_object_ids.add(element_id)

    if not user_object_ids:
        return ["User element is missing from the main model."]

    adjacency: Dict[str, Set[str]] = {element_id: set() for element_id in object_elements.keys()}
    for relationship in relationships.values():
        if not isinstance(relationship, dict):
            continue

        relationship_type = relationship.get("type")
        if relationship_type not in {"UserModelLink", "ObjectLink"}:
            continue

        source_id = relationship.get("source", {}).get("element")
        target_id = relationship.get("target", {}).get("element")
        if source_id in object_elements and target_id in object_elements:
            adjacency[source_id].add(target_id)
            adjacency[target_id].add(source_id)

    visited: Set[str] = set()
    stack = list(user_object_ids)
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(neighbor for neighbor in adjacency.get(current, set()) if neighbor not in visited)

    disconnected_ids = [element_id for element_id in object_elements.keys() if element_id not in visited]
    if disconnected_ids:
        first_disconnected_id = disconnected_ids[0]
        first_disconnected_element = object_elements[first_disconnected_id]
        first_disconnected_class_name = _resolve_element_class_name(
            first_disconnected_element,
            reference_elements,
        )
        element_label = first_disconnected_class_name or first_disconnected_element.get("name") or first_disconnected_id
        return [
            f"Make sure that every element is connected, element {element_label} is not linked to the main model."
        ]

    return []
