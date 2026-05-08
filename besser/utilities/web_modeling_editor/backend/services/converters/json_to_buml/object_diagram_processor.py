"""
Object diagram processing for converting v4 JSON to BUML format.

Reads the v4 wire shape (``{nodes, edges}``) natively. The v4 spec
collapses ``ObjectAttribute`` rows onto the parent ``objectName`` node's
``data.attributes`` array; we walk that directly. ``referenceDiagramData``
arrives in v4 too (and is consumed by walking its ``nodes``/``edges``
likewise).
"""

import logging

from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError
from besser.BUML.metamodel.object import ObjectModel

logger = logging.getLogger(__name__)
from besser.BUML.metamodel.object.builder import ObjectBuilder
from besser.BUML.metamodel.object import Link, LinkEnd
from besser.BUML.metamodel.structural import Metadata
from besser.utilities.web_modeling_editor.backend.services.converters.parsers import parse_attribute
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml._node_helpers import (
    node_data,
)
from datetime import datetime, timedelta


def get_all_attributes(cls, domain_model):
    """Get all attributes including inherited ones via recursive traversal."""
    attrs = set(cls.attributes)
    for gen in domain_model.generalizations:
        if gen.specific == cls:
            parent_attrs = get_all_attributes(gen.general, domain_model)
            attrs.update(parent_attrs)
    return attrs


def parse_datetime_value(value, type_name):
    """Parse datetime values from string format."""
    try:
        if type_name in ['datetime', 'DateTimeType']:
            if 'T' in value:
                if len(value) == 16:
                    return datetime.strptime(value, '%Y-%m-%dT%H:%M')
                elif len(value) == 19:
                    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')
                else:
                    clean_value = value.split('.')[0].split('+')[0].split('Z')[0]
                    if len(clean_value) == 16:
                        return datetime.strptime(clean_value, '%Y-%m-%dT%H:%M')
                    else:
                        return datetime.strptime(clean_value, '%Y-%m-%dT%H:%M:%S')
            else:
                try:
                    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    return datetime.strptime(value, '%Y-%m-%d %H:%M')
        elif type_name in ['date', 'DateType']:
            return datetime.strptime(value, '%Y-%m-%d').date()
        elif type_name in ['time', 'TimeType']:
            if len(value.split(':')) == 2:
                return datetime.strptime(value, '%H:%M').time()
            else:
                return datetime.strptime(value, '%H:%M:%S').time()
        elif type_name in ['timedelta', 'TimeDeltaType']:
            if 'day' in value:
                parts = value.split(',')
                days = int(parts[0].split()[0])
                time_part = parts[1].strip() if len(parts) > 1 else "0:00:00"
                time_components = time_part.split(':')
                if len(time_components) == 2:
                    hours, minutes = map(int, time_components)
                    return timedelta(days=days, hours=hours, minutes=minutes)
                else:
                    hours, minutes, seconds = map(int, time_components)
                    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
            else:
                time_components = value.split(':')
                if len(time_components) == 2:
                    hours, minutes = map(int, time_components)
                    return timedelta(hours=hours, minutes=minutes)
                else:
                    hours, minutes, seconds = map(int, time_components)
                    return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except (ValueError, IndexError):
        logger.warning("Could not parse %s value", type_name)
        return value

    return value


def _reference_class_name(class_node_id: str, reference_data: dict) -> str:
    """Look up a class node's name in v4 reference diagram data."""
    if not isinstance(reference_data, dict):
        return ""
    for node in reference_data.get("nodes") or []:
        if node.get("id") != class_node_id:
            continue
        if node.get("type") == "class":
            return (node.get("data") or {}).get("name", "")
    return ""


def _reference_association_name(assoc_id: str, reference_data: dict) -> str:
    """Look up an association edge's name in v4 reference diagram data."""
    if not isinstance(reference_data, dict):
        return ""
    for edge in reference_data.get("edges") or []:
        if edge.get("id") != assoc_id:
            continue
        return (edge.get("data") or {}).get("name", "")
    return ""


def process_object_diagram(json_data, domain_model):
    """Process an Object Diagram in the v4 wire shape and return an ObjectModel."""
    if domain_model is None:
        raise ConversionError(
            "Object diagram processing requires a reference class diagram (domain model). "
            "Please ensure a ClassDiagram is provided or linked as reference data."
        )

    title = json_data.get('title', 'Generated_Object_Model')
    if ' ' in title:
        title = title.replace(' ', '_')

    object_model = ObjectModel(title)
    model_data = json_data.get('model', {}) or {}
    nodes = model_data.get('nodes') or []
    edges = model_data.get('edges') or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    reference_data = model_data.get('referenceDiagramData', {}) or {}

    # Track objects by node id for link creation.
    objects_by_id = {}

    comment_nodes = {}
    comment_links = {}

    for node in nodes:
        node_type = node.get("type")
        node_id = node.get("id")
        data = node_data(node)

        if node_type == "Comments":
            comment_nodes[node_id] = (data.get("name") or "").strip()
            continue

        if node_type not in ("objectName", "UserModelName"):
            continue

        object_name = data.get("name", "") or ""
        class_id = data.get("classId")

        class_obj = None
        class_name = None
        if class_id and reference_data:
            class_name = _reference_class_name(class_id, reference_data)
        if not class_name:
            class_name = data.get("className")
        if class_name:
            class_obj = domain_model.get_class_by_name(class_name)
        if not class_obj:
            class_obj = domain_model.get_class_by_name(object_name)
        if not class_obj:
            raise ConversionError(
                f"Could not find class for object '{object_name}' with class ID '{class_id}'. "
                "Ensure either reference diagram data or explicit class names are provided."
            )

        builder = ObjectBuilder(class_obj).name(object_name)

        attributes_dict = {}
        for attr in data.get("attributes") or []:
            attr_string = attr.get("name", "") or ""
            attr_name = None
            value = None
            if node_type == "objectName":
                # Format: "name = value"
                if " = " in attr_string:
                    attr_part, value_part = attr_string.split(" = ", 1)
                    value = value_part.strip()
                    attr_string = attr_part.strip()
                try:
                    _, attr_name, _ = parse_attribute(attr_string, domain_model)
                except Exception as e:
                    logger.warning(
                        "Could not process attribute '%s' for object '%s': %s",
                        attr_string, object_name, e,
                    )
                    continue
            else:  # UserModelName
                operator_str = attr.get("attributeOperator", "==")
                if operator_str and operator_str in attr_string:
                    attr_part, value_part = attr_string.split(operator_str, 1)
                    attr_name = attr_part.strip()
                    value = value_part.strip()
                else:
                    attr_name = attr_string.strip()
                    value = attr.get("attributeValue")

            if attr_name and value is not None:
                property_obj = None
                all_attrs = get_all_attributes(class_obj, domain_model)
                for prop in all_attrs:
                    if prop.name == attr_name:
                        property_obj = prop
                        break

                if property_obj:
                    converted_value = value
                    if hasattr(property_obj.type, 'literals'):
                        try:
                            for literal in property_obj.type.literals:
                                if literal.name == value:
                                    converted_value = literal
                                    break
                            else:
                                converted_value = getattr(property_obj.type, value)
                        except (AttributeError, StopIteration):
                            logger.warning(
                                "Enumeration literal '%s' not found in %s",
                                value, property_obj.type.name,
                            )
                            converted_value = value
                    elif hasattr(property_obj.type, 'name'):
                        type_name = (
                            property_obj.type.name
                            if hasattr(property_obj.type, 'name')
                            else str(property_obj.type)
                        )
                        if type_name in ['int', 'IntegerType']:
                            try:
                                converted_value = int(value)
                            except (TypeError, ValueError) as exc:
                                raise ConversionError(
                                    f"Object '{object_name}' (class '{class_obj.name}'): "
                                    f"attribute '{attr_name}' expects an integer, "
                                    f"but received {value!r}."
                                ) from exc
                        elif type_name in ['float', 'FloatType']:
                            try:
                                converted_value = float(value)
                            except (TypeError, ValueError) as exc:
                                raise ConversionError(
                                    f"Object '{object_name}' (class '{class_obj.name}'): "
                                    f"attribute '{attr_name}' expects a number, "
                                    f"but received {value!r}."
                                ) from exc
                        elif type_name in ['bool', 'BooleanType']:
                            converted_value = value.lower() in ['true', '1', 'yes']
                        elif type_name in ['datetime', 'DateTimeType', 'date', 'DateType', 'time', 'TimeType', 'timedelta', 'TimeDeltaType']:
                            converted_value = parse_datetime_value(value, type_name)
                    attributes_dict[attr_name] = converted_value

        if attributes_dict:
            builder = builder.attributes(**attributes_dict)

        try:
            obj = builder.build()
        except (TypeError, ValueError) as exc:
            raise ConversionError(
                f"Object '{object_name}' (class '{class_obj.name}'): {exc}"
            ) from exc
        logger.debug("Created object '%s' of class '%s'", object_name, class_obj.name)

        object_model.add_object(obj)
        objects_by_id[node_id] = obj

    # Process links (edges).
    for edge in edges:
        edge_type = edge.get("type")
        if edge_type == "Link":
            source_id = edge.get("source")
            target_id = edge.get("target")
            comment_id = None
            target = None
            if source_id in comment_nodes:
                comment_id = source_id
                target = target_id
            elif target_id in comment_nodes:
                comment_id = target_id
                target = source_id
            if comment_id and target:
                comment_links.setdefault(comment_id, []).append(target)
            continue

        if edge_type == "ObjectLink":
            source_id = edge.get("source")
            target_id = edge.get("target")
            edge_data = edge.get("data") or {}
            link_name = edge_data.get("name", "") or ""
            association_id = edge_data.get("associationId")

            source_obj = objects_by_id.get(source_id)
            target_obj = objects_by_id.get(target_id)
            if not source_obj or not target_obj:
                raise ConversionError(
                    f"Could not find objects for link '{link_name}'. Please ensure all objects in the link exist in the diagram."
                )

            association_obj = None
            if association_id:
                if hasattr(domain_model, 'association_by_id') and domain_model.association_by_id:
                    association_obj = domain_model.association_by_id.get(association_id)
                if not association_obj and reference_data:
                    assoc_name = _reference_association_name(association_id, reference_data)
                    if assoc_name:
                        for assoc in domain_model.associations:
                            if assoc.name == assoc_name:
                                association_obj = assoc
                                break

            if not association_obj:
                for assoc in domain_model.associations:
                    end_types = {end.type for end in assoc.ends}
                    if (source_obj.classifier in end_types and target_obj.classifier in end_types):
                        association_obj = assoc
                        break

            if association_obj:
                link_ends = []
                for end in association_obj.ends:
                    source_matches = False
                    if end.type == source_obj.classifier:
                        source_matches = True
                    else:
                        for gen in domain_model.generalizations:
                            if gen.specific == source_obj.classifier and gen.general == end.type:
                                source_matches = True
                                break
                    target_matches = False
                    if end.type == target_obj.classifier:
                        target_matches = True
                    else:
                        for gen in domain_model.generalizations:
                            if gen.specific == target_obj.classifier and gen.general == end.type:
                                target_matches = True
                                break

                    if source_matches:
                        link_end = LinkEnd(name=f"{end.name}_end", association_end=end, object=source_obj)
                        link_ends.append(link_end)
                    elif target_matches:
                        link_end = LinkEnd(name=f"{end.name}_end", association_end=end, object=target_obj)
                        link_ends.append(link_end)

                if len(link_ends) == 2:
                    link_display_name = link_name if link_name else f"{source_obj.name}_{target_obj.name}_link"
                    Link(name=link_display_name, association=association_obj, connections=link_ends)
                else:
                    raise ConversionError(
                        f"Expected 2 link ends but got {len(link_ends)} for link '{link_name}'. There may be an issue with the association structure."
                    )
            else:
                raise ConversionError(
                    f"Could not find association for link '{link_name}'. Please ensure all links correspond to valid associations in the class diagram."
                )

    # Apply comments.
    for comment_id, comment_text in comment_nodes.items():
        if comment_id in comment_links:
            for linked_id in comment_links[comment_id]:
                obj = objects_by_id.get(linked_id)
                if obj:
                    if not obj.classifier.metadata:
                        obj.classifier.metadata = Metadata(description=comment_text)
                    else:
                        if obj.classifier.metadata.description:
                            obj.classifier.metadata.description += f"\n{comment_text}"
                        else:
                            obj.classifier.metadata.description = comment_text
        else:
            if not object_model.metadata:
                object_model.metadata = Metadata(description=comment_text)
            else:
                if object_model.metadata.description:
                    object_model.metadata.description += f"\n{comment_text}"
                else:
                    object_model.metadata.description = comment_text

    return object_model
