"""
Object converter module for BUML to JSON conversion.
Handles object diagram processing and attribute mapping.
"""

import ast
import uuid
from typing import Dict, Any, Optional

from besser.BUML.metamodel.object import ObjectModel, Object, Link
from besser.utilities.web_modeling_editor.backend.services.utils import (
    determine_connection_direction, calculate_connection_points,
    calculate_path_points, calculate_relationship_bounds
)


def object_buml_to_json(content: str, domain_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an object model Python file content to JSON format matching the frontend structure.
    
    Args:
        content: Object model Python code as string
        domain_json: Reference domain model JSON for class mapping
        
    Returns:
        Dictionary representing the object diagram in JSON format
    """
    elements = {}
    relationships = {}
    
    # Default diagram size
    default_size = {"width": 960, "height": 670}
    
    # Grid layout configuration for positioning objects
    grid_size = {
        "x_spacing": 250,
        "y_spacing": 180,
        "max_columns": 4,
    }
    
    current_column = 0
    current_row = 0
    
    def get_position():
        nonlocal current_column, current_row
        x = -460 + (current_column * grid_size["x_spacing"])
        y = -300 + (current_row * grid_size["y_spacing"])
        
        current_column += 1
        if current_column >= grid_size["max_columns"]:
            current_column = 0
            current_row += 1
            
        return x, y
    
    try:
        reference_diagram_json = domain_json

        # Create mapping from class names to class IDs in reference diagram
        class_name_to_id = {}
        class_id_to_attributes = {}
        
        for elem_id, elem in reference_diagram_json["elements"].items():
            if elem["type"] == "Class":
                class_name_to_id[elem["name"]] = elem_id
                # Store class attributes for object attribute mapping
                class_attributes = {}
                for attr_id in elem.get("attributes", []):
                    if attr_id in reference_diagram_json["elements"]:
                        attr_elem = reference_diagram_json["elements"][attr_id]
                        # Extract attribute name (remove visibility and type info)
                        attr_name = attr_elem["name"].split(":")[0].strip().lstrip("+-#~")
                        class_attributes[attr_name] = attr_id
                class_id_to_attributes[elem_id] = class_attributes
        
        # Parse the Python code to extract object instances
        tree = ast.parse(content)
        
        # Track objects and their information
        objects_by_name = {}
        object_class_mapping = {}
        object_comments = {}  # object_var -> comment_text
        om_comment = None  # ObjectModel metadata comment
        
        # Extract object instantiations using fluent API
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        
                        # Check if this is an object instantiation using fluent API
                        if isinstance(node.value, ast.Call):
                            # Look for pattern: ClassName("ObjectName").attributes(...).build()
                            call_chain = []
                            current = node.value
                            
                            # Walk back through the method chain
                            while isinstance(current, ast.Call):
                                if isinstance(current.func, ast.Attribute):
                                    call_chain.append(current.func.attr)
                                    current = current.func.value
                                elif isinstance(current.func, ast.Name):
                                    call_chain.append(current.func.id)
                                    break
                                else:
                                    break
                            
                            # Reverse to get the correct order
                            call_chain.reverse()
                            
                            # Check if this matches the fluent pattern
                            if len(call_chain) >= 3 and call_chain[-1] == "build":
                                class_name = call_chain[0]
                                if class_name in class_name_to_id:
                                    # Extract object name and attributes
                                    object_name = var_name
                                    object_instance_name = None
                                    attributes = {}
                                    
                                    # Get the initial call arguments for object name
                                    init_call = node.value
                                    while isinstance(init_call, ast.Call) and isinstance(init_call.func, ast.Attribute):
                                        init_call = init_call.func.value
                                    
                                    if isinstance(init_call, ast.Call) and len(init_call.args) > 0:
                                        if isinstance(init_call.args[0], ast.Constant):
                                            object_instance_name = init_call.args[0].value
                                    
                                    # Find attributes call in the chain
                                    attr_call = node.value
                                    while isinstance(attr_call, ast.Call):
                                        if isinstance(attr_call.func, ast.Attribute) and attr_call.func.attr == "attributes":
                                            # Extract attributes from keyword arguments
                                            for kw in attr_call.keywords:
                                                if isinstance(kw.value, ast.Constant):
                                                    attributes[kw.arg] = kw.value.value
                                            break
                                        attr_call = attr_call.func.value if isinstance(attr_call.func, ast.Attribute) else None
                                        if not attr_call:
                                            break
                                    
                                    objects_by_name[object_name] = {
                                        "class_name": class_name,
                                        "class_id": class_name_to_id[class_name],
                                        "instance_name": object_instance_name or object_name,
                                        "attributes": attributes
                                    }
                                    object_class_mapping[object_name] = class_name
                            
                            # Check for ObjectModel instantiation with metadata
                            elif call_chain and call_chain[0] == "ObjectModel":
                                # Extract ObjectModel metadata
                                for kw in node.value.keywords:
                                    if kw.arg == "metadata":
                                        if isinstance(kw.value, ast.Call):
                                            for meta_kw in kw.value.keywords:
                                                if meta_kw.arg == "description":
                                                    om_comment = ast.literal_eval(meta_kw.value)
                
                # Check for object.classifier.metadata = Metadata(...) patterns
                target = node.targets[0]
                if isinstance(target, ast.Attribute) and target.attr == "metadata":
                    # This could be: obj.classifier.metadata = ...
                    if isinstance(target.value, ast.Attribute) and target.value.attr == "classifier":
                        obj_var = target.value.value.id if isinstance(target.value.value, ast.Name) else None
                        if obj_var and isinstance(node.value, ast.Call):
                            for kw in node.value.keywords:
                                if kw.arg == "description":
                                    object_comments[obj_var] = ast.literal_eval(kw.value)
        
        # Create object elements in JSON format
        for obj_name, obj_info in objects_by_name.items():
            x, y = get_position()
            object_id = str(uuid.uuid4())
            
            # Create object attribute elements
            object_attribute_ids = []
            class_id = obj_info["class_id"]
            class_attributes = class_id_to_attributes.get(class_id, {})
            
            attr_y_offset = 30
            for attr_name, attr_value in obj_info["attributes"].items():
                attr_id = str(uuid.uuid4())
                object_attribute_ids.append(attr_id)
                
                # Format the value appropriately
                if isinstance(attr_value, str):
                    formatted_value = attr_value
                else:
                    formatted_value = str(attr_value)
                
                # Get the corresponding class attribute ID if available
                class_attr_id = class_attributes.get(attr_name, None)
                
                elements[attr_id] = {
                    "id": attr_id,
                    "name": f"+ {attr_name}: str = {formatted_value}",
                    "type": "ObjectAttribute",
                    "owner": object_id,
                    "bounds": {
                        "x": x + 0.5,
                        "y": y + attr_y_offset - 0.5,
                        "width": 199,
                        "height": 30
                    },
                    "attributeId": class_attr_id
                }
                attr_y_offset += 30
            
            # Calculate object height based on attributes
            object_height = max(70, 40 + len(object_attribute_ids) * 30)
            
            elements[object_id] = {
                "id": object_id,
                "name": obj_info["instance_name"],
                "type": "ObjectName",
                "owner": None,
                "bounds": {
                    "x": x,
                    "y": y,
                    "width": 200,
                    "height": object_height
                },
                "attributes": object_attribute_ids,
                "methods": [],
                "classId": class_id
            }
        
        # Look for object links/relationships
        association_id_map = {}
        for rel_id, rel in reference_diagram_json["relationships"].items():
            if rel["type"] in ["ClassBidirectional", "ClassUnidirectional"]:
                source_role = rel.get("source", {}).get("role", "")
                target_role = rel.get("target", {}).get("role", "")
                if source_role or target_role:
                    association_id_map[f"{source_role}-{target_role}"] = rel_id
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Look for assignments like: obj1.relationship = obj2
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        obj_name = target.value.id if isinstance(target.value, ast.Name) else None
                        relationship_name = target.attr
                        
                        if obj_name in objects_by_name:
                            # Get the target object
                            if isinstance(node.value, ast.Name) and node.value.id in objects_by_name:
                                target_obj = node.value.id
                                
                                # Create a link relationship
                                rel_id = str(uuid.uuid4())
                                
                                # Find the object IDs
                                source_id = None
                                target_id = None
                                
                                for elem_id, elem in elements.items():
                                    if elem["type"] == "ObjectName" and elem["name"] == objects_by_name[obj_name]["instance_name"]:
                                        source_id = elem_id
                                    if elem["type"] == "ObjectName" and elem["name"] == objects_by_name[target_obj]["instance_name"]:
                                        target_id = elem_id
                                
                                if source_id and target_id:
                                    # Find corresponding association ID
                                    assoc_id = None
                                    for key, aid in association_id_map.items():
                                        if relationship_name in key:
                                            assoc_id = aid
                                            break
                                    
                                    relationships[rel_id] = {
                                        "id": rel_id,
                                        "name": f"{relationship_name}",
                                        "type": "ObjectLink",
                                        "owner": None,
                                        "bounds": {
                                            "x": -260,
                                            "y": -315,
                                            "width": 300,
                                            "height": 80
                                        },
                                        "path": [
                                            {"x": 0, "y": 80},
                                            {"x": 40, "y": 80},
                                            {"x": 40, "y": 0},
                                            {"x": 300, "y": 0},
                                            {"x": 300, "y": 65}
                                        ],
                                        "source": {
                                            "direction": "Right",
                                            "element": source_id
                                        },
                                        "target": {
                                            "direction": "Topleft",
                                            "element": target_id
                                        },
                                        "isManuallyLayouted": False,
                                        "associationId": assoc_id
                                    }

        # Position for comments
        comment_x = -970
        comment_y = -300

        # Create comment elements from metadata
        # 1. ObjectModel comment (unlinked)
        if om_comment:
            comment_id = str(uuid.uuid4())
            elements[comment_id] = {
                "id": comment_id,
                "name": om_comment,
                "type": "Comments",
                "owner": None,
                "bounds": {
                    "x": comment_x,
                    "y": comment_y,
                    "width": 200,
                    "height": 100,
                },
            }
            comment_y += 130

        # 2. Object comments (linked to objects)
        object_var_to_id = {}  # Map object variable names to their element IDs
        for obj_name, obj_info in objects_by_name.items():
            # Find the object ID for this object
            for elem_id, elem in elements.items():
                if elem.get("type") == "ObjectName" and elem.get("name") == obj_info["instance_name"]:
                    object_var_to_id[obj_name] = elem_id
                    break

        for obj_var, comment_text in object_comments.items():
            if obj_var in object_var_to_id:
                comment_id = str(uuid.uuid4())
                object_id = object_var_to_id[obj_var]
                
                elements[comment_id] = {
                    "id": comment_id,
                    "name": comment_text,
                    "type": "Comments",
                    "owner": None,
                    "bounds": {
                        "x": comment_x,
                        "y": comment_y,
                        "width": 200,
                        "height": 100,
                    },
                }
                
                # Create Link relationship
                link_id = str(uuid.uuid4())
                source_element = elements[comment_id]
                target_element = elements[object_id]
                
                source_dir, target_dir = determine_connection_direction(
                    source_element["bounds"], target_element["bounds"]
                )
                
                source_point = calculate_connection_points(
                    source_element["bounds"], source_dir
                )
                target_point = calculate_connection_points(
                    target_element["bounds"], target_dir
                )
                
                path_points = calculate_path_points(
                    source_point, target_point, source_dir, target_dir
                )
                rel_bounds = calculate_relationship_bounds(path_points)
                
                relationships[link_id] = {
                    "id": link_id,
                    "name": "",
                    "type": "Link",
                    "owner": None,
                    "bounds": rel_bounds,
                    "path": path_points,
                    "source": {
                        "direction": source_dir,
                        "element": comment_id,
                        "bounds": {
                            "x": source_point["x"],
                            "y": source_point["y"],
                            "width": 0,
                            "height": 0,
                        },
                    },
                    "target": {
                        "direction": target_dir,
                        "element": object_id,
                        "bounds": {
                            "x": target_point["x"],
                            "y": target_point["y"],
                            "width": 0,
                            "height": 0,
                        },
                    },
                    "isManuallyLayouted": False,
                }
                
                comment_y += 130

        return {
            "version": "3.0.0",
            "type": "ObjectDiagram",
            "size": default_size,
            "interactive": {"elements": {}, "relationships": {}},
            "elements": elements,
            "relationships": relationships,
            "assessments": {},
            "referenceDiagramData": reference_diagram_json
        }
  
    except Exception as e:
        print(f"Error parsing object BUML content: {str(e)}")
        return {
            "version": "3.0.0",
            "type": "ObjectDiagram", 
            "size": default_size,
            "interactive": {"elements": {}, "relationships": {}},
            "elements": {},
            "relationships": {},
            "assessments": {},
        }
