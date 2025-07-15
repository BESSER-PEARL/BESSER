"""
Object diagram processing for converting JSON to BUML format.
"""

from fastapi import HTTPException
from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.object.builder import ObjectBuilder
from besser.BUML.metamodel.object import Link, LinkEnd
from besser.utilities.web_modeling_editor.backend.services.converters.parsers import parse_attribute
from datetime import datetime, date, time, timedelta
import re


def parse_datetime_value(value, type_name):
    """Parse datetime values from string format."""
    try:
        if type_name in ['datetime', 'DateTimeType']:
            # Try different datetime formats
            if 'T' in value:
                if len(value) == 16:  # Format: YYYY-MM-DDTHH:MM
                    return datetime.strptime(value, '%Y-%m-%dT%H:%M')
                elif len(value) == 19:  # Format: YYYY-MM-DDTHH:MM:SS
                    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')
                else:  # Try with milliseconds or other formats
                    # Remove milliseconds and timezone info for parsing
                    clean_value = value.split('.')[0].split('+')[0].split('Z')[0]
                    if len(clean_value) == 16:  # After cleaning, check if it's YYYY-MM-DDTHH:MM
                        return datetime.strptime(clean_value, '%Y-%m-%dT%H:%M')
                    else:
                        return datetime.strptime(clean_value, '%Y-%m-%dT%H:%M:%S')
            else:
                # Try other common formats
                try:
                    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Try without seconds
                    return datetime.strptime(value, '%Y-%m-%d %H:%M')
        elif type_name in ['date', 'DateType']:
            return datetime.strptime(value, '%Y-%m-%d').date()
        elif type_name in ['time', 'TimeType']:
            # Try different time formats
            if len(value.split(':')) == 2:  # Format: HH:MM
                return datetime.strptime(value, '%H:%M').time()
            else:  # Format: HH:MM:SS
                return datetime.strptime(value, '%H:%M:%S').time()
        elif type_name in ['timedelta', 'TimeDeltaType']:
            # Parse timedelta from string (e.g., "1 day, 2:30:00", "2:30:00", or "1:30")
            if 'day' in value:
                parts = value.split(',')
                days = int(parts[0].split()[0])
                time_part = parts[1].strip() if len(parts) > 1 else "0:00:00"
                time_components = time_part.split(':')
                if len(time_components) == 2:  # HH:MM format
                    hours, minutes = map(int, time_components)
                    return timedelta(days=days, hours=hours, minutes=minutes)
                else:  # HH:MM:SS format
                    hours, minutes, seconds = map(int, time_components)
                    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
            else:
                time_components = value.split(':')
                if len(time_components) == 2:  # HH:MM format
                    hours, minutes = map(int, time_components)
                    return timedelta(hours=hours, minutes=minutes)
                else:  # HH:MM:SS format
                    hours, minutes, seconds = map(int, time_components)
                    return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse {type_name} value '{value}': {e}")
        return value
    
    return value


def process_object_diagram(json_data, domain_model):
    """Process Object Diagram specific elements and return an ObjectModel."""
    title = json_data.get('title', 'Generated_Object_Model')
    if ' ' in title:
        title = title.replace(' ', '_')

    object_model = ObjectModel(title)
    # Get elements and relationships from the JSON data
    elements = json_data.get('model', {}).get('elements', {})
    relationships = json_data.get('model', {}).get('relationships', {})

     # If elements is empty, try the nested structure
    if not elements and 'model' in json_data.get('model', {}):
        elements = json_data.get('model', {}).get('elements', {})
        relationships = json_data.get('model', {}).get('relationships', {})

    # Track objects by their ID for link creation
    objects_by_id = {}

    # First pass: Create objects using fluent API
    for element_id, element in elements.items():
        if element.get("type") == "ObjectName":
            # Extract object name and class ID
            object_name = element.get("name", "")
            class_id = element.get("classId")

            # Find the corresponding class in the domain model using classId
            class_obj = None
            if class_id:
                # Get the reference data to find the class name by ID
                reference_data = json_data.get('model', {}).get('referenceDiagramData', {})

                if reference_data:
                    reference_elements = reference_data.get('elements', {})
                    class_element = reference_elements.get(class_id)
                    if class_element:
                        class_name = class_element.get("name", "")
                        class_obj = domain_model.get_class_by_name(class_name)
                        if not class_obj:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Class '{class_name}' with ID '{class_id}' not found in domain model. Please ensure the class diagram contains all required classes."
                            )
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Could not find class element with ID '{class_id}' in reference diagram data. Please verify the object diagram references are correct."
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="No reference diagram data found. Please ensure the class diagram is properly linked to the object diagram."
                    )

            # Fall back to searching by object name if class ID lookup fails
            if not class_obj:
                class_obj = domain_model.get_class_by_name(object_name)

            if not class_obj:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not find class for object '{object_name}' with class ID '{class_id}'. Please ensure all objects have corresponding classes in the class diagram."
                )

            # Create object using fluent API
            builder = ObjectBuilder(class_obj).name(object_name)

            # Process object attributes (slots) and add them to builder
            attributes_dict = {}
            for attr_id in element.get("attributes", []):
                attr_element = elements.get(attr_id)
                if attr_element and attr_element.get("type") == "ObjectAttribute":
                    attr_string = attr_element.get("name", "")

                    # Parse the attribute string to extract name, type, and value
                    # Format: "+ name: type = value"
                    value = None
                    if " = " in attr_string:
                        attr_part, value_part = attr_string.split(" = ", 1)
                        value = value_part.strip()
                        attr_string = attr_part.strip()
                    
                    # Parse the attribute definition
                    try:
                        visibility, attr_name, attr_type = parse_attribute(attr_string, domain_model)
                        if attr_name and value is not None:
                            # Find the corresponding property in the class or its parents
                            property_obj = None

                            # First check the class itself
                            for prop in class_obj.attributes:
                                if prop.name == attr_name:
                                    property_obj = prop
                                    break

                            # If not found, check parent classes (for inheritance)
                            if not property_obj:
                                for gen in domain_model.generalizations:
                                    if gen.specific == class_obj:
                                        for prop in gen.general.attributes:
                                            if prop.name == attr_name:
                                                property_obj = prop
                                                break
                                        if property_obj:
                                            break

                            if property_obj:
                                # Convert value to appropriate type
                                converted_value = value
                                
                                # Check if the property type is an enumeration
                                if hasattr(property_obj.type, 'literals'):  # This is an enumeration
                                    # Use the enumeration's __getattr__ method to get the literal
                                    # This will return the actual literal from the enumeration
                                    try:
                                        # Find the literal in the enumeration by name
                                        for literal in property_obj.type.literals:
                                            if literal.name == value:
                                                converted_value = literal
                                                break
                                        else:
                                            # If not found by iteration, try getattr as fallback
                                            converted_value = getattr(property_obj.type, value)
                                    except (AttributeError, StopIteration):
                                        # If literal not found, keep original value and warn
                                        print(f"Warning: Enumeration literal '{value}' not found in {property_obj.type.name}")
                                        converted_value = value
                                    
                                elif hasattr(property_obj.type, 'name'):
                                    type_name = property_obj.type.name if hasattr(property_obj.type, 'name') else str(property_obj.type)
                                    if type_name in ['int', 'IntegerType']:
                                        try:
                                            converted_value = int(value)
                                        except ValueError:
                                            converted_value = value
                                    elif type_name in ['float', 'FloatType']:
                                        try:
                                            converted_value = float(value)
                                        except ValueError:
                                            converted_value = value
                                    elif type_name in ['bool', 'BooleanType']:
                                        converted_value = value.lower() in ['true', '1', 'yes']
                                    elif type_name in ['datetime', 'DateTimeType', 'date', 'DateType', 'time', 'TimeType', 'timedelta', 'TimeDeltaType']:
                                        converted_value = parse_datetime_value(value, type_name)
                                
                                attributes_dict[attr_name] = converted_value

                    except Exception as e:
                        print(f"Warning: Could not process attribute '{attr_string}' for object '{object_name}': {e}")
                        continue

            # Add attributes to builder if any were found
            if attributes_dict:
                builder = builder.attributes(**attributes_dict)

            # Build the object
            obj = builder.build()
            # print(f"Created object '{object_name}' of class '{class_obj.name}'")

            # Add the object to the model and track it
            object_model.add_object(obj)
            objects_by_id[element_id] = obj

    # Second pass: Create links between objects
    for rel_id, relationship in relationships.items():
        if relationship.get("type") == "ObjectLink":
            source_id = relationship.get("source", {}).get("element")
            target_id = relationship.get("target", {}).get("element")
            link_name = relationship.get("name", "")
            association_id = relationship.get("associationId")

            source_obj = objects_by_id.get(source_id)
            target_obj = objects_by_id.get(target_id)
            
            if not source_obj or not target_obj:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not find objects for link '{link_name}'. Please ensure all objects in the link exist in the diagram."
                )
          # Find the corresponding association in the domain model
            association_obj = None
            if association_id:
                # First try to find the association directly by ID from the domain model
                if hasattr(domain_model, 'association_by_id') and domain_model.association_by_id:
                    association_obj = domain_model.association_by_id.get(association_id)

                # If not found by direct ID lookup, try the reference diagram approach
                if not association_obj:
                    # Look for the association by ID in the reference diagram data
                    reference_data = json_data.get('model', {}).get('referenceDiagramData', {})
                    if reference_data:
                        reference_relationships = reference_data.get('relationships', {})
                        assoc_element = reference_relationships.get(association_id)
                        if assoc_element:
                            # Found association in reference data
                            assoc_name = assoc_element.get("name", "")
                            # Only try to find by name if the association name is not empty
                            if assoc_name:
                                # Try to find the association by name
                                for assoc in domain_model.associations:
                                    if assoc.name == assoc_name:
                                        association_obj = assoc
                                        break

            # If still not found, try to find association by matching the connected classes
            if not association_obj:
                for assoc in domain_model.associations:
                    # Check if this association connects the right classes
                    end_types = [end.type for end in assoc.ends]
                    end_type_names = [end_type.name for end_type in end_types]
                    if (source_obj.classifier in end_types and target_obj.classifier in end_types):
                        association_obj = assoc
                        break

            if association_obj:
                # Create link using association ends
                link_ends = []
                for end in association_obj.ends:
                    # Check if end type matches source object (considering inheritance)
                    source_matches = False
                    if end.type == source_obj.classifier:
                        source_matches = True
                    else:
                        # Check if source object's class inherits from end type
                        for gen in domain_model.generalizations:
                            if gen.specific == source_obj.classifier and gen.general == end.type:
                                source_matches = True
                                break

                    # Check if end type matches target object (considering inheritance)
                    target_matches = False
                    if end.type == target_obj.classifier:
                        target_matches = True
                    else:
                        # Check if target object's class inherits from end type
                        for gen in domain_model.generalizations:
                            if gen.specific == target_obj.classifier and gen.general == end.type:
                                target_matches = True
                                break

                    if source_matches:
                        link_end = LinkEnd(name=f"{end.name}_end", association_end=end, object=source_obj)
                        link_ends.append(link_end)
                        # print(f"  -> Created link end for source: {end.name}_end")
                    elif target_matches:
                        link_end = LinkEnd(name=f"{end.name}_end", association_end=end, object=target_obj)
                        link_ends.append(link_end)

                if len(link_ends) == 2:
                    # Use link_name if provided, otherwise generate a default name
                    link_display_name = link_name if link_name else f"{source_obj.name}_{target_obj.name}_link"
                    link = Link(name=link_display_name, association=association_obj, connections=link_ends)
                    # Links are automatically added to objects via the Link constructor
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error: Expected 2 link ends but got {len(link_ends)} for link '{link_name}'. There may be an issue with the association structure."
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not find association for link '{link_name}'. Please ensure all links correspond to valid associations in the class diagram."
                )
    return object_model
