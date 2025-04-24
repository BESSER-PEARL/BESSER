"""
Module for converting Draw.io structural UML diagrams to B-UML models.

This module provides functionality to parse Draw.io files containing UML class diagrams
and convert them into B-UML model representations. It handles classes, enumerations,
associations, generalizations, attributes, methods and their relationships.
"""

import os
import re
import xml.etree.ElementTree as ET
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Multiplicity, BinaryAssociation,
    Enumeration, EnumerationLiteral, Generalization, Method, Parameter, StringType, IntegerType, FloatType, BooleanType, TimeType, DateType, DateTimeType, TimeDeltaType
)
from besser.utilities import domain_model_to_code

# Map primitive type strings to their corresponding type classes
PRIMITIVE_TYPE_MAPPING = {
    'str': StringType,
    'string': StringType, 
    'int': IntegerType,
    'integer': IntegerType,
    'float': FloatType,
    'bool': BooleanType,
    'boolean': BooleanType,
    'time': TimeType,
    'date': DateType,
    'datetime': DateTimeType,
    'timedelta': TimeDeltaType
}

def structural_drawio_to_buml(drawio_file_path: str, buml_file_path: str = None) -> DomainModel:
    """
    Transform a Draw.io structural model into a B-UML model.

    Args:
        drawio_file_path: Path to the Draw.io file containing the UML diagram
        buml_file_path: Path for the output B-UML model file (default: None)

    Returns:
        DomainModel: The generated B-UML model object
    """
    buml_model, _ = generate_buml_from_xml(drawio_file_path)

    if buml_file_path:
        # Save model to Python file
        domain_model_to_code(buml_model, buml_file_path)

    return buml_model

def clean_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Input text containing HTML tags

    Returns:
        Text with HTML tags removed
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).strip()

def calculate_absolute_position(edge: dict, label_geometry: ET.Element) -> tuple:
    """
    Calculate absolute position of a label on an edge.

    Args:
        edge: Dictionary containing edge source and target coordinates
        label_geometry: XML element with label geometry information

    Returns:
        Tuple of (x, y) absolute coordinates
    """
    source_x, source_y = edge['source_x'], edge['source_y']
    target_x, target_y = edge['target_x'], edge['target_y']
    relative_x = float(label_geometry.get('x', 0))
    relative_y = float(label_geometry.get('y', 0))

    abs_x = source_x + (target_x - source_x) * relative_x
    abs_y = source_y + (target_y - source_y) * relative_x + relative_y

    offset = label_geometry.find(".//mxPoint[@as='offset']")
    if offset is not None:
        offset_x = float(offset.get('x', 0))
        offset_y = float(offset.get('y', 0))
        abs_x += offset_x
        abs_y += offset_y

    return abs_x, abs_y

def extract_classes_from_drawio(drawio_file: str) -> tuple:
    """
    Extract UML elements from Draw.io XML file.

    Args:
        drawio_file: Path to Draw.io file

    Returns:
        Tuple containing:
        - classes: Dict mapping class names to attributes/methods
        - enumerations: Dict mapping enum names to literals
        - associations: List of association dictionaries
        - generalizations: List of generalization dictionaries  
        - cells: Dict mapping cell IDs to cell data
    """
    try:
        tree = ET.parse(drawio_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return {}, [], {}, {}, {}

    # Group model elements
    model_elements = {
        'classes': {},
        'enumerations': {},
        'associations': [],
        'generalizations': []
    }

    # Group graph data - simplified, no positions needed
    graph_data = {
        'cells': {}
    }

    # First pass: collect cells
    for cell in root.findall(".//mxCell"):
        cell_id = cell.get('id')
        value = cell.get('value', '')
        style = cell.get('style', '')
        parent = cell.get('parent')

        # Store basic cell data
        graph_data['cells'][cell_id] = {
            'value': value,
            'style': style,
            'parent': parent
        }

        # Check for enumeration in swimlane format
        if style and 'swimlane' in style and is_enumeration(value):
            enum_name = clean_enumeration_name(value.split('\n')[1] if '\n' in value else value)
            if enum_name:
                model_elements['enumerations'][enum_name] = []
                # Find all child cells (literals)
                for literal_cell in root.findall(f".//mxCell[@parent='{cell_id}']"):
                    literal_value = literal_cell.get('value', '')
                    if literal_value and not is_enumeration(literal_value):
                        clean_literal = clean_html_tags(literal_value).strip()
                        if clean_literal:
                            model_elements['enumerations'][enum_name].append(clean_literal)
            continue

        # Skip processing as class if it's an enumeration
        if value and is_enumeration(value):
            enum_name = ""
            literals = []

            # Extract enum name
            if "<div>" in value:
                parts = value.split("<div>")
                for part in parts:
                    clean_part = clean_html_tags(part).strip()
                    if clean_part and not is_enumeration(clean_part):
                        enum_name = clean_part
                        break
            elif "<br>" in value:
                parts = value.split("<br>")
                enum_name = clean_enumeration_name(parts[1].strip())
            else:
                enum_name = clean_enumeration_name(value)

            if enum_name:
                model_elements['enumerations'][enum_name] = []
                # Extract literals from child cells
                for literal_cell in root.findall(f".//mxCell[@parent='{cell_id}']"):
                    literal_value = literal_cell.get('value', '')
                    if literal_value:
                        # Use the modified extract_literals_from_html function
                        literals = extract_literals_from_html(literal_value)
                        for literal in literals:
                            if literal and not is_enumeration(literal):
                                model_elements['enumerations'][enum_name].append(literal)
            else:
                print(f"Warning: Enum found without a name in value: {value}")
            continue

        # Process classes (only if not an enumeration)
        if value and not is_enumeration(value):
            if is_class_style(style, value):
                class_name = extract_class_name(value)
                if not class_name:
                    continue
                elif class_name.lower() == "class":
                    class_name = f"_{class_name}"

                model_elements['classes'][class_name] = {
                    'attributes': [],
                    'methods': []
                }

                # Parse attributes and methods based on format
                if value.startswith("<p") and "margin-top:4px" in value:
                    # Handle new HTML format
                    sections = value.split('<hr size="1"')
                    for section in sections[1:]:  # Skip the class name section
                        if '<p style="margin:0px;margin-left:4px;">' in section:
                            field = clean_html_tags(re.search(r'<p[^>]*>(.*?)</p>', section).group(1))
                            if field:
                                if '(' in field:  # Method
                                    visibility = "+" if field.startswith("+ ") else "-" \
                                        if field.startswith("- ") else "+"
                                    method_str = field.lstrip("+ -")
                                    method_name, params_str, return_type = process_method_string(method_str)
                                    if method_name:
                                        parameters = parse_parameters(params_str)
                                        model_elements['classes'][class_name]['methods'].append(
                                            (visibility, method_name, parameters, return_type)
                                        )
                                else:  # Attribute
                                    visibility = "+" if field.startswith("+ ") else "-" \
                                        if field.startswith("- ") else "+"
                                    attr_str = field.lstrip("+ -")
                                    if ':' in attr_str:
                                        name, type_str = attr_str.split(':', 1)
                                        model_elements['classes'][class_name]['attributes'].append(
                                            (visibility, name.strip(), type_str.strip())
                                        )
                                    else:
                                        name = attr_str.strip()
                                        model_elements['classes'][class_name]['attributes'].append(
                                            (visibility, name, "str")
                                        )
                else:
                    # Original HTML format handling (unchanged)
                    if value.startswith("<p") and "<b>" in value:
                        class_match = re.search(r'<b>(.*?)</b>', value)
                        class_name = clean_html_tags(class_match.group(1)) if class_match else None
                        if not class_name:
                            raise ValueError(f"Invalid class name: '{class_name}'. Class name cannot be empty. Skipping class")
                        elif class_name.lower() == "class":
                            class_name = f"_{class_name}"

                            model_elements['classes'][class_name] = {
                                'attributes': [],
                                'methods': []
                            }
                            #print(f"\nProcessing class: {class_name}")
                            # Parse attributes and methods from HTML
                            field_matches = re.finditer(r'<p[^>]*>([^<]*)</p>', value)
                            for match in field_matches:
                                field = clean_html_tags(match.group(1).strip())
                                if field and field != class_name:  # Skip class name
                                    if '(' in field:  # Method
                                        visibility = "+" if field.startswith("+ ") else "-" \
                                            if field.startswith("- ") else "+"
                                        method_str = field.lstrip("+ -")
                                        method_name, params_str, return_type = process_method_string(method_str)
                                        if method_name:
                                            parameters = parse_parameters(params_str)
                                            model_elements['classes'][class_name]['methods'].append(
                                                (visibility, method_name, parameters, return_type)
                                            )

                                    else:  # Attribute
                                        visibility = "+" if field.startswith("+ ") else "-" \
                                            if field.startswith("- ") else "+"
                                        attr_str = field.lstrip("+ -")
                                        if ':' in attr_str:
                                            name, type_str = attr_str.split(':', 1)
                                            model_elements['classes'][class_name]['attributes'].append(
                                                (visibility, name.strip(), type_str.strip())
                                            )
                                        else:
                                            # Attribute with only a name, no type
                                            name = attr_str.strip()
                                            model_elements['classes'][class_name]['attributes'].append(
                                                (visibility, name, "str")  # None for type if not specified
                                            )

            # Swimlane format class
            elif style and "swimlane" in style:
                class_name = clean_html_tags(value.strip())
                if not class_name:
                    raise ValueError(f"Invalid class name: '{class_name}'. Class name cannot be empty. Skipping class")
                elif class_name.lower() == "class":
                    class_name = f"_{class_name}"

                if class_name:
                    model_elements['classes'][class_name] = {
                        'attributes': [],
                        'methods': []
                    }

    # Second pass: process swimlane members
    for cell in root.findall(".//mxCell"):
        value = cell.get('value')
        parent = cell.get('parent')

        if value and parent in graph_data['cells']:
            parent_cell = graph_data['cells'][parent]
            if parent_cell.get('style') and "swimlane" in parent_cell.get('style'):
                parent_class = clean_html_tags(parent_cell['value'].strip())
                if parent_class in model_elements['classes']:
                    clean_value = clean_html_tags(value.strip())
                    if clean_value:
                        if '(' in clean_value:  # Method
                            visibility = "+" if clean_value.startswith("+ ") else "-" \
                                if clean_value.startswith("- ") else "+"
                            method_str = clean_value.lstrip("+ -")
                            method_match = re.match(r"(.*?)\((.*?)\)(?:\s*:\s*(.*))?", method_str)
                            if method_match:
                                method_name = method_match.group(1).strip()
                                params_str = method_match.group(2).strip()
                                return_type = method_match.group(3).strip() \
                                    if method_match.group(3) else None
                                # Parse parameters properly
                                parameters = []
                                if params_str:
                                    param_list = [p.strip() for p in params_str.split(',')]
                                    for param in param_list:
                                        if '=' in param:  # Handle default values
                                            param_parts = param.split('=')
                                            param_name_type = param_parts[0].strip()
                                            default_value = param_parts[1].strip().strip('"\'')
                                            if ':' in param_name_type:
                                                param_name, param_type = param_name_type.split(':')
                                                parameters.append({
                                                    'name': param_name.strip(),
                                                    'type': param_type.strip(),
                                                    'default': default_value
                                                })
                                            else:
                                                parameters.append({
                                                    'name': param_name_type,
                                                    'type': 'str',
                                                    'default': default_value
                                                })
                                        elif ':' in param:  # Handle type annotations
                                            param_name, param_type = param.split(':')
                                            parameters.append({
                                                'name': param_name.strip(),
                                                'type': param_type.strip()
                                            })
                                        else:  # Handle plain parameters
                                            parameters.append({
                                                'name': param.strip(),
                                                'type': 'str'
                                            })
                                model_elements['classes'][parent_class]['methods'].append(
                                    (visibility, method_name, parameters, return_type)
                                )
                        elif ':' in clean_value:  # Attribute
                            visibility = "+" if clean_value.startswith("+ ") else "-" \
                                if clean_value.startswith("- ") else "+"
                            attr_str = clean_value.lstrip("+ -")
                            name, type_str = attr_str.split(':', 1)
                            model_elements['classes'][parent_class]['attributes'].append(
                                (visibility, name.strip(), type_str.strip())
                            )
                        elif clean_value.isalpha() or clean_value.replace(" ", "").isalpha():
                            visibility = "+" if clean_value.startswith("+ ") else "-" \
                                if clean_value.startswith("- ") else "+"
                            name = clean_value.lstrip("+ -").strip()
                            model_elements['classes'][parent_class]['attributes'].append(
                            (visibility, name, "str")  # None for type if not specified
                            )

    # Process edges and associations
    for cell in root.findall(".//mxCell"):
        source = cell.get('source')
        target = cell.get('target')
        style = cell.get('style', '')

        # Handle generalizations (empty arrow)
        if source and target and "endArrow=block" in style and "endFill=0" in style:
            source_cell = graph_data['cells'].get(source, {})
            target_cell = graph_data['cells'].get(target, {})

            source_class = extract_class_name(source_cell.get('value', ''))
            target_class = extract_class_name(target_cell.get('value', ''))

            if not source_cell:
                print(f"Error: Source cell {source} not found for connection")
                continue
            if not target_cell:
                print(f"Error: Target cell {target} not found for connection")
                continue

            if source_class and target_class:
                model_elements['generalizations'].append({
                    'specific': source_class.strip(),
                    'general': target_class.strip()
                })

        # Handle one-way associations (filled arrow)
        elif source and target and ((("endArrow=block" and "edgeStyle=orthogonalEdgeStyle") or "endArrow=open") in style and "endFill=1" in style or
              "startArrow=diamondThin" in style and "startFill=0" in style
              or "endArrow=diamondThin" in style and "endFill=0" in style
              or "endArrow=open;endFill=1" in style):
            forced_source_multiplicity = None
            forced_target_multiplicity = None
            if "startArrow=diamondThin" in style:
                forced_source_multiplicity = Multiplicity(1, 1)
            elif "endArrow=diamondThin" in style:
                forced_target_multiplicity = Multiplicity(1, 1)

            association_name_source = None
            association_name_target = None

            if "endArrow=block" in style or "endArrow=open" in style or "startArrow=diamondThin" in style:
                start = True
            else:
                start = False

            # Get source and target cells
            source_cell = graph_data['cells'].get(source, {})
            target_cell = graph_data['cells'].get(target, {})

            # Extract class names, checking parent if not a class
            source_class = extract_class_name(source_cell.get('value', ''))
            if not ('swimlane' in source_cell.get('style', '') or '<b>' in source_cell.get('value', '')):
                source_parent = graph_data['cells'].get(source_cell.get('parent', ''))
                if source_parent and 'swimlane' in source_parent.get('style', ''):
                    source_class = extract_class_name(source_parent.get('value', ''))

            target_class = extract_class_name(target_cell.get('value', ''))
            if not ('swimlane' in target_cell.get('style', '') or '<b>' in target_cell.get('value', '')):
                target_parent = graph_data['cells'].get(target_cell.get('parent', ''))
                if target_parent and 'swimlane' in target_parent.get('style', ''):
                    target_class = extract_class_name(target_parent.get('value', ''))

            # Get role labels
            source_label = f"{source_class}_end"
            target_label = f"{target_class}_end"

            # Initialize default multiplicities
            source_multiplicity = Multiplicity(1, 1)  # Default multiplicity
            target_multiplicity = Multiplicity(1, 1)  # Default multiplicity

            # Find edge labels
            for label_cell in root.findall(f".//mxCell[@parent='{cell.get('id')}']"):
                if "edgeLabel" in label_cell.get('style', ''):
                    label_value = label_cell.get('value', '')
                    if label_value:
                        geometry = label_cell.find(".//mxGeometry")
                        if geometry is not None:
                            x = float(geometry.get('x', 0))

                            # Handle multiplicity labels
                            if label_value[0].isdigit() or label_value[0] == '*':
                                multiplicity_str = clean_html_tags(label_value)
                                if multiplicity_str == '*':
                                    multiplicity = Multiplicity(0, '*')
                                elif '..' in multiplicity_str:
                                    lower, upper = multiplicity_str.split('..')
                                    if upper == 'n':
                                        upper = '*'
                                    multiplicity = Multiplicity(int(lower), upper \
                                                                if upper == '*' else int(upper))
                                else:
                                    multiplicity = Multiplicity(int(multiplicity_str),\
                                                                 int(multiplicity_str))

                                if x < 0:
                                    source_multiplicity = multiplicity
                                else:
                                    target_multiplicity = multiplicity

                            # Handle role labels
                            else:
                                clean_value = clean_html_tags(label_value)
                                # Check for multiplicity in role name [0..*] format
                                multiplicity_match = re.search\
                                    (r'\[(\d+\.\.\d+|\d+\.\.\*|\d+|\*)\]', clean_value)
                                if multiplicity_match:
                                    mult_str = multiplicity_match.group(1)
                                    if mult_str == '*':
                                        multiplicity = Multiplicity(0, '*')
                                    elif '..' in mult_str:
                                        lower, upper = mult_str.split('..')
                                        if upper == 'n':
                                            upper = '*'
                                        multiplicity = Multiplicity(int(lower), upper \
                                                                    if upper == '*' else int(upper))
                                    else:
                                        multiplicity = Multiplicity(int(mult_str), int(mult_str))
                                    clean_value = clean_value.split('[')[0].strip()
                                else:
                                    multiplicity = Multiplicity(1, 1)  # Default multiplicity

                                if x < 0:
                                    # If it starts with capital, it's an association name
                                    if clean_value and clean_value[0].isupper():
                                        association_name_source = clean_value
                                        source_label = f"{source_class}_end"
                                    else:
                                        source_label = clean_value + "_non_navigable" if start else clean_value
                                    source_multiplicity = multiplicity if not forced_source_multiplicity else forced_source_multiplicity
                                else:
                                    # If it starts with capital, it's an association name
                                    if clean_value and clean_value[0].isupper():
                                        association_name_target = clean_value
                                        target_label = f"{target_class}_end"
                                    else:
                                        target_label = clean_value + "_non_navigable" if not start else clean_value
                                    target_multiplicity = multiplicity if not forced_target_multiplicity else forced_target_multiplicity

            # Add the association if both classes are found
            if source_class and target_class:
                source_assoc = {
                    'name': source_label,
                    'class': source_class,
                    'multiplicity': source_multiplicity,
                    'visibility': 'public'
                }
                if start:
                    source_assoc['navigable'] = False
                if association_name_source:
                    source_assoc['association_name'] = association_name_source
                    
                target_assoc = {
                    'name': target_label,
                    'class': target_class,
                    'multiplicity': target_multiplicity,
                    'visibility': 'public',
                }
                if not start:
                    target_assoc['navigable'] = False
                if association_name_target:
                    target_assoc['association_name'] = association_name_target

                model_elements['associations'].append(source_assoc)
                model_elements['associations'].append(target_assoc)

        elif source and target and ("startArrow=diamondThin" in style and "startFill=1"
              in style or "endArrow=diamondThin" in style and "endFill=1" in style):
            forced_source_multiplicity = None
            forced_target_multiplicity = None
            association_name_source = None
            association_name_target = None
            if "startArrow=diamondThin" in style:
                forced_source_multiplicity = Multiplicity(1, 1)
            elif "endArrow=diamondThin" in style:
                forced_target_multiplicity = Multiplicity(1, 1)

            if "startArrow=diamondThin" in style and "startFill=1" in style:
                start = True
            else:
                start = False

            # Get source and target cells
            source_cell = graph_data['cells'].get(source, {})
            target_cell = graph_data['cells'].get(target, {})

            # Extract class names, checking parent if not a class
            source_class = extract_class_name(source_cell.get('value', ''))
            if not ('swimlane' in source_cell.get('style', '') or '<b>' in source_cell.get('value', '')):
                source_parent = graph_data['cells'].get(source_cell.get('parent', ''), {})
                if 'swimlane' in source_parent.get('style', ''):
                    source_class = extract_class_name(source_parent.get('value', ''))

            target_class = extract_class_name(target_cell.get('value', ''))
            if not ('swimlane' in target_cell.get('style', '') or '<b>' in target_cell.get('value', '')):
                target_parent = graph_data['cells'].get(target_cell.get('parent', ''), {})
                if 'swimlane' in target_parent.get('style', ''):
                    target_class = extract_class_name(target_parent.get('value', ''))

            # Get role labels
            source_label = f"{source_class}_end"
            target_label = f"{target_class}_end"
            # Initialize default multiplicities
            source_multiplicity = Multiplicity(1, 1)  # Default multiplicity
            target_multiplicity = Multiplicity(1, 1)  # Default multiplicity

            # Find edge labels
            for label_cell in root.findall(f".//mxCell[@parent='{cell.get('id')}']"):
                if "edgeLabel" in label_cell.get('style', ''):
                    label_value = label_cell.get('value', '')
                    if label_value:
                        geometry = label_cell.find(".//mxGeometry")
                        if geometry is not None:
                            x = float(geometry.get('x', 0))

                            # Handle multiplicity labels
                            if label_value[0].isdigit() or label_value[0] == '*':
                                multiplicity_str = clean_html_tags(label_value)
                                if multiplicity_str == '*':
                                    multiplicity = Multiplicity(0, '*')
                                elif '..' in multiplicity_str:
                                    lower, upper = multiplicity_str.split('..')
                                    if upper == 'n':
                                        upper = '*'
                                    multiplicity = Multiplicity(int(lower), upper \
                                                                if upper == '*' else int(upper))
                                else:
                                    multiplicity = Multiplicity(int(multiplicity_str),\
                                                                 int(multiplicity_str))

                                if x < 0:
                                    source_multiplicity = multiplicity
                                else:
                                    target_multiplicity = multiplicity

                            # Handle role labels
                            else:
                                clean_value = clean_html_tags(label_value)
                                # Check for multiplicity in role name [0..*] format
                                multiplicity_match = re.search\
                                    (r'\[(\d+\.\.\d+|\d+\.\.\*|\d+|\*)\]', clean_value)
                                if multiplicity_match:
                                    mult_str = multiplicity_match.group(1)
                                    if mult_str == '*':
                                        multiplicity = Multiplicity(0, '*')
                                    elif '..' in mult_str:
                                        lower, upper = mult_str.split('..')
                                    if upper == 'n':
                                        upper = '*'
                                        multiplicity = Multiplicity(int(lower), upper \
                                                                    if upper == '*' else int(upper))
                                    else:
                                        multiplicity = Multiplicity(int(mult_str), int(mult_str))
                                    clean_value = clean_value.split('[')[0].strip()
                                else:
                                    multiplicity = Multiplicity(1, 1)  # Default multiplicity

                                if x < 0:
                                    # If it starts with capital, it's an association name
                                    if clean_value and clean_value[0].isupper():
                                        association_name_source = clean_value
                                        source_label = f"{source_class}_end"
                                    else:
                                        source_label = clean_value + "_composite" if start else clean_value
                                    source_multiplicity = multiplicity if not forced_source_multiplicity else forced_source_multiplicity
                                else:
                                    # If it starts with capital, it's an association name
                                    if clean_value and clean_value[0].isupper():
                                        association_name_target = clean_value
                                        target_label = f"{target_class}_end"
                                    else:
                                        target_label = clean_value + "_composite" if not start else clean_value
                                    target_multiplicity = multiplicity if not forced_target_multiplicity else forced_target_multiplicity

            # Add the association if both classes are found
            if source_class and target_class:
                source_assoc = {
                    'name': source_label,
                    'class': source_class,
                    'multiplicity': source_multiplicity,
                    'visibility': 'public'
                }
                if start:
                    source_assoc['composite'] = True
                if association_name_source:
                    source_assoc['association_name'] = association_name_source

                target_assoc = {
                    'name': target_label,
                    'class': target_class,
                    'multiplicity': target_multiplicity,
                    'visibility': 'public'
                }
                if not start:
                    target_assoc['composite'] = True
                if association_name_target:
                    target_assoc['association_name'] = association_name_target

                model_elements['associations'].append(source_assoc)
                model_elements['associations'].append(target_assoc)

        # Handle binary associations (edges with labels)
        if source and target and "endArrow=none" in style or \
            "endArrow=block;startArrow=block" in style :
            # Get source and target classes
            source_cell = root.find(f".//mxCell[@id='{source}']")
            target_cell = root.find(f".//mxCell[@id='{target}']")
            association_name_source = None
            association_name_target = None
            source_class = extract_class_name(source_cell.get('value', ''))
            if not ('swimlane' in source_cell.get('style', '') or '<b>' in source_cell.get('value', '')):
                source_parent = root.find(f".//mxCell[@id='{source_cell.get('parent')}']")
                if source_parent is not None and 'swimlane' in source_parent.get('style', ''):
                    source_class = extract_class_name(source_parent.get('value', ''))

            target_class = extract_class_name(target_cell.get('value', ''))
            if not ('swimlane' in target_cell.get('style', '') or '<b>' in target_cell.get('value', '')):
                target_parent = root.find(f".//mxCell[@id='{target_cell.get('parent')}']")
                if target_parent is not None and 'swimlane' in target_parent.get('style', ''):
                    target_class = extract_class_name(target_parent.get('value', ''))

            # Get role labels
            source_label = f"{source_class}_end"
            target_label = f"{target_class}_end"

            # Initialize default multiplicities
            source_multiplicity = Multiplicity(1, 1)  # Default multiplicity
            target_multiplicity = Multiplicity(1, 1)  # Default multiplicity

            # Find edge labels
            for label_cell in root.findall(f".//mxCell[@parent='{cell.get('id')}']"):
                if "edgeLabel" in label_cell.get('style', ''):
                    label_value = label_cell.get('value', '')
                    if label_value:
                        geometry = label_cell.find(".//mxGeometry")
                        if geometry is not None:
                            x = float(geometry.get('x', 0))

                            # Handle multiplicity labels
                            if label_value[0].isdigit() or label_value[0] == '*':
                                multiplicity_str = clean_html_tags(label_value)
                                if multiplicity_str == '*':
                                    multiplicity = Multiplicity(0, '*')
                                elif '..' in multiplicity_str:
                                    lower, upper = multiplicity_str.split('..')
                                    if upper == 'n':
                                        upper = '*'
                                    multiplicity = Multiplicity(int(lower), upper \
                                                                if upper == '*' else int(upper))
                                else:
                                    multiplicity = Multiplicity(int(multiplicity_str), \
                                                                int(multiplicity_str))

                                if x < 0:
                                    source_multiplicity = multiplicity
                                else:
                                    target_multiplicity = multiplicity

                            # Handle role labels
                            else:
                                clean_value = clean_html_tags(label_value)
                                # Check for multiplicity in role name [0..*] format
                                multiplicity_match = re.search\
                                    (r'\[(\d+\.\.\d+|\d+\.\.\*|\d+|\*)\]', clean_value)
                                if multiplicity_match:
                                    mult_str = multiplicity_match.group(1)
                                    if mult_str == '*':
                                        multiplicity = Multiplicity(0, '*')
                                    elif '..' in mult_str:
                                        lower, upper = mult_str.split('..')
                                        if upper == 'n':
                                            upper = '*'
                                        multiplicity = Multiplicity(int(lower), upper \
                                                                    if upper == '*' else int(upper))
                                    else:
                                        multiplicity = Multiplicity(int(mult_str), int(mult_str))
                                    clean_value = clean_value.split('[')[0].strip()
                                else:
                                    multiplicity = Multiplicity(1, 1)  # Default multiplicity

                                if x < 0:
                                    # If it starts with capital, it's an association name
                                    if clean_value and clean_value[0].isupper():
                                        association_name_source = clean_value
                                        source_label = f"{source_class}_end"
                                    else:
                                        source_label = clean_value
                                    source_multiplicity = multiplicity
                                else:
                                    # If it starts with capital, it's an association name
                                    if clean_value and clean_value[0].isupper():
                                        association_name_target = clean_value
                                        target_label = f"{target_class}_end"
                                    else:
                                        target_label = clean_value
                                    target_multiplicity = multiplicity

            # Add the association if both classes are found
            if source_class and target_class:
                source_assoc = {
                    'name': source_label,
                    'class': source_class,
                    'multiplicity': source_multiplicity,
                    'visibility': 'public'
                }
                if association_name_source:
                    source_assoc['association_name'] = association_name_source
                model_elements['associations'].append(source_assoc)

                target_assoc = {
                    'name': target_label,
                    'class': target_class,
                    'multiplicity': target_multiplicity,
                    'visibility': 'public'
                }
                if association_name_target is not None:
                    target_assoc['association_name'] = association_name_target
                model_elements['associations'].append(target_assoc)


    return (model_elements['classes'], model_elements['enumerations'],
            model_elements['associations'], model_elements['generalizations'],
            graph_data['cells'])

def generate_buml_from_xml(drawio_file: str) -> tuple:
    """
    Generate B-UML model from XML content.

    Creates B-UML model elements from extracted XML data including classes,
    enumerations, associations and generalizations.

    Args:
        drawio_file: Path to Draw.io file

    Returns:
        Tuple of (DomainModel, dict) containing the generated model and association properties
    """
    classes, enumerations, associations, generalizations, _ \
        = extract_classes_from_drawio(drawio_file)
    buml_classes = {}
    buml_enumerations = {}
    buml_associations = set()
    buml_generalizations = []
    association_properties = {}


    # Create enumerations
    for enum_name, enum_values in enumerations.items():
        enum_literals = set()
        for value in enum_values:
            literal = EnumerationLiteral(name=value)
            enum_literals.add(literal)
        buml_enum = Enumeration(name=enum_name, literals=enum_literals)
        buml_enumerations[enum_name] = buml_enum

    # First create all empty classes
    for class_name in classes:
        buml_classes[class_name] = Class(name=class_name)

    # Then populate classes with attributes and methods
    for class_name, class_data in classes.items():
        buml_class = buml_classes[class_name]
        buml_attributes = set()
        buml_methods = set()

        # Create attributes
        if 'attributes' in class_data:
            for attr_data in class_data['attributes']:
                # Make sure we have all three values
                if len(attr_data) == 3:
                    visibility, attr_name, attr_type = attr_data
                    #print(f"\nProcessing attribute: {attr_name} ({attr_type})")
                    visibility = "public" if visibility == "+" else "private"
                    #print(f": {buml_enumerations}")
                    # First check if it's an enumeration or class
                    if attr_type in buml_enumerations:
                        type_obj = buml_enumerations[attr_type]
                    elif attr_type in classes:  # Check if class exists in original classes dict
                        type_obj = buml_classes[attr_type]
                    else:
                        type_obj = PRIMITIVE_TYPE_MAPPING.get(attr_type.lower(), StringType)

                    buml_attribute = Property(
                        name=attr_name,
                        type=type_obj,
                        visibility=visibility
                    )
                    buml_attributes.add(buml_attribute)


        # Create methods
        if 'methods' in class_data:
            for method_data in class_data.get('methods', []):
                if len(method_data) == 2:  # Old format with just visibility and raw method string
                    visibility, method_str = method_data
                    # Parse method string
                    method_match = re.match(r"(.*?)\((.*?)\)(?:\s*:\s*(.*))?", method_str.strip())
                    if method_match:
                        method_name = method_match.group(1).strip()
                        params_str = method_match.group(2).strip()
                        return_type = method_match.group(3).strip() \
                            if method_match.group(3) else None

                        # Parse parameters
                        parameters = []
                        if params_str:
                            param_list = params_str.split(',')
                            for param in param_list:
                                param = param.strip()
                                if ':' in param:
                                    param_name, param_type = param.split(':')
                                    parameters.append({
                                        'name': param_name.strip(),
                                        'type': param_type.strip()
                                    })
                                else:
                                    parameters.append({
                                        'name': param.strip(),
                                        'type': StringType
                                    })

                        method_data = (visibility, method_name, parameters, return_type)

                if len(method_data) == 4:  # New format with all components
                    visibility, method_name, parameters, return_type = method_data
                    visibility = "public" if visibility == "+" else "private"

                    # Convert parameters
                    buml_parameters = set()
                    for param in parameters:
                        param_name = param['name']
                        param_type = param['type']
                        
                        # Si le nom contient encore le type (comme "str sms"), on le sépare
                        if ' ' in param_name:
                            type_str, param_name = param_name.split(' ', 1)
                            # On utilise le type explicite s'il est spécifié
                            if param_type == 'str':
                                param_type = type_str.lower()
                        
                        if param_type in buml_enumerations:
                            type_obj = buml_enumerations[param_type]
                        elif param_type in classes:
                            type_obj = buml_classes[param_type]
                        else:
                            type_obj = PRIMITIVE_TYPE_MAPPING.get(param_type.lower(), StringType)

                        param_obj = Parameter(
                            name=param_name,
                            type=type_obj
                        )
                        if 'default' in param:
                            param_obj.default_value = param['default']
                        buml_parameters.add(param_obj)

                    # Handle return type
                    if return_type:
                        if return_type in buml_enumerations:
                            return_type_obj = buml_enumerations[return_type]
                        elif return_type in classes:
                            return_type_obj = buml_classes[return_type]
                        else:
                            return_type_obj = PRIMITIVE_TYPE_MAPPING.get(return_type.lower())
                            if return_type_obj is None:
                                raise ValueError(f"Unknown return type '{return_type}'. It must be an enumeration, class, or primitive type.")
                    else:
                        return_type_obj = None

                    buml_method = Method(
                        name=method_name,
                        parameters=buml_parameters,
                        visibility=visibility,
                        type=return_type_obj
                    )
                    buml_methods.add(buml_method)

        # Update the class with its attributes and methods
        buml_class.attributes = buml_attributes
        buml_class.methods = buml_methods

    # Create associations
    for i in range(0, len(associations), 2):
        assoc1 = associations[i]
        assoc2 = associations[i + 1] if i + 1 < len(associations) else None

        if assoc2:
            class1 = buml_classes.get(assoc1['class'])
            class2 = buml_classes.get(assoc2['class'])

            if not class1:
                print(f"Error: Association references non-existent class '{assoc1['class']}'."
                      f"Make sure you are connecting to a valid class, not an attribute, method, or enumeration.")
                continue
            if not class2:
                print(f"Error: Association references non-existent class '{assoc2['class']}'."
                      f"Make sure you are connecting to a valid class, not an attribute, method, or enumeration.")
                continue

            if class1 and class2:
                assoc_property_1 = Property(
                    name=assoc1['name'].split('[')[0].strip(),
                    type=class1,
                    multiplicity=assoc1['multiplicity'] or Multiplicity(1, "*"),
                    visibility=assoc1['visibility'],
                    is_composite=assoc1.get('composite', False),
                    is_navigable=assoc1.get('navigable', True)
                )
                assoc_property_2 = Property(
                    name=assoc2['name'].split('[')[0].strip(),
                    type=class2,
                    multiplicity=assoc2['multiplicity'] or Multiplicity(1, "*"),
                    visibility=assoc2['visibility'],
                    is_composite=assoc2.get('composite', False),
                    is_navigable=assoc2.get('navigable', True)
                )

                association_name_user = assoc1.get('association_name') or assoc2.get('association_name')

                # Base association name
                if association_name_user:
                    base_name = association_name_user
                else:
                    base_name = f"{class1.name}_{class2.name}"
                    if assoc_property_1.is_composite:
                        base_name += "_composite"
                    elif not assoc_property_2.is_navigable:
                        base_name += "_non_navigable"
                    else:
                        base_name += "_association"

                # Check for existing associations with the same name
                association_name = base_name
                counter = 1
                while any(assoc.name == association_name for assoc in buml_associations):
                    association_name = f"{base_name}_{counter}"
                    counter += 1

                binary_assoc = BinaryAssociation(
                    name=association_name,
                    ends={assoc_property_1, assoc_property_2}
                )
                buml_associations.add(binary_assoc)
                association_properties[assoc1['name']] = assoc_property_1
                association_properties[assoc2['name']] = assoc_property_2

                # Add debug print
                #print(f"\nBinary Association created: {association_name}")
                #for end in binary_assoc.ends:
                #    mult_str = f"{end.multiplicity.min}..
                # {'*' if end.multiplicity.max == '*' else end.multiplicity.max}"
    # Create generalizations
    for gen in generalizations:
        specific_class = buml_classes.get(gen['specific'])
        general_class = buml_classes.get(gen['general'])

        if specific_class and general_class:
            buml_generalization = Generalization(general=general_class, specific=specific_class)
            buml_generalizations.append(buml_generalization)


    # Create domain model with associations
    domain_model = DomainModel(
        name="Generated_Model",
        types=set(buml_classes.values()) | set(buml_enumerations.values()),
        associations=buml_associations,
        generalizations=set(buml_generalizations)
    )

    #print(f"\nDomain Model created: {domain_model}")
    return domain_model, association_properties

def is_class_style(style: str, value: str) -> bool:
    """Check if the cell represents a class based on its style and value."""
    return (
        'swimlane' in style 
        or '<b>' in value 
        or (style and 'verticalAlign=top;align=left;overflow=fill' in style)
    )

def extract_class_name(cell_value: str) -> str:
    """
    Extract class name from a cell value, handling both HTML and plain text formats.
    """
    if not cell_value:
        return None

    # Handle HTML format with <b> tags
    if '<b>' in cell_value:
        class_match = re.search(r'<b>(.*?)</b>', cell_value)
        if class_match:
            return clean_html_tags(class_match.group(1))
    
    # Handle HTML format with margin-top style
    if 'margin-top:4px' in cell_value:
        class_match = re.search(r'<b>(.*?)</b>', cell_value)
        if class_match:
            return clean_html_tags(class_match.group(1))
    
    return clean_html_tags(cell_value)

def is_enumeration(value: str) -> bool:
    """Check if a value represents an enumeration using various notations."""
    enum_patterns = [
        "&lt;&lt;Enum&gt;&gt;", "<<Enum>>", "«Enum»",
        "&lt;&lt;enum&gt;&gt;", "<<enum>>", "«enum»",
        "&lt;&lt;Enumeration&gt;&gt;", "<<Enumeration>>", "«Enumeration»",
        "&lt;&lt;enumeration&gt;&gt;", "<<enumeration>>", "«enumeration»",
        "Enumeration", "enumeration"
    ]
    # Handle multi-line values (swimlane format)
    if '\n' in value:
        first_line = value.split('\n')[0]
        return any(pattern in first_line for pattern in enum_patterns)
    return any(pattern in value for pattern in enum_patterns)

def clean_enumeration_name(value: str) -> str:
    """Remove enumeration stereotypes and clean the name."""
    # Handle HTML encoded characters first
    value = value.replace('&lt;', '<').replace('&gt;', '>')
    
    replacements = [
        ("<<Enum>>", ""), ("«Enum»", ""),
        ("<<enum>>", ""), ("«enum»", ""),
        ("<<Enumeration>>", ""), ("«Enumeration»", ""),
        ("<<enumeration>>", ""), ("«enumeration»", ""),
        ("Enumeration", ""), ("enumeration", "")
    ]
    
    result = value
    for old, new in replacements:
        result = result.replace(old, new)
    
    # Clean any remaining HTML and whitespace
    result = clean_html_tags(result)
    
    # If there are multiple lines, take the last non-empty line
    if '\n' in result:
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        if lines:
            result = lines[-1]
    
    return result.strip()

def extract_enum_name_from_html(value: str) -> str:
    """Extract enumeration name from HTML formatted value."""
    b_tags = re.findall(r'<b>(.*?)</b>', value)
    for tag in b_tags:
        if not is_enumeration(tag):
            return tag
    return ""

def extract_literals_from_html(value: str) -> list:
    """Extract enumeration literals from HTML content."""
    literals = []
    
    # Handle div-separated literals
    if '<div>' in value:
        # Split by div tags and clean each part
        parts = value.split('</div>')
        for part in parts:
            # Clean the part from remaining div tags and other HTML
            clean_part = clean_html_tags(part).strip()
            if clean_part and not is_enumeration(clean_part) and clean_part != '<br>':
                literals.append(clean_part)
    else:
        # Handle other formats (unchanged)
        matches = re.findall(r'<hr[^>]*>.*?<p[^>]*>(.*?)</p>', value)
        for literal in matches:
            clean_literal = clean_html_tags(literal).lstrip('-').strip()
            if clean_literal and not is_enumeration(clean_literal) and clean_literal != '<br>':
                literals.append(clean_literal)
    
    return literals

def extract_literals_from_cells(root: ET.Element, cell_id: str) -> list:
    """Extract enumeration literals from child cells."""
    literals = []
    for literal_cell in root.findall(f".//mxCell[@parent='{cell_id}']"):
        literal_value = literal_cell.get('value', '')
        if literal_value and not is_enumeration(literal_value):
            clean_literal = clean_html_tags(literal_value).lstrip('-').strip()
            if clean_literal:
                literals.append(clean_literal)
    return literals

def process_method_string(method_str: str) -> tuple:
    """Process method string to extract name, parameters, and return type."""
    method_match = re.match(r"(.*?)\((.*?)\)(?:\s*:\s*(.*))?", method_str.strip())
    if method_match:
        method_name = method_match.group(1).strip()
        params_str = method_match.group(2).strip()
        return_type = method_match.group(3).strip() if method_match.group(3) else None
        return method_name, params_str, return_type
    return None, None, None

def parse_parameters(params_str: str) -> list:
    """Parse parameter string into list of parameter dictionaries."""
    parameters = []
    if params_str:
        param_list = params_str.split(',')
        for param in param_list:
            param = param.strip()
            
            # Check for default value
            default_value = None
            if '=' in param:
                param_parts = param.split('=')
                param = param_parts[0].strip()
                default_value = param_parts[1].strip().strip('"\'')  # Remove quotes
            
            # Check for type annotation
            if ':' in param:
                param_name, param_type = param.split(':')
                parameters.append({
                    'name': param_name.strip(),
                    'type': param_type.strip(),
                    'default': default_value
                })
            else:
                # Handle cases like "str sms"
                parts = param.split()
                if len(parts) > 1:
                    param_type = parts[0]
                    param_name = ' '.join(parts[1:])
                else:
                    param_type = 'str'
                    param_name = param
                
                parameters.append({
                    'name': param_name.strip(),
                    'type': param_type,
                    'default': default_value
                })
    return parameters

def extract_attributes_from_html(value: str) -> list:
    """Extract class attributes from HTML content."""
    attributes = []
    
    # Handle div-separated attributes
    if '<div>' in value:
        parts = value.split('</div>')
        for part in parts:
            # Clean the part from HTML tags
            clean_part = clean_html_tags(part).strip()
            if clean_part and clean_part != '<br>':
                # Parse attribute
                attr_parts = clean_part.split(':')
                attr_name = attr_parts[0].strip().lstrip('+')
                attr_type = attr_parts[1].strip() if len(attr_parts) > 1 else 'str'
                attributes.append((attr_name, attr_type))
    else:
        # Handle single line attributes
        attr_parts = clean_html_tags(value).split(':')
        attr_name = attr_parts[0].strip().lstrip('+')
        attr_type = attr_parts[1].strip() if len(attr_parts) > 1 else 'str'
        attributes.append((attr_name, attr_type))
    
    return attributes

def process_class_attributes(cell_value: str) -> list:
    """Process class attributes and return a list of Property objects."""
    properties = []
    
    attributes = extract_attributes_from_html(cell_value)
    for attr_name, attr_type in attributes:
        # Map type strings to actual types
        type_mapping = {
            'str': StringType,
            'int': IntegerType,
            'float': FloatType,
            'bool': BooleanType,
            'time': TimeType,
            'date': DateType,
            'datetime': DateTimeType,
            'timedelta': TimeDeltaType
        }
        
        attr_type_obj = type_mapping.get(attr_type, StringType)
        properties.append(Property(name=attr_name, type=attr_type_obj))
    
    return properties