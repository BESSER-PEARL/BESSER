"""
Module for converting Draw.io structural UML diagrams to B-UML models.

This module provides functionality to parse Draw.io files containing UML class diagrams
and convert them into B-UML model representations. It handles classes, enumerations,
associations, generalizations, attributes, methods and their relationships.
"""

import os
import xml.etree.ElementTree as ET
import re
import warnings
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

def structural_drawio_to_buml(drawio_file_path: str, buml_model_file_name: str = "buml_model", \
                              save_buml: bool = True) -> DomainModel:
    """
    Transform a Draw.io structural model into a B-UML model.

    Args:
        drawio_file_path: Path to the Draw.io file containing the UML diagram
        buml_model_file_name: Name for the output B-UML model file (default: "buml_model")
        save_buml: Whether to save the model to a file (default: True)

    Returns:
        DomainModel: The generated B-UML model object
    """
    buml_model, _ = generate_buml_from_xml(drawio_file_path)

    if save_buml:
        # Save model to Python file
        output_file_path = os.path.join("buml", buml_model_file_name + ".py")
        domain_model_to_code(buml_model, output_file_path)

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

        # Skip processing as class if it's an enumeration
        if value and ("&lt;&lt;Enum&gt;&gt;" in value or "<<Enum>>" in value or "«Enum»" in value): # "&lt;&lt;Enum&gt;&gt;&#xa;asa
            # Extract enum name and literals...
            enum_name = ""
            #print(f"\nProcessing cell: {value}")
            if "<p" in value and "<b>" in value:
                # Format: <p><b><<Enum>></b></p><p><b>EnumName</b></p>
                b_tags = re.findall(r'<b>(.*?)</b>', value)

                # Find the name by looking for a <b> tag that doesn't contain "Enum"
                for tag in b_tags:
                    if "Enum" not in tag and "&lt;&lt;Enum&gt;&gt;" not in tag:
                        enum_name = tag
                        break
                # Initialize the list for this enum
                if enum_name:
                    model_elements['enumerations'][enum_name] = []

                    # Extract literals from the same value
                    p_tags = re.findall(r'<p[^>]*>(.*?)</p>', value)
                    for p_tag in p_tags:
                        if "<b>" not in p_tag and p_tag.strip() and "<br>" not in p_tag and "<hr" not in p_tag:
                            clean_literal = clean_html_tags(p_tag.lstrip("+-")).strip()
                            if clean_literal:
                                model_elements['enumerations'][enum_name].append(clean_literal)
                else:
                    print(f"Enum found without a name in value: {value}")

            elif "<br>" in value:
                # Format: <<Enum>><br>EnumName
                parts = value.split("<br>")
                enum_name = parts[-1]

                # Clean the enum name
                enum_name = clean_html_tags(enum_name.replace("&lt;&lt;Enum&gt;&gt;", "")
                                             .replace("<<Enum>>", "")
                                             .replace("«Enum»", "")).strip()

                if enum_name:
                    model_elements['enumerations'][enum_name] = []
                    # Find literals in child cells
                    for literal_cell in root.findall(f".//mxCell[@parent='{cell_id}']"):
                        literal_value = literal_cell.get('value', '')
                        if literal_value:
                            clean_literal = clean_html_tags(literal_value.lstrip("+-")).strip()
                            if clean_literal:
                                model_elements['enumerations'][enum_name].append(clean_literal)
                else:
                    print(f"Enum found without a name in value: {value}")
            else:
        # Format: <<Enum>>EnumName
                enum_name = value.replace("&lt;&lt;Enum&gt;&gt;", "").replace("<<Enum>>", "").replace("«Enum»", "").strip()
                if enum_name:
                    model_elements['enumerations'][enum_name] = []
                else:
                    print(f"Enum found without a name in value: {value}")
            continue  # Skip processing this cell as a class

        # Process classes (only if not an enumeration)
        if value:
            # HTML format class
            if value.startswith("<p") and "<b>" in value:
                class_match = re.search(r'<b>(.*?)</b>', value)
                if class_match:
                    class_name = clean_html_tags(class_match.group(1))
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
                                method_match = re.match(r"(.*?)\((.*?)\)(?:\s*:\s*(.*))?",\
                                                         method_str)
                                if method_match:
                                    method_name = method_match.group(1).strip()
                                    params_str = method_match.group(2).strip()
                                    return_type = method_match.group(3).strip() \
                                    if method_match.group(3) else None
                                    model_elements['classes'][class_name]['methods'].append(
                                        (visibility, method_name, params_str, return_type)
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
                                model_elements['classes'][parent_class]['methods'].append(
                                    (visibility, method_name, [], return_type)
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
            
            if "startArrow=diamondThin" in style:
                forced_source_multiplicity = Multiplicity(1, 1)
            elif "endArrow=diamondThin" in style:
                forced_target_multiplicity = Multiplicity(1, 1)
                

            if "endArrow=block" in style or "endArrow=open" in style or "startArrow=diamondThin" in style:
                start = True
            else:
                start = False

            source_cell = graph_data['cells'].get(source, {})
            target_cell = graph_data['cells'].get(target, {})

            source_class = extract_class_name(source_cell.get('value', ''))
            target_class = extract_class_name(target_cell.get('value', ''))
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
                                    multiplicity = Multiplicity(1, '*')  # Default multiplicity

                                if x < 0:
                                    source_label = clean_value + "_non_navigable" if start else clean_value
                                    source_multiplicity = multiplicity if not forced_source_multiplicity else forced_source_multiplicity
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
                target_assoc = {
                    'name': target_label,
                    'class': target_class,
                    'multiplicity': target_multiplicity,
                    'visibility': 'public',
                }
                if not start:
                    target_assoc['navigable'] = False

                model_elements['associations'].append(source_assoc)
                model_elements['associations'].append(target_assoc)

        elif source and target and ("startArrow=diamondThin" in style and "startFill=1"
              in style or "endArrow=diamondThin" in style and "endFill=1" in style):

            if "startArrow=diamondThin" in style:
                forced_source_multiplicity = Multiplicity(1, 1)
            elif "endArrow=diamondThin" in style:
                forced_target_multiplicity = Multiplicity(1, 1)

            if "startArrow=diamondThin" in style and "startFill=1" in style:
                start = True
            else:
                start = False

            # Get source and target classes
            source_cell = graph_data['cells'].get(source, {})
            target_cell = graph_data['cells'].get(target, {})

            source_class = extract_class_name(source_cell.get('value', ''))
            target_class = extract_class_name(target_cell.get('value', ''))

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
                                    multiplicity = Multiplicity(1, '*')  # Default multiplicity

                                if x < 0:
                                    source_label = clean_value + "_composite" if start else clean_value
                                    source_multiplicity = multiplicity if not forced_source_multiplicity else forced_source_multiplicity
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

                target_assoc = {
                    'name': target_label,
                    'class': target_class,
                    'multiplicity': target_multiplicity,
                    'visibility': 'public'
                }
                if not start:
                    target_assoc['composite'] = True

                model_elements['associations'].append(source_assoc)
                model_elements['associations'].append(target_assoc)

        # Handle binary associations (edges with labels)
        if source and target and "endArrow=none" in style or \
            "endArrow=block;startArrow=block" in style :
            # Get source and target classes
            source_cell = root.find(f".//mxCell[@id='{source}']")
            target_cell = root.find(f".//mxCell[@id='{target}']")

            source_class = extract_class_name(source_cell.get('value', ''))
            target_class = extract_class_name(target_cell.get('value', ''))

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
                                    multiplicity = Multiplicity(1, '*')  # Default multiplicity

                                if x < 0:
                                    source_label = clean_value
                                    source_multiplicity = multiplicity
                                else:
                                    target_label = clean_value
                                    target_multiplicity = multiplicity

            # Add the association if both classes are found
            if source_class and target_class:
                model_elements['associations'].append({
                    'name': source_label,
                    'class': source_class,
                    'multiplicity': source_multiplicity,
                    'visibility': 'public'
                })
                model_elements['associations'].append({
                    'name': target_label,
                    'class': target_class,
                    'multiplicity': target_multiplicity,
                    'visibility': 'public'
                })
             
            

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
    for class_name in classes.keys():
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
                    elif attr_type in classes.keys():  # Check if class exists in original classes dict
                        type_obj = buml_classes[attr_type]
                    else:
                        # Convert primitive type string to actual type class
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
                        param_type = param['type']
                        if param_type in buml_enumerations:
                            type_obj = buml_enumerations[param_type]
                        elif param_type in classes.keys():
                            type_obj = buml_classes[param_type]
                        else:
                            type_obj = PRIMITIVE_TYPE_MAPPING.get(param_type.lower(), StringType)

                        param_obj = Parameter(
                            name=param['name'],
                            type=type_obj
                        )
                        if 'default' in param:
                            param_obj.default_value = param['default']
                        buml_parameters.add(param_obj)

                    # Handle return type
                    if return_type:
                        if return_type in buml_enumerations:
                            return_type_obj = buml_enumerations[return_type]
                        elif return_type in classes.keys():
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
                    is_navigable=assoc2.get('navigable', True)  # Default to True if not specified
                )

                # Base association name
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
        name="Generated Model",
        types=set(buml_classes.values()) | set(buml_enumerations.values()),
        associations=buml_associations,
        generalizations=set(buml_generalizations)
    )

    #print(f"\nDomain Model created: {domain_model}")
    return domain_model, association_properties

def extract_class_name(cell_value: str) -> str:
    """
    Extract class name from a cell value, handling both HTML and plain text formats.
    
    Args:
        cell_value: The value from the cell, which may contain HTML tags
        
    Returns:
        The extracted class name as a string
    """
    if not cell_value:
        return None
        
    if '<b>' in cell_value:
        class_match = re.search(r'<b>(.*?)</b>', cell_value)
        if class_match:
            return clean_html_tags(class_match.group(1))
    return clean_html_tags(cell_value)
