"""
Module for converting Draw.io XML UML diagrams to B-UML models.

This module provides functionality to parse Draw.io XML files containing UML class diagrams
and convert them into B-UML model representations. It handles classes, enumerations,
associations, generalizations, attributes, methods and their relationships.
"""

import os
import xml.etree.ElementTree as ET
import re
from besser.BUML.metamodel.structural import DomainModel, Class, Property, PrimitiveDataType, \
        Multiplicity, BinaryAssociation, Enumeration, EnumerationLiteral, \
        Generalization, Method, Parameter

def xml_to_buml(xml_file_path: str, buml_model_file_name: str = "buml_model", save_buml: bool = True) -> DomainModel:
    """
    Transform a Draw.io XML model into a B-UML model.

    Args:
        xml_file_path: Path to the XML file containing the UML diagram
        buml_model_file_name: Name for the output B-UML model file (default: "buml_model")
        save_buml: Whether to save the model to a file (default: True)

    Returns:
        DomainModel: The generated B-UML model object
    """
    buml_model, _ = generate_buml_from_xml(xml_file_path)

    if save_buml:
        # Create output directory if needed
        output_dir = "buml"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save model to Python file
        output_file_path = os.path.join(output_dir, buml_model_file_name + ".py")
        save_buml_to_file(buml_model, output_file_path)
        print(f"BUML model saved to {output_file_path}")

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
    relative_x = float(label_geometry.get('x')) if label_geometry.get('x') is not None else 0
    relative_y = float(label_geometry.get('y')) if label_geometry.get('y') is not None else 0

    abs_x = source_x + (target_x - source_x) * relative_x
    abs_y = source_y + (target_y - source_y) * relative_x + relative_y

    offset = label_geometry.find(".//mxPoint[@as='offset']")
    if offset is not None:
        offset_x = float(offset.get('x')) if offset.get('x') else 0
        offset_y = float(offset.get('y')) if offset.get('y') else 0
        abs_x += offset_x
        abs_y += offset_y

    return abs_x, abs_y

def extract_classes_from_xml(xml_file: str) -> tuple:
    """
    Extract UML elements from Draw.io XML file.

    Parses the XML file to extract classes, enumerations, associations, generalizations
    and their positions/relationships.

    Args:
        xml_file: Path to Draw.io XML file

    Returns:
        Tuple containing:
        - classes: Dict mapping class names to attributes/methods
        - enumerations: Dict mapping enum names to literals
        - associations: List of association dictionaries
        - generalizations: List of generalization dictionaries  
        - cells: Dict mapping cell IDs to cell data
        - class_positions: Dict mapping class names to positions
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return {}, [], {}, {}, {}, []

    # Group model elements
    model_elements = {
        'classes': {},
        'enumerations': {},
        'associations': [],
        'generalizations': []
    }
    
    # Group graph data
    graph_data = {
        'cells': {},
        'class_positions': {},
        'edges': {}
    }

    # First pass: collect cells
    for cell in root.findall(".//mxCell"):
        cell_id = cell.get('id')
        value = cell.get('value') 
        style = cell.get('style')
        source = cell.get('source')
        target = cell.get('target')
        geometry = cell.find(".//mxGeometry")

        # Get cell positions
        x = float(geometry.get('x')) if geometry is not None and geometry.get('x') else None
        y = float(geometry.get('y')) if geometry is not None and geometry.get('y') else None

        if cell_id:
            graph_data['cells'][cell_id] = {'value': value, 'x': x, 'y': y}

        # Process class elements
        if value and (style and "swimlane" in style):
            if "<<Enum>>" in value or "&lt;&lt;Enum&gt;&gt;" in value or "&amp;lt;&amp;lt;Enum&amp;gt;&amp;gt;" in value:
                # Handle enumeration
                if "&#10;" in value:
                    enum_name = clean_html_tags(value.split("&#10;")[1].strip())
                elif "<br>" in value:
                    enum_name = clean_html_tags(value.split("<br>")[1].strip())
                else:
                    enum_name = clean_html_tags(value.split("\n")[1].strip())
                if enum_name:
                    model_elements['enumerations'][enum_name] = []
                    graph_data['class_positions'][enum_name] = {'x': x, 'y': y}
            else:
                # Handle regular class
                class_name = clean_html_tags(value.strip().replace("+", ""))
                if class_name:
                    model_elements['classes'][class_name] = {'attributes': [], 'methods': []}
                    graph_data['class_positions'][class_name] = {'x': x, 'y': y}

        # Process class members
        elif value and ("+ " in value or "- " in value):
            clean_value = clean_html_tags(value.strip())
            parent_id = cell.get('parent')
            
            if parent_id and parent_id in graph_data['cells']:
                parent_value = graph_data['cells'][parent_id]['value']
                
                # Handle enumeration literals
                if parent_value and ("<<Enum>>" in parent_value or "&lt;&lt;Enum&gt;&gt;" in parent_value or "&amp;lt;&amp;lt;Enum&amp;gt;&amp;gt;" in parent_value):
                    if "&#10;" in parent_value:
                        enum_name = clean_html_tags(parent_value.split("&#10;")[1].strip())
                    elif "<br>" in parent_value:
                        enum_name = clean_html_tags(parent_value.split("<br>")[1].strip())
                    else:
                        enum_name = clean_html_tags(parent_value.split("\n")[1].strip())
                    if enum_name in model_elements['enumerations']:
                        literal_name = clean_value.lstrip("+-").strip()
                        model_elements['enumerations'][enum_name].append(literal_name)
                    continue

                # Handle methods
                if '(' in value:
                    method_match = re.match(r"([\+\-]) (.*?)\((.*?)\)(?:\s*:\s*(.*))?", value)
                    if method_match:
                        visibility = "+" if "+ " in value else "-"
                        method_name = method_match.group(2).strip()
                        params_str = method_match.group(3).strip()
                        return_type = None
                        
                        # Extract return type after the colon
                        if method_match.group(4):
                            return_type = method_match.group(4).strip()
                            # Clean any trailing comments or extra whitespace
                            if ' ' in return_type:
                                return_type = return_type.split(' ')[0].strip()

                        parameters = []
                        if params_str:
                            param_list = params_str.split(',')
                            for param in param_list:
                                param = param.strip()
                                
                                # Handle parameter with type but no name (e.g., "str sms")
                                if ' ' in param and ':' not in param:
                                    param_type, param_name = param.split(' ', 1)
                                    param_name = param_name.strip()
                                    param_type = param_type.strip()
                                # Handle parameter with explicit type (e.g., "title:str")
                                elif ':' in param:
                                    param_name, param_type = param.split(':')
                                    param_name = param_name.strip()
                                    param_type = param_type.strip()
                                else:
                                    param_name = param
                                    param_type = "str"  # default type if none specified
                                
                                # Handle default value if present
                                if '=' in param_name:
                                    param_name, default_value = param_name.split('=')
                                    parameters.append({
                                        'name': param_name.strip(),
                                        'type': param_type,
                                        'default': default_value.strip('"\'')  # Remove quotes
                                    })
                                elif '=' in param_type:
                                    param_type, default_value = param_type.split('=')
                                    parameters.append({
                                        'name': param_name,
                                        'type': param_type.strip(),
                                        'default': default_value.strip('"\'')  # Remove quotes
                                    })
                                else:
                                    parameters.append({
                                        'name': param_name,
                                        'type': param_type
                                    })

                        method_data = (visibility, method_name, parameters, return_type)
                        if parent_value and parent_value.strip() in model_elements['classes']:
                            class_name = parent_value.strip()
                            if isinstance(model_elements['classes'][class_name], list):
                                model_elements['classes'][class_name] = {'attributes': model_elements['classes'][class_name], 'methods': []}
                            model_elements['classes'][class_name]['methods'].append(method_data)

                # Handle attributes
                else:
                    match = re.match(r"[\+\-] (.*?):\s*(.*)", value)
                    if match:
                        visibility = "+" if "+ " in value else "-"
                        field_name = match.group(1).strip()
                        field_type = match.group(2).strip()
                        attribute_data = (visibility, field_name, field_type)

                        if parent_value and parent_value.strip() in model_elements['classes']:
                            class_name = parent_value.strip()
                            if isinstance(model_elements['classes'][class_name], list):
                                model_elements['classes'][class_name] = {'attributes': model_elements['classes'][class_name], 'methods': []}
                            model_elements['classes'][class_name]['attributes'].append(attribute_data)

        # Store edge information
        if source and target and "endArrow" in (style or ""):
            graph_data['edges'][cell_id] = {
                'source': source,
                'target': target,
                'geometry': geometry,
                'source_x': 0,
                'source_y': 0,
                'target_x': 0,
                'target_y': 0
            }

    # Second pass: process relationships
    for cell in root.findall(".//mxCell"):
        cell_id = cell.get('id')
        style = cell.get('style')
        source = cell.get('source')
        target = cell.get('target')

        # Handle generalizations
        if source and target and "endArrow=block" in (style or ""):
            source_class = clean_html_tags(graph_data['cells'][source]['value']).replace("+", "") if source in graph_data['cells'] else None
            target_class = clean_html_tags(graph_data['cells'][target]['value']).replace("+", "") if target in graph_data['cells'] else None
            if source_class and target_class:
                model_elements['generalizations'].append({'specific': source_class, 'general': target_class})

        # Handle association labels
        elif style and "edgeLabel" in style and cell.get('value'):
            value = cell.get('value')
            geometry = cell.find(".//mxGeometry")
            edge_id = cell.get('parent')
            
            if edge_id in graph_data['edges']:
                edge = graph_data['edges'][edge_id]
                label_x = float(geometry.get('x')) if geometry.get('x') is not None else 0

                # Handle multiplicity
                if value[0].isdigit() or value[0] == '*':
                    multiplicity_str = value
                    if multiplicity_str == '*':
                        multiplicity = Multiplicity(0, '*')
                    elif multiplicity_str == '1':
                        print("im here")
                        multiplicity = Multiplicity(1, 1)
                    elif '..' in multiplicity_str:
                        lower, upper = multiplicity_str.split('..')
                        multiplicity = Multiplicity(int(lower), upper if upper == '*' else int(upper))
                    else:
                        multiplicity = Multiplicity(int(multiplicity_str), int(multiplicity_str))

                    if label_x > 0:
                        print(f"Label x > 0: {label_x}")
                        print(f"Current multiplicity: {multiplicity}")
                        print(f"Target class: {clean_html_tags(graph_data['cells'][edge['target']]['value']).replace('+', '')}")
                        
                        # Find the most recent matching association (searching backwards)
                        target_class = clean_html_tags(graph_data['cells'][edge['target']]['value']).replace('+', '')
                        matching_assoc_idx = next((i for i in range(len(model_elements['associations'])-1, -1, -1) 
                                                if model_elements['associations'][i]['class'] == target_class), None)
                        
                        if matching_assoc_idx is not None:
                            print(f"Found matching association at index {matching_assoc_idx}")
                            model_elements['associations'][matching_assoc_idx]['multiplicity'] = multiplicity
                    else:
                        print(f"Label x <= 0: {label_x}")
                        print(f"Current multiplicity: {multiplicity}")
                        print(f"Associations: {model_elements['associations']}")
                        if len(model_elements['associations']) >= 2 and model_elements['associations'][-2]['class'] == clean_html_tags(graph_data['cells'][edge['source']]['value']).replace("+", ""):
                            print("Assigning multiplicity to source end")
                            model_elements['associations'][-2]['multiplicity'] = multiplicity


                # Handle association names
                else:
                    visibility = "public" if value.startswith("+") else "private" if value.startswith("-") else "public"
                    clean_value = value.lstrip("+-").strip()

                    multiplicity = None
                    multiplicity_match = re.search(r'\[(\d+\.\.\d+|\d+\.\.\*|\d+)\]', clean_value)
                    if multiplicity_match:
                        multiplicity_str = multiplicity_match.group(1)
                        if multiplicity_str == '*':
                            multiplicity = Multiplicity(0, '*')
                        elif multiplicity_str == '1':
                            multiplicity = Multiplicity(1, 1)
                        elif '..' in multiplicity_str:
                            lower, upper = multiplicity_str.split('..')
                            multiplicity = Multiplicity(int(lower), upper if upper == '*' else int(upper))
                        else:
                            multiplicity = Multiplicity(int(multiplicity_str), int(multiplicity_str))
                        clean_value = clean_value.split('[')[0].strip()

                    if label_x < 0:
                        assoc_class = clean_html_tags(graph_data['cells'][edge['source']]['value']).replace("+", "")
                        model_elements['associations'].append({
                            'name': clean_value,
                            'class': assoc_class,
                            'multiplicity': multiplicity,
                            'visibility': visibility
                        })
                        
                    elif label_x > 0:
                        assoc_class = clean_html_tags(graph_data['cells'][edge['target']]['value']).replace("+", "")
                        model_elements['associations'].append({
                            'name': clean_value,
                            'class': assoc_class,
                            'multiplicity': multiplicity,
                            'visibility': visibility
                        })
                        

    return (model_elements['classes'], model_elements['enumerations'],
            model_elements['associations'], model_elements['generalizations'],
            graph_data['cells'], graph_data['class_positions'])

def generate_buml_from_xml(xml_file: str) -> tuple:
    """
    Generate B-UML model from XML content.

    Creates B-UML model elements from extracted XML data including classes,
    enumerations, associations and generalizations.

    Args:
        xml_file: Path to XML file

    Returns:
        Tuple of (DomainModel, dict) containing the generated model and association properties
    """
    classes, enumerations, associations, generalizations, cells, class_positions = extract_classes_from_xml(xml_file)
    
    buml_classes = {}
    buml_enumerations = set()
    buml_associations = []
    buml_generalizations = []
    association_properties = {}

    # Create enumerations
    for enum_name, enum_values in enumerations.items():
        enum_literals = set()
        for value in enum_values:
            literal = EnumerationLiteral(name=value)
            enum_literals.add(literal)
        buml_enum = Enumeration(name=enum_name, literals=enum_literals)
        buml_enumerations.add(buml_enum)

    # Create classes
    for class_name, class_data in classes.items():
        buml_attributes = set()
        buml_methods = set()

        # Create attributes
        for attr_data in class_data['attributes']:
            visibility, attr_name, attr_type = attr_data
            visibility = "public" if visibility == "+" else "private"

            enum_type = next((enum for enum in buml_enumerations if enum.name == attr_type), None)
            if enum_type:
                buml_attribute = Property(
                    name=attr_name,
                    type=enum_type,
                    visibility=visibility
                )
            else:
                buml_attribute = Property(
                    name=attr_name,
                    type=PrimitiveDataType(attr_type),
                    visibility=visibility
                )
            buml_attributes.add(buml_attribute)

        # Create methods
        for method_data in class_data.get('methods', []):
            visibility, method_name, parameters, return_type = method_data
            visibility = "public" if visibility == "+" else "private"
            
            # Convert parameters to Parameter objects
            buml_parameters = set()
            for param in parameters:
                # Resolve parameter type
                param_type = (next((enum for enum in buml_enumerations if enum.name == param['type']), None) or
                            next((cls for cls in buml_classes.values() if cls.name == param['type']), None) or
                            PrimitiveDataType(param['type']))
                
                # Create Parameter object
                param_obj = Parameter(
                    name=param['name'],
                    type=param_type
                )
                if 'default' in param:
                    param_obj.default_value = param['default']
                
                buml_parameters.add(param_obj)
            
            # Handle return type (could be enum, class, or primitive)
            if return_type:
                return_type_obj = (next((enum for enum in buml_enumerations if enum.name == return_type), None) or
                                 next((cls for cls in buml_classes.values() if cls.name == return_type), None) or
                                 return_type)
            else:
                return_type_obj = None

            buml_method = Method(
                name=method_name,
                parameters=buml_parameters,
                visibility=visibility,
                type=return_type_obj
            )
            buml_methods.add(buml_method)

        # Create class
        buml_class = Class(
            name=class_name,
            attributes=buml_attributes,
            methods=buml_methods
        )
        buml_classes[class_name] = buml_class

    # Create associations
    for i in range(0, len(associations), 2):
        assoc1 = associations[i]
        assoc2 = associations[i + 1] if i + 1 < len(associations) else None

        if assoc2:
            class1 = buml_classes.get(assoc1['class'])
            class2 = buml_classes.get(assoc2['class'])

            if class1 and class2:
                assoc_property_1 = Property(
                    name=assoc1['name'].split('[')[0].strip(),
                    owner=class1,
                    type=class2,
                    multiplicity=assoc1['multiplicity'] or Multiplicity(1, "*"),
                    visibility=assoc1['visibility']
                )
                assoc_property_2 = Property(
                    name=assoc2['name'].split('[')[0].strip(),
                    owner=class2,
                    type=class1,
                    multiplicity=assoc2['multiplicity'] or Multiplicity(1, "*"),
                    visibility=assoc2['visibility']
                )

                association_name = f"{class1.name}_{class2.name}_association"
                buml_association = BinaryAssociation(
                    name=association_name,
                    ends={assoc_property_1, assoc_property_2}
                )
                buml_associations.append(buml_association)
                association_properties[assoc1['name']] = assoc_property_1
                association_properties[assoc2['name']] = assoc_property_2

    # Create generalizations
    for gen in generalizations:
        specific_class = buml_classes.get(gen['specific'])
        general_class = buml_classes.get(gen['general'])

        if specific_class and general_class:
            buml_generalization = Generalization(general=general_class, specific=specific_class)
            buml_generalizations.append(buml_generalization)

    # Create domain model
    domain_model = DomainModel(
        name="Generated Model",
        types=set(buml_classes.values()),
        associations=set(buml_associations),
        generalizations=set(buml_generalizations),
        enumerations=set(buml_enumerations)
    )

    return domain_model, association_properties

def save_buml_to_file(model: DomainModel, file_name: str):
    """
    Save B-UML model to a Python file.

    Args:
        model: The B-UML domain model to save
        file_name: Path to the output Python file
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        # Write imports
        f.write("# Generated B-UML Model\n")
        f.write("from besser.BUML.metamodel.structural import (\n")
        f.write("    Class, Property, Method, Parameter, PrimitiveDataType,\n")
        f.write("    BinaryAssociation, Generalization, DomainModel,\n") 
        f.write("    Enumeration, EnumerationLiteral, Multiplicity\n")
        f.write(")\n\n")

        # Write enumerations
        if model.enumerations:
            f.write("# Enumerations\n")
            for enum in model.enumerations:
                literals_str = ", ".join([f"EnumerationLiteral(name=\"{lit.name}\")" for lit in enum.literals])
                f.write(f"{enum.name} = Enumeration(name=\"{enum.name}\", literals={{{literals_str}}})\n")
            f.write("\n")

        # Declare classes first
        f.write("# Classes\n")
        for cls in model.types:
            f.write(f"{cls.name}: Class = Class(name=\"{cls.name}\")\n")
        f.write("\n")

        # Write class members
        for cls in model.types:
            if cls.attributes or cls.methods:
                f.write(f"# {cls.name} class attributes and methods\n")
                
                # Write attributes
                for attr in cls.attributes:
                    type_str = attr.type.name if isinstance(attr.type, (Class, Enumeration)) else f"\"{attr.type.name}\""
                    f.write(f"{cls.name}_{attr.name}: Property = Property(name=\"{attr.name}\", "
                           f"type={type_str}, visibility=\"{attr.visibility}\")\n")
                
                # Write methods
                for method in cls.methods:
                    params = []
                    for param in method.parameters:
                        type_str = param.type.name if isinstance(param.type, (Class, Enumeration)) else f"\"{param.type.name}\""
                        param_str = f"Parameter(name=\"{param.name}\", type={type_str}"
                        if hasattr(param, 'default_value') and param.default_value is not None:
                            param_str += f", default_value=\"{param.default_value}\""
                        param_str += ")"
                        params.append(param_str)

                    params_str = ", ".join(params)
                    type_str = ""
                    if method.type:
                        if isinstance(method.type, (Class, Enumeration)):
                            type_str = f", type={method.type.name}"
                        else:
                            type_str = f", type=\"{method.type}\""
                    
                    f.write(f"{cls.name}_m_{method.name}: Method = Method(name=\"{method.name}\", "
                           f"visibility=\"{method.visibility}\", parameters={{{params_str}}}{type_str})\n")
                
                # Write class members assignment
                if cls.attributes:
                    attrs_str = ", ".join([f"{cls.name}_{attr.name}" for attr in cls.attributes])
                    f.write(f"{cls.name}.attributes={{{attrs_str}}}\n")
                if cls.methods:
                    methods_str = ", ".join([f"{cls.name}_m_{method.name}" for method in cls.methods])
                    f.write(f"{cls.name}.methods={{{methods_str}}}\n")
                f.write("\n")

        # Write associations
        if model.associations:
            f.write("# Relationships\n")
            for assoc in model.associations:
                ends_str = []
                for end in assoc.ends:
                    max_value = '"*"' if end.multiplicity.max == "*" else end.multiplicity.max
                    print(end.multiplicity.max)
                    print(end.multiplicity.min)
                    ends_str.append(f"Property(name=\"{end.name}\", type={end.type.name}, "
                                  f"multiplicity=Multiplicity({end.multiplicity.min}, {max_value}))")
                f.write(f"{assoc.name}: BinaryAssociation = BinaryAssociation(name=\"{assoc.name}\", "
                       f"ends={{{', '.join(ends_str)}}})\n")
            f.write("\n")

        # Write generalizations
        if model.generalizations:
            f.write("# Generalizations\n")
            for gen in model.generalizations:
                f.write(f"gen_{gen.specific.name}_{gen.general.name} = Generalization(general={gen.general.name}, "
                       f"specific={gen.specific.name})\n")
            f.write("\n")

        # Write domain model
        f.write("# Domain Model\n")
        f.write("domain_model = DomainModel(\n")
        f.write("    name=\"Generated Model\",\n")
        f.write(f"    types={{{', '.join(cls.name for cls in model.types)}}},\n")
        f.write(f"    associations={{{', '.join(assoc.name for assoc in model.associations)}}},\n")
        f.write(f"    generalizations={{{', '.join(f'gen_{gen.specific.name}_{gen.general.name}' for gen in model.generalizations)}}},\n")
        f.write(f"    enumerations={{{', '.join(enum.name for enum in model.enumerations)}}}\n")
        f.write(")\n")
