import xml.etree.ElementTree as ET
import re
import math
from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    PrimitiveDataType, Multiplicity, BinaryAssociation

# Function to clean HTML tags from class names
def clean_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).strip()

# Function to compute the absolute position of a label on an edge
def calculate_absolute_position(edge, label_geometry):
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

# Function to extract classes and associations from the XML file
def extract_classes_from_xml(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Erreur lors du parsing du fichier XML: {e}")
        return {}, [], {}, {}, {}

    classes = {}
    associations = []
    cells = {}
    class_positions = {}
    edges = {}

    for cell in root.findall(".//mxCell"):
        cell_id = cell.get('id')
        value = cell.get('value')
        style = cell.get('style')
        source = cell.get('source')
        target = cell.get('target')
        geometry = cell.find(".//mxGeometry")

        # Get cell positions if available
        x = float(geometry.get('x')) if geometry is not None and geometry.get('x') else None
        y = float(geometry.get('y')) if geometry is not None and geometry.get('y') else None

        if cell_id:
            cells[cell_id] = {'value': value, 'x': x, 'y': y}

        # Check if it's a class
        if value and (style and "swimlane" in style):
            class_name = clean_html_tags(value.strip().replace("+", ""))
            if class_name:
                classes[class_name] = []
                class_positions[class_name] = {'x': x, 'y': y}
                print(f"Found class: {class_name} at position ({x}, {y})")

        # Check if it's a class attribute
        elif value and "+ " in value:
            field_name = value.split(":")[0].strip().replace("+ ", "")
            if classes:
                class_name = list(classes.keys())[-1]  # Assign to the last class found
                classes[class_name].append(field_name)
                print(f"Found attribute: {field_name} for class {class_name}")

        # Store edge information (source and target)
        if source and target and "endArrow" in (style or ""):
            source_x, source_y = 0, 0
            target_x, target_y = 0, 0

            edges[cell_id] = {
                'source': source, 'target': target, 'geometry': geometry,
                'source_x': source_x, 'source_y': source_y,
                'target_x': target_x, 'target_y': target_y
            }

        # Calculate absolute positions of association labels
        elif style and "edgeLabel" in style and value:
            edge_id = cell.get('parent')
            if edge_id in edges:
                edge = edges[edge_id]
                label_x = float(geometry.get('x')) if geometry.get('x') is not None else 0

                # Check if the label contains multiplicity
                multiplicity_match = re.search(r'\[(\d+\.\.\d+|\d+\.\.\*|\d+)\]', value)
                multiplicity = None
                #print(multiplicity_match)
                if multiplicity_match:
                    multiplicity_str = multiplicity_match.group(1)
                    if multiplicity_str == '*':
                        multiplicity = Multiplicity(0, '*')
                    elif '..' in multiplicity_str:
                        lower, upper = multiplicity_str.split('..')
                        multiplicity = Multiplicity(int(lower), upper if upper == '*' else int(upper))
                    elif '.' in multiplicity_str:
                        lower, upper = multiplicity_str.split('.')
                        multiplicity = Multiplicity(int(lower), upper if upper == '*' else int(upper))
                    else:
                        multiplicity = Multiplicity(int(multiplicity_str), int(multiplicity_str))

                # If x = -1, associate the label with the source class
                if label_x == -1:
                    assoc_class = clean_html_tags(cells[edge['source']]['value']).replace("+", "")
                    print(f"Label '{value}' is associated with source class: {assoc_class}")
                    associations.append({'name': clean_html_tags(value.split('[')[0]).strip(), 'class': assoc_class, 'multiplicity': multiplicity})

                # If x = 1, associate the label with the target class
                elif label_x == 1:
                    assoc_class = clean_html_tags(cells[edge['target']]['value']).replace("+", "")
                    print(f"Label '{value}' is associated with target class: {assoc_class}")
                    associations.append({'name': value, 'class': assoc_class, 'multiplicity': multiplicity})

                else:
                    raise ValueError(f"Invalid association used for label '{value}', please use the UML package notation from DrawIO.")
                    #abs_x, abs_y = calculate_absolute_position(edge, geometry)
                    #print(f"Label '{value}' absolute position calculated as ({abs_x}, {abs_y})")
                    #associations.append({'name': value, 'x': abs_x, 'y': abs_y, 'multiplicity': multiplicity})

    return classes, associations, cells, class_positions

# Function to generate BUML model from the extracted XML content
def generate_buml_from_xml(xml_file):
    classes, associations, cells, class_positions = extract_classes_from_xml(xml_file)

    buml_classes = {}
    buml_associations = []
    association_properties = {}

    # Define classes and their attributes
    for class_name, attributes in classes.items():
        buml_attributes = set()
        for attr in attributes:
            buml_attributes.add(Property(name=attr, type=PrimitiveDataType("str")))   #TODO: Add support for other data types
            print(f"Creating property {attr} for class {class_name}")

        buml_class = Class(name=class_name, attributes=buml_attributes)
        buml_classes[class_name] = buml_class
        print(f"Creating class {buml_class}")

    # Process each association
    for i in range(0, len(associations), 2):
        assoc1 = associations[i]
        assoc2 = associations[i + 1] if i + 1 < len(associations) else None

        print(f"Processing associations: {assoc1} and {assoc2}")

        if assoc2:
            # Find the classes associated with each label
            class1 = buml_classes.get(assoc1['class'])
            class2 = buml_classes.get(assoc2['class'])

            # Verify that the two classes are distinct before creating a binary association
            if class1 and class2 and class1 != class2:
                print(f"Creating binary association between {class1.name} and {class2.name}")

                # Create a binary association between the two classes
                assoc_property_1 = Property(name=assoc1['name'].split('[')[0].strip(), owner=class1, type=class2, multiplicity=assoc1['multiplicity'] or Multiplicity(1, "*"))
                assoc_property_2 = Property(name=assoc2['name'].split('[')[0].strip(), owner=class2, type=class1, multiplicity=assoc2['multiplicity'] or Multiplicity(1, "*"))

                buml_association = BinaryAssociation(
                    name=f"{assoc1['name']}_{assoc2['name']}_association",
                    ends={assoc_property_1, assoc_property_2}
                )
                buml_associations.append(buml_association)
                association_properties[assoc1['name']] = assoc_property_1
                association_properties[assoc2['name']] = assoc_property_2
            else:
                print(f"Skipping association between {class1} and {class2}: classes must be distinct.")
        else:
            print("No matching association found for label")

    # Define the domain model
    domain_model = DomainModel(name="Generated Model", types=set(buml_classes.values()), associations=set(buml_associations))
    print(f"Generated BUML model: {domain_model}")
    return domain_model, association_properties

# Function to save the generated BUML model to a Python file
def save_buml_to_file(model, association_properties, file_name):
    with open(file_name, 'w') as f:
        f.write("# BUML Generated Model\n")
        f.write("from besser.BUML.metamodel.structural import DomainModel, Class, Property, PrimitiveDataType, Multiplicity, BinaryAssociation\n\n")

        # Write the classes
        for buml_class in model.types:
            f.write(f"# Class: {buml_class.name}\n")
            for attribute in buml_class.attributes:
                f.write(f"{attribute.name}: Property = Property(name=\"{attribute.name}\", owner=None, type=PrimitiveDataType(\"str\"))\n")
            f.write(f"{buml_class.name}: Class = Class(name=\"{buml_class.name}\", attributes={{")
            f.write(", ".join([attr.name for attr in buml_class.attributes]))
            f.write("})\n\n")

        # Write the association properties first
        for assoc_name, assoc_property in association_properties.items():
            print(f"Writing association property {assoc_name}")
            print(f"{assoc_name}: Property = Property(name=\"{assoc_name}\", type={assoc_property.type.name}, multiplicity={assoc_property.multiplicity})\n")
            f.write(f"{assoc_name}: Property = Property(name=\"{assoc_name}\", type={assoc_property.type.name}, multiplicity={assoc_property.multiplicity})\n")

        # Write the associations after the properties
        f.write("\n# Associations\n")
        for association in model.associations:
            f.write(f"{association.name}: BinaryAssociation = BinaryAssociation(name=\"{association.name}\", ends={{")
            f.write(", ".join([end.name for end in association.ends]))
            f.write("})\n\n")

        # Finalize with the domain model definition
        f.write("domain_model: DomainModel = DomainModel(name=\"Generated Model\", types={")
        f.write(", ".join([buml_class.name for buml_class in model.types]))
        f.write("}, associations={")
        f.write(", ".join([association.name for association in model.associations]))
        f.write("})\n")

# Example usage
xml_file = 'aa.xml'
buml_model, association_properties = generate_buml_from_xml(xml_file)

# Save the BUML model to a Python file
output_file = 'generated_buml_model.py'
save_buml_to_file(buml_model, association_properties, output_file)

print(f"BUML model has been saved to {output_file}")