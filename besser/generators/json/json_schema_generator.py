import os
import json
import re
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import (
    DomainModel, IntegerType, StringType, Class,
    BooleanType, FloatType, Enumeration, DateType,
    DateTimeType, TimeType,
)
from besser.generators import GeneratorInterface

class JSONSchemaGenerator(GeneratorInterface):
    """
    JSONSchemaGenerator is a class that implements the GeneratorInterface and is responsible for generating
    JSONSchema code based on B-UML models.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
        mode (str, optional): The generation mode, either 'regular' or 'smart_data'. Defaults to 'regular'.
    """

    def __init__(self, model: DomainModel, output_dir: str = None, mode: str = 'regular'):
        super().__init__(model, output_dir)
        self.mode = mode
        # Add enums to TYPES dictionary

    def _get_property_type(self, property_type):
        """
        Maps B-UML types to JSON schema types and returns format when needed.
        
        Args:
            property_type: The B-UML type to map.
            
        Returns:
            tuple: (type, format) where format is None if not applicable
        """
        type_mapping = {
            IntegerType: "number",
            StringType: "string",
            BooleanType: "boolean",
            FloatType: "number",
            DateTimeType: "string",
            TimeType: "string",
            DateType: "string",
            list: "array"
        }

        format_mapping = {
            DateTimeType: "date-time",
            DateType: "date",
            TimeType: "time"
        }

        if isinstance(property_type, Enumeration):
            return "string", None

        json_type = type_mapping.get(property_type, "string")
        json_format = format_mapping.get(property_type, None)

        return json_type, json_format

    def _prepare_smart_data_schema_for_class(self, class_def):
        """
        Prepares schema data for Smart Data format for a specific class.
        
        Args:
            class_def: The class to prepare the schema for.
            
        Returns:
            dict: A dictionary containing the schema data.
        """
        schema_data = {
            "type": "object",
            "properties": {},
            "required": ["id", "type"],
            "class_name": class_def.name
        }

        # Add class description if available
        if class_def.metadata and class_def.metadata.description:
            schema_data["model_description"] = class_def.metadata.description

        # Process class attributes
        for attr in class_def.attributes:
            prop_type, prop_format = self._get_property_type(attr.type)
            prop_def = {"type": prop_type}

            if prop_format is not None:
                prop_def["format"] = prop_format

            if attr.metadata and attr.metadata.description:
                prop_def["description"] = attr.metadata.description
            else:
                prop_def["description"] = "Property"

            if isinstance(attr.type, Enumeration):
                prop_def["enum"] = [lit.name for lit in attr.type.literals]
                # Fix for enumeration type
                prop_def["type"] = "string"

            if hasattr(attr, 'multiplicity') and attr.multiplicity.max > 1:
                prop_def = {
                    "type": "array",
                    "items": prop_def
                }

            schema_data["properties"][attr.name] = prop_def

        # Process association ends for this class - only include direct associations
        for association in self.model.associations:
            # Find ends that reference this class
            class_ends = [end for end in association.ends if end.type == class_def]

            # Skip if this class is not directly involved in the association
            if not class_ends:
                continue

            # Find the other end of the association
            for class_end in class_ends:
                other_ends = [end for end in association.ends if end != class_end]
                if not other_ends:
                    continue

                other_end = other_ends[0]
                prop_name = other_end.name

                # Skip if the property name is not defined
                if not prop_name:
                    continue

                # Use association-level description if available
                relationship_description = (
                    association.metadata.description
                    if association.metadata and association.metadata.description
                    else f"Relationship. Reference to {other_end.type.name} entity"
                )

                relationship_def = {
                    "anyOf": [
                        {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 256,
                            "pattern": r"^[\w\-\.\{\}\$\+\*\[\]`|~^@!,:\\]+$",
                            "description": "Property. Identifier format of any NGSI entity"
                        },
                        {
                            "type": "string",
                            "format": "uri",
                            "description": "Property. Identifier format of any NGSI entity"
                        }
                    ],
                    "description": relationship_description
                }

                # Handle multiplicity
                if hasattr(other_end, 'multiplicity') and other_end.multiplicity:
                    if other_end.multiplicity.max > 1:
                        schema_data["properties"][prop_name] = {
                            "type": "array",
                            "items": relationship_def
                        }
                    else:
                        schema_data["properties"][prop_name] = relationship_def

                    # If multiplicity.min >= 1, make the field required
                    if other_end.multiplicity.min >= 1:
                        schema_data["required"].append(prop_name)
                else:
                    schema_data["properties"][prop_name] = relationship_def

        return schema_data

    def generate(self):
        """
        Generates JSONSchema code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file
        """
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))

        env.filters['tojson_ordered'] = self.tojson_ordered

        if self.mode == 'smart_data':
            # Smart Data mode - generate a schema for each class
            template = env.get_template('smart_data_schema.json.j2')

            # Get all classes from the model
            classes = [c for c in self.model.types if isinstance(c, Class)]

            for class_def in classes:
                # Create a directory for each class
                class_dir = os.path.join(self.output_dir, class_def.name)

                os.makedirs(class_dir, exist_ok=True)

                # Prepare schema data for this class
                schema_data = self._prepare_smart_data_schema_for_class(class_def)
                # Generate schema file in the class directory
                file_path = os.path.join(class_dir, "schema.json")

                with open(file_path, mode="w", encoding='utf-8') as f:
                    generated_code = template.render(schema=schema_data)
                    f.write(generated_code)

                examples_dir = os.path.join(class_dir, "examples")
                os.makedirs(examples_dir, exist_ok=True)

                file_example_path = os.path.join(examples_dir, "example-normalized.json")
                with open(file_example_path, mode="w", encoding='utf-8') as f:
                    generated_example_code = self.generate_example(generated_code)
                    f.write(generated_example_code)

                print(f"Smart Data schema for {class_def.name} generated in: {file_path}")

            # Create empty ADOPTERS.yaml and notes.yaml in the root output directory
            adopters_file_path = os.path.join(self.output_dir, "ADOPTERS.yaml")
            notes_file_path = os.path.join(self.output_dir, "notes.yaml")

            with open(adopters_file_path, mode="w", encoding='utf-8') as f:
                pass

            with open(notes_file_path, mode="w", encoding='utf-8') as f:
                pass
        else:
            # Regular JSON Schema mode
            template = env.get_template('json_schema.json.j2')
            file_path = self.build_generation_path(file_name="json_schema.json")

            with open(file_path, mode="w") as f:
                generated_code = template.render(
                    classes=self.model.classes_sorted_by_inheritance(),
                    enumerations = self.model.get_enumerations(),
                )
                f.write(generated_code)
                print("Code generated in the location: " + file_path)

    def generate_example(self, schema_str):
        """Generates an example JSON based on the rendered schema string."""
        schema = json.loads(schema_str)

        title = schema.get('title', 'Entity')
        match = re.search(r'Smart Data models - (\w+)', title)
        model_name = match.group(1) if match else title

        # Extract all properties
        all_props = {}
        if 'properties' in schema:
            all_props = schema['properties']
        else:
            for block in schema.get('allOf', []):
                if 'properties' in block:
                    all_props.update(block['properties'])

        # Build NGSI-LD example
        example_obj = {
            "id": f"urn:ngsi-ld:{model_name}:{model_name}-001",
            "type": model_name
        }

        for prop, details in all_props.items():
            ngsi_type, value = extract_example_and_type(prop, details)
            example_obj[prop] = {
                "type": ngsi_type,
                "value": value
            }

        return json.dumps(example_obj, indent=4)

    @staticmethod
    def tojson_ordered(value, indent=4, base_indent=0):
        """
        Serializa JSON ordenado con indentación adicional solo para líneas internas,
        manteniendo la primera línea sin sangría extra.

        Args:
            value (dict or list): La estructura JSON a serializar.
            indent (int): Espacios de indentación por nivel.
            base_indent (int): Espacios extra agregados a todas las líneas excepto la primera.

        Returns:
            str: Cadena JSON con indentación formateada correctamente.
        """
        json_str = json.dumps(value, indent=indent)
        lines = json_str.splitlines()

        if base_indent > 0 and len(lines) > 1:
            pad = ' ' * base_indent
            lines = [lines[0]] + [pad + line for line in lines[1:]]

        return '\n'.join(lines)

def extract_example_and_type(prop, details):
    """Infer NGSI type and value based on JSON schema, handling anyOf, array, and formats."""

    # anyOf: look for 'format' or 'type'
    if 'anyOf' in details:
        option = details['anyOf'][-1]  # Take the last option only
        ngsi_type, value = extract_example_and_type(prop, option)
        return ngsi_type, value

    # array: process items
    if details.get('type') == 'array':
        items = details.get('items', {})
        ngsi_type, item_value = extract_example_and_type(prop, items)
        return ngsi_type, [item_value]

    # Use type and format to determine NGSI type
    ngsi_type = get_ngsi_type(details)
    value = get_example_value(prop, details)

    return ngsi_type, value

def get_example_value(prop, details):
    """Returns a mock example value based on the schema property definition."""

    # Handle anyOf by recursively checking the first valid option
    if 'anyOf' in details:
        for option in details['anyOf']:
            val = get_example_value(prop, option)
            if val is not None:
                return val

    # Handle arrays by generating example for each item
    if details.get('type') == 'array':
        items = details.get('items', {})
        item_val = get_example_value(prop, items)
        return [item_val]

    t = details.get('type')
    fmt = details.get('format')
    enum = details.get('enum')

    if enum:
        return enum[0]

    # Generate values based on type/format
    if t == 'string':
        if fmt == 'uri':
            return f"urn:ngsi-ld:{prop}:{prop}-001"
        elif fmt == 'date':
            return "2023-10-01"
        elif fmt == 'date-time':
            return "2023-10-01T12:00:00Z"
        elif fmt == 'time':
            return "14:30:00Z"
        else:
            return f"example-{prop}"

    if t == 'integer':
        return 30

    if t == 'number':
        return 30 if "money" in prop.lower() else 3.14

    if t == 'boolean':
        return True

    if t == 'object':
        return {"exampleKey": "exampleValue"}

    return f"example-{prop}"

def get_ngsi_type(details):
    """Determines the NGSI type label based on JSON Schema field definition."""

    t = details.get('type')
    fmt = details.get('format')

    # Direct relationship: string + uri format
    if t == 'string' and fmt == 'uri':
        return 'Relationship'

    # Arrays may contain relationships via anyOf in items
    if t == 'array':
        items = details.get('items', {})

        # If items has anyOf, check if any option is uri
        if isinstance(items, dict) and 'anyOf' in items:
            for option in items['anyOf']:
                # If option is a relationship, return it
                if option.get('type') == 'string' and option.get('format') == 'uri':
                    return 'Relationship'

        # If items is directly a uri string type
        if isinstance(items, dict) and items.get('type') == 'string' and items.get('format') == 'uri':
            return 'Relationship'

        # Fallback for arrays
        return 'Property'

    # Format-based DateTime detection (handle case-insensitive)
    if fmt and fmt.lower() in ('date-time', 'datetime', 'date', 'time'):
        return 'DateTime'

    # Default mapping
    mapping = {
        'string': 'Text',
        'integer': 'Number',
        'number': 'Number',
        'boolean': 'Boolean',
        'object': 'StructuredValue'
    }

    return mapping.get(t, 'Property')
