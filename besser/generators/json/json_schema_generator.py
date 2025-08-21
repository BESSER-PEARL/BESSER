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
        # Check if it's an enumeration instance
        if hasattr(property_type, '__class__') and property_type.__class__.__name__ == 'Enumeration':
            return "string", None

        # For BUML types, they are usually type objects/classes, not instances
        # Check the actual type or class name
        if property_type == IntegerType or (hasattr(property_type, '__name__') and property_type.__name__ == 'IntegerType'):
            return "integer", None
        elif property_type == StringType or (hasattr(property_type, '__name__') and property_type.__name__ == 'StringType'):
            return "string", None
        elif property_type == BooleanType or (hasattr(property_type, '__name__') and property_type.__name__ == 'BooleanType'):
            return "boolean", None
        elif property_type == FloatType or (hasattr(property_type, '__name__') and property_type.__name__ == 'FloatType'):
            return "number", None
        elif property_type == DateTimeType or (hasattr(property_type, '__name__') and property_type.__name__ == 'DateTimeType'):
            return "string", "date-time"
        elif property_type == DateType or (hasattr(property_type, '__name__') and property_type.__name__ == 'DateType'):
            return "string", "date"
        elif property_type == TimeType or (hasattr(property_type, '__name__') and property_type.__name__ == 'TimeType'):
            return "string", "time"
        else:
            return "string", None

    def _get_property_description(self, attr, prop_type):
        """
        Generate proper description format for Smart Data Models.
        
        Args:
            attr: The attribute object
            prop_type: The JSON schema type
            
        Returns:
            str: Formatted description
        """
        # Map JSON schema types to schema.org URLs
        schema_org_mapping = {
            "integer": "https://schema.org/Number",
            "number": "https://schema.org/Number", 
            "string": "https://schema.org/Text",
            "boolean": "https://schema.org/Boolean"
        }
        
        model_url = schema_org_mapping.get(prop_type, "https://schema.org/Text")
        
        if attr.metadata and attr.metadata.description:
            return f"Property. Model:'{model_url}'. {attr.metadata.description}"
        else:
            return f"Property. Model:'{model_url}'. {attr.name} value"

    def _prepare_smart_data_schema_for_class(self, class_def):
        """
        Prepares schema data for Smart Data format for a specific class.
        
        Args:
            class_def: The class to prepare the schema for.
            
        Returns:
            dict: A dictionary containing the schema data.
        """
        # Build class-specific properties
        class_properties = {}
        
        # Add the mandatory type property
        class_properties["type"] = {
            "type": "string",
            "enum": [class_def.name],
            "description": "Property. NGSI Entity type"
        }
        
        # Process class attributes
        for attr in class_def.attributes:
            prop_type, prop_format = self._get_property_type(attr.type)
            prop_def = {"type": prop_type}

            if prop_format is not None:
                prop_def["format"] = prop_format

            # Use the proper description format
            prop_def["description"] = self._get_property_description(attr, prop_type)

            if hasattr(attr.type, '__class__') and attr.type.__class__.__name__ == 'Enumeration':
                prop_def["enum"] = [lit.name for lit in attr.type.literals]
                prop_def["type"] = "string"

            if hasattr(attr, 'multiplicity') and attr.multiplicity.max > 1:
                prop_def = {
                    "type": "array",
                    "items": prop_def
                }

            class_properties[attr.name] = prop_def

        # Process association ends for this class
        for association in self.model.associations:
            class_ends = [end for end in association.ends if end.type == class_def]

            if not class_ends:
                continue

            for class_end in class_ends:
                other_ends = [end for end in association.ends if end != class_end]
                if not other_ends:
                    continue

                other_end = other_ends[0]
                prop_name = other_end.name

                if not prop_name:
                    continue

                relationship_description = (
                    f"Relationship. Model:'https://schema.org/URL'. Reference to {other_end.type.name}"
                    if association.metadata and association.metadata.description
                    else f"Relationship. Model:'https://schema.org/URL'. Reference to {other_end.type.name}"
                )

                relationship_def = {
                    "description": relationship_description,
                    "type": "string",
                    "format": "uri"
                }

                # Handle multiplicity
                if hasattr(other_end, 'multiplicity') and other_end.multiplicity:
                    if other_end.multiplicity.max > 1:
                        class_properties[prop_name] = {
                            "type": "array",
                            "items": relationship_def
                        }
                    else:
                        class_properties[prop_name] = relationship_def
                else:
                    class_properties[prop_name] = relationship_def

        # Create the complete schema structure
        schema_data = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$schemaVersion": "0.0.1",
            "$id": f"https://smart-data-models.github.io/dataModel.{self.model.name}/{class_def.name}/schema.json",
            "title": f"Smart Data models - {class_def.name} schema",
            "modelTags": "",
            "description": (
                class_def.metadata.description if class_def.metadata and class_def.metadata.description 
                else f"This class represents {class_def.name} for smart data models implementation"
            ),
            "type": "object",
            "required": ["id", "type"],
            "allOf": [
                {
                    "$ref": "https://smart-data-models.github.io/data-models/common-schema.json#/definitions/GSMA-Commons"
                },
                {
                    "$ref": "https://smart-data-models.github.io/data-models/common-schema.json#/definitions/Location-Commons"
                },
                {
                    "properties": class_properties
                }
            ],
            "derivedFrom": "3GPP TS 28.541",
            "license": ""
        }

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

                # Create examples directory
                examples_dir = os.path.join(class_dir, "examples")
                os.makedirs(examples_dir, exist_ok=True)

                # Prepare schema data for this class
                schema_data = self._prepare_smart_data_schema_for_class(class_def)
                
                # Generate schema.json file
                file_path = os.path.join(class_dir, "schema.json")
                with open(file_path, mode="w", encoding='utf-8') as f:
                    generated_code = template.render(schema=schema_data)
                    f.write(generated_code)

                # Generate all required example files
                base_example = self.generate_base_example(class_def)
                
                # example.json (key-value format)
                example_path = os.path.join(examples_dir, "example.json")
                with open(example_path, mode="w", encoding='utf-8') as f:
                    f.write(json.dumps(base_example, indent=2))

                # example-normalized.json (NGSI v2 normalized format)
                normalized_example = self.generate_normalized_example(base_example)
                normalized_path = os.path.join(examples_dir, "example-normalized.json")
                with open(normalized_path, mode="w", encoding='utf-8') as f:
                    f.write(json.dumps(normalized_example, indent=2))

                # example.jsonld (JSON-LD format)
                jsonld_example = self.generate_jsonld_example(base_example)
                jsonld_path = os.path.join(examples_dir, "example.jsonld")
                with open(jsonld_path, mode="w", encoding='utf-8') as f:
                    f.write(json.dumps(jsonld_example, indent=2))

                # example-normalized.jsonld (NGSI-LD normalized format)
                normalized_jsonld_example = self.generate_normalized_jsonld_example(base_example)
                normalized_jsonld_path = os.path.join(examples_dir, "example-normalized.jsonld")
                with open(normalized_jsonld_path, mode="w", encoding='utf-8') as f:
                    f.write(json.dumps(normalized_jsonld_example, indent=2))

                # Create ADOPTERS.yaml and notes.yaml in each class directory
                adopters_file_path = os.path.join(class_dir, "ADOPTERS.yaml")
                notes_file_path = os.path.join(class_dir, "notes.yaml")

                with open(adopters_file_path, mode="w", encoding='utf-8') as f:
                    f.write("# List of adopters\n")

                with open(notes_file_path, mode="w", encoding='utf-8') as f:
                    f.write("# Notes about the data model\n")

                print(f"Smart Data schema for {class_def.name} generated in: {file_path}")
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

    def generate_base_example(self, class_def):
        """Generate a base example in key-value format."""
        example = {
            "id": f"urn:ngsi-ld:{class_def.name}:{class_def.name}-001",
            "type": class_def.name
        }

        # Add common Smart Data Model properties
        example.update({
            "name": "example-name",
            "description": "example-description", 
            "dateCreated": "2025-06-25T08:00:00Z",
            "dateModified": "2025-06-30T14:15:00Z",
            "source": "https://example.org/source"
        })

        # Add class attributes
        for attr in class_def.attributes:
            example[attr.name] = self._get_example_value_for_attribute(attr)

        # Add association relationships
        for association in self.model.associations:
            class_ends = [end for end in association.ends if end.type == class_def]
            if not class_ends:
                continue

            for class_end in class_ends:
                other_ends = [end for end in association.ends if end != class_end]
                if not other_ends:
                    continue

                other_end = other_ends[0]
                if other_end.name:
                    example[other_end.name] = f"urn:ngsi-ld:{other_end.name}:{other_end.name}-01"

        return example

    def generate_normalized_example(self, base_example):
        """Generate NGSI v2 normalized example."""
        normalized = {}
        
        for key, value in base_example.items():
            if key in ["id", "type"]:
                normalized[key] = value
            else:
                # Determine the NGSI type based on the value
                if isinstance(value, str) and value.startswith("urn:ngsi-ld:"):
                    normalized[key] = {
                        "type": "Relationship",
                        "value": value
                    }
                elif isinstance(value, str) and ("date" in key.lower() or "time" in key.lower()):
                    normalized[key] = {
                        "type": "DateTime", 
                        "value": value
                    }
                elif isinstance(value, bool):
                    normalized[key] = {
                        "type": "Boolean",
                        "value": value
                    }
                elif isinstance(value, (int, float)):
                    normalized[key] = {
                        "type": "Number",
                        "value": value
                    }
                else:
                    normalized[key] = {
                        "type": "Text",
                        "value": value
                    }

        return normalized

    def generate_jsonld_example(self, base_example):
        """Generate JSON-LD example."""
        jsonld = base_example.copy()
        jsonld["@context"] = [
            "https://smartdatamodels.org/context.jsonld"
        ]
        return jsonld

    def generate_normalized_jsonld_example(self, base_example):
        """Generate NGSI-LD normalized example."""
        normalized_jsonld = {}
        
        for key, value in base_example.items():
            if key in ["id", "type"]:
                normalized_jsonld[key] = value
            else:
                # Create NGSI-LD format
                if isinstance(value, str) and value.startswith("urn:ngsi-ld:"):
                    normalized_jsonld[key] = {
                        "type": "Relationship",
                        "object": value
                    }
                elif isinstance(value, str) and ("date" in key.lower() or "time" in key.lower()):
                    normalized_jsonld[key] = {
                        "type": "Property",
                        "value": {
                            "@type": "DateTime",
                            "@value": value
                        }
                    }
                else:
                    normalized_jsonld[key] = {
                        "type": "Property",
                        "value": value
                    }

        normalized_jsonld["@context"] = [
            "https://smartdatamodels.org/context.jsonld"
        ]
        
        return normalized_jsonld

    def _get_example_value_for_attribute(self, attr):
        """Get example value for a specific attribute."""
        if hasattr(attr.type, '__class__') and attr.type.__class__.__name__ == 'Enumeration':
            return next(iter(attr.type.literals)).name if attr.type.literals else "enum-value"
        elif attr.type == IntegerType or (hasattr(attr.type, '__name__') and attr.type.__name__ == 'IntegerType'):
            return 10
        elif attr.type == FloatType or (hasattr(attr.type, '__name__') and attr.type.__name__ == 'FloatType'):
            return 3.14
        elif attr.type == BooleanType or (hasattr(attr.type, '__name__') and attr.type.__name__ == 'BooleanType'):
            return True
        elif attr.type == DateTimeType or (hasattr(attr.type, '__name__') and attr.type.__name__ == 'DateTimeType'):
            return "2025-06-25T08:00:00Z"
        elif attr.type == DateType or (hasattr(attr.type, '__name__') and attr.type.__name__ == 'DateType'):
            return "2025-06-25"
        elif attr.type == TimeType or (hasattr(attr.type, '__name__') and attr.type.__name__ == 'TimeType'):
            return "08:00:00Z"
        else:  # StringType or default
            return f"example-{attr.name}"

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
