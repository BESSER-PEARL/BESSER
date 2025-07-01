import os
import json
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import (
    DomainModel, IntegerType, StringType, Class, 
    BooleanType, FloatType, Enumeration
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
        Maps B-UML types to JSON schema types.
        
        Args:
            property_type: The B-UML type to map.
            
        Returns:
            str: The corresponding JSON schema type.
        """
        type_mapping = {
            IntegerType: "integer",
            StringType: "string",
            BooleanType: "boolean",
            FloatType: "number",
            list: "array"
        }
        
        # Check if property_type is a datatype with a name attribute
        if hasattr(property_type, 'name'):
            if property_type.name == 'str':
                return "string"
            elif property_type.name == 'int':
                return "integer"
            elif property_type.name == 'float':
                return "number"
            elif property_type.name == 'bool':
                return "boolean"
            return property_type.name
        
        if isinstance(property_type, Enumeration):
            return "string"
        return type_mapping.get(type(property_type), "string")

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
        if hasattr(class_def, 'synonyms') and class_def.synonyms:
            description = ". ".join(class_def.synonyms) if isinstance(class_def.synonyms, list) else class_def.synonyms
            schema_data["model_description"] = description
        
        # Process class attributes
        for attr in class_def.attributes:
            prop_type = self._get_property_type(attr.type)
            prop_def = {
                "type": prop_type,
                "description": "Property"
            }

            if hasattr(attr, 'synonyms') and attr.synonyms:
                prop_def["description"] = (
                    ". ".join(attr.synonyms) if isinstance(attr.synonyms, list)
                    else attr.synonyms
                )

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
            
            if hasattr(attr, 'required') and attr.required:
                schema_data["required"].append(attr.name)
        
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
                
                prop_def = {
                    "type": "string",
                    "format": "uri",
                    "description": f"Relationship to {other_end.type.name}"
                }
                
                if hasattr(other_end, 'multiplicity') and other_end.multiplicity.max > 1:
                    prop_def = {
                        "type": "array",
                        "items": prop_def
                    }
                
                schema_data["properties"][prop_name] = prop_def
                
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
                print(schema_data)
                # Generate schema file in the class directory
                file_path = os.path.join(class_dir, "schema.json")

                with open(file_path, mode="w", encoding='utf-8') as f:
                    generated_code = template.render(schema=schema_data)
                    f.write(generated_code)

                file_example_path = os.path.join(class_dir, "example.json")
                with open(file_example_path, mode="w", encoding='utf-8') as f:
                    generated_example_code = self.generate_example(generated_code)
                    f.write(generated_example_code)

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

        model_name = schema.get('title', 'Entity').split()[-1]

        # Collect properties from 'allOf' or 'properties'
        properties = {}
        if 'properties' in schema:
            all_props = schema.get('properties', {})
        else:
            # Assume Smart Data Models style with 'allOf' blocks
            all_props = {}
            for block in schema.get('allOf', []):
                if 'properties' in block:
                    all_props.update(block['properties'])

        # Add known GSMA-Commons/Location-Commons fields manually if not already present
        commons_defaults = {
            "name": {"type": "string", "example": f"{model_name}-001"},
            "description": {"type": "string", "example": f"Example instance of {model_name}"},
            "location": {
                "type": "object",
                "example": {
                    "type": "Point",
                    "coordinates": [12.4924, 41.8902]
                }
            },
            "address": {
                "type": "object",
                "example": {
                    "streetAddress": "Via Example 123",
                    "addressLocality": "Rome",
                    "addressRegion": "Lazio",
                    "postalCode": "00100",
                    "addressCountry": "IT"
                }
            },
            "dateCreated": {"type": "string", "example": "2025-06-25T08:00:00Z"},
            "dateModified": {"type": "string", "example": "2025-06-30T14:15:00Z"},
            "source": {"type": "string", "example": "https://example.org/source"}
        }

        for prop, details in commons_defaults.items():
            if prop not in all_props:
                all_props[prop] = details

        # Generate example values
        example_obj = {
            "id": f"urn:ngsi-ld:{model_name}:{model_name}-001",
            "type": model_name
        }

        for prop, details in all_props.items():
            example_obj[prop] = get_example_value(prop, details)

        # Pretty-print the example
        return json.dumps(example_obj, indent=2)

def get_example_value(prop, details):
    """Generate an example value based on property type and format."""
    COMMON_EXAMPLES = {
        "location": {
            "type": "Point",
            "coordinates": [12.4924, 41.8902]
        },
        "address": {
            "streetAddress": "Via Example 123",
            "addressLocality": "Rome",
            "addressRegion": "Lazio",
            "postalCode": "00100",
            "addressCountry": "IT"
        },
        "dateCreated": "2025-06-25T08:00:00Z",
        "dateModified": "2025-06-30T14:15:00Z",
        "source": "https://example.org/source"
    }

    if prop in COMMON_EXAMPLES:
        return COMMON_EXAMPLES[prop]

    t = details.get('type')
    fmt = details.get('format')
    enum = details.get('enum')

    if enum:
        return enum[0]
    if t == 'string':
        if fmt == 'uri':
            return f"urn:ngsi-ld:{prop}:{prop}-01"
        return f"example-{prop}"
    elif t == 'integer':
        return 10123 if "id" in prop.lower() else 10
    elif t == 'array':
        item_fmt = details.get('items', {}).get('format')
        if item_fmt == 'uri':
            return [f"urn:ngsi-ld:{prop}:{prop}-001"]
        return []
    elif t == 'boolean':
        return True
    elif t == 'object':
        return {}
    elif t == 'number':
        return 3.14
    return None
