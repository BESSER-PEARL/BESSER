"""
Attribute parsing utilities for converting JSON to BUML format.
"""

from besser.BUML.metamodel.structural import Enumeration, Class
from besser.utilities.web_modeling_editor.backend.constants.constants import VISIBILITY_MAP, VALID_PRIMITIVE_TYPES


def parse_attribute(attribute_name, domain_model=None):
    """Parse an attribute string to extract visibility, name, and type, removing any colons."""
    # Split the string by colon first to separate name and type
    name_type_parts = attribute_name.split(":")

    if len(name_type_parts) > 1:
        name_part = name_type_parts[0].strip()
        type_part = name_type_parts[1].strip()

        # Check for visibility symbol at start of name
        if name_part[0] in VISIBILITY_MAP:
            visibility = VISIBILITY_MAP[name_part[0]]
            name = name_part[1:].strip()
        else:
            # Existing split logic for space-separated visibility
            name_parts = name_part.split()
            if len(name_parts) > 1:
                visibility_symbol = name_parts[0] if name_parts[0] in VISIBILITY_MAP else "+"
                visibility = VISIBILITY_MAP.get(visibility_symbol, "public")
                name = name_parts[1]
            else:
                visibility = "public"
                name = name_parts[0]

        # Handle the type
        if domain_model and any(isinstance(t, (Enumeration, Class)) and t.name == type_part for t in domain_model.types):
            attr_type = type_part
        else:
            attr_type = VALID_PRIMITIVE_TYPES.get(type_part.lower(), None)
            if attr_type is None:
                raise ValueError(f"Invalid type: {type_part}")
    else:
        # Handle case without type specification
        parts = attribute_name.split()

        if len(parts) == 1:
            part = parts[0].strip()
            if part and part[0] in VISIBILITY_MAP:
                visibility = VISIBILITY_MAP[part[0]]
                name = part[1:].strip()
                attr_type = "str"
            else:
                visibility = "public"
                name = part
                attr_type = "str"
        else:
            visibility_symbol = parts[0] if parts[0] in VISIBILITY_MAP else "+"
            visibility = VISIBILITY_MAP.get(visibility_symbol, "public")
            name = parts[1]
            attr_type = "str"
    if not name:  # Skip if name is empty
        return None, None, None
    return visibility, name, attr_type
