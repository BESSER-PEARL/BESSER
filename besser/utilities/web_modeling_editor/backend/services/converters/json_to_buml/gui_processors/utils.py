"""
Utility functions for GUI diagram processing.
"""

import re
from typing import Any, Dict, Optional


def sanitize_name(raw_name: Optional[str]) -> str:
    """
    Convert an arbitrary string into a valid BUML element name.
    
    Args:
        raw_name: The raw name to sanitize
        
    Returns:
        A valid BUML element name
    """
    if not raw_name:
        return ""
    name = str(raw_name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    if not name:
        return ""
    if name[0].isdigit():
        name = f"_{name}"
    return name


def parse_style_string(style_string: str) -> Dict[str, str]:
    """
    Parse an inline CSS string into a dictionary of property/value pairs.
    
    Args:
        style_string: CSS style string (e.g., "color: red; font-size: 14px")
        
    Returns:
        Dictionary of CSS properties
    """
    properties: Dict[str, str] = {}
    if not style_string:
        return properties

    for part in style_string.split(";"):
        if not part.strip():
            continue
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        properties[key.strip().lower()] = value.strip()
    return properties


def parse_bool(value: Any, default: bool = True) -> bool:
    """
    Parse a boolean value from various input types.
    
    Args:
        value: Value to parse (bool, str, None)
        default: Default value if parsing fails
        
    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() == 'true'
    return bool(value)


def clean_attribute_name(attr_text: str) -> str:
    """
    Cleans attribute names by removing visibility and type annotations.
    
    Args:
        attr_text: Raw attribute text
        
    Returns:
        Cleaned attribute name
    """
    text = re.sub(r'^[\+\-\#]\s*', '', attr_text)  # remove + - #
    text = re.sub(r'\s*:\s*.*$', '', text)  # remove type annotation
    return text.strip()


def get_element_by_id(class_model, element_id):
    """
    Resolve an element by its ID from the class_model.
    
    Args:
        class_model: The class model (dict or list)
        element_id: The element ID to find
        
    Returns:
        The element dict or None if not found
    """
    if not class_model or not element_id:
        return None
    
    if isinstance(class_model, dict):
        if element_id in class_model:
            return class_model[element_id]
        
        # If this is a full diagram dict, look under 'elements'
        if 'elements' in class_model and isinstance(class_model['elements'], dict):
            if element_id in class_model['elements']:
                return class_model['elements'][element_id]
    
    # List format
    if isinstance(class_model, list):
        for el in class_model:
            if el.get('id') == element_id:
                return el
    
    return None


def extract_text_content(component: Dict[str, Any]) -> str:
    """
    Collect text content from a GrapesJS component (including nested text nodes).
    Converts <br> tags to newlines. Preserves nested HTML structure for nested text elements.
    
    Args:
        component: GrapesJS component dict
        
    Returns:
        Extracted text content
    """
    from .constants import TEXT_TAGS
    
    if not isinstance(component, dict):
        return ""

    comp_type = str(component.get("type", "")).lower()
    if comp_type == "textnode":
        return str(component.get("content", "")).strip()

    content = ""
    if component.get("content"):
        content += str(component.get("content"))

    # Check if component has nested children
    children = component.get("components", []) or []
    if children:
        # Check if any child is a text element (not just textnode)
        has_text_elements = any(
            isinstance(child, dict) and (
                str(child.get("type", "")).lower() in {"text", "textnode"} or
                str(child.get("tagName", "")).lower() in TEXT_TAGS
            )
            for child in children
        )
        
        if has_text_elements:
            # Build HTML from nested structure
            for child in children:
                if not isinstance(child, dict):
                    continue
                    
                child_type = str(child.get("type", "")).lower()
                child_tag = str(child.get("tagName", "")).lower()
                
                if child_type == "textnode":
                    content += str(child.get("content", ""))
                elif child_tag == "br":
                    # Convert <br> to newline
                    content += "\n"
                elif child_tag in TEXT_TAGS or child_type == "text":
                    # Wrap in HTML tag
                    tag = child_tag if child_tag else "span"
                    child_content = extract_text_content(child)
                    content += f"<{tag}>{child_content}</{tag}>"
                else:
                    content += extract_text_content(child)
        else:
            # Simple text aggregation for non-text children
            for child in children:
                child_tag = str(child.get("tagName", "")).lower() if isinstance(child, dict) else ""
                if child_tag == "br":
                    content += "\n"
                else:
                    content += extract_text_content(child)
    
    return content.strip()
