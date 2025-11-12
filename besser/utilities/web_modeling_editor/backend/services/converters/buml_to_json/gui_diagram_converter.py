"""
GUI Diagram converter module for BUML to JSON conversion.
Handles GUI model structure processing and conversion to GrapesJS format.
"""

import re
from typing import Dict, Any, Optional


def gui_buml_to_json(buml_content: str) -> Dict[str, Any]:
    """
    Convert BUML GUI model Python code to GrapesJS JSON format.
    
    This is a simplified converter that preserves the GUI model structure
    by extracting the serialized data directly from the BUML code if available,
    or creates an empty GrapesJS structure.
    
    Args:
        buml_content: GUI model Python code as string
        
    Returns:
        Dictionary representing GrapesJS project data with pages structure
    """
    
    # Try to find serialized GUI data in comments or docstrings
    # (This would be used if we embed JSON data in the BUML export)
    serialized_match = re.search(r'# SERIALIZED_GUI_DATA: ({.*})', buml_content, re.DOTALL)
    if serialized_match:
        try:
            import json
            return json.loads(serialized_match.group(1))
        except Exception:
            pass
    
    # For now, we'll extract basic screen information and create a minimal GrapesJS structure
    # Future enhancement: fully parse the BUML AST to reconstruct the GUI
    
    pages = []
    
    # Try to extract screen definitions
    screen_pattern = r'(\w+)\s*=\s*gui_model\.new_screen\s*\(\s*name\s*=\s*["\']([^"\']+)["\']'
    screens = re.findall(screen_pattern, buml_content)
    
    if screens:
        for screen_var, screen_name in screens:
            # Create a basic page structure for each screen
            page = {
                "name": screen_name,
                "frames": [
                    {
                        "component": {
                            "type": "wrapper",
                            "stylable": [
                                "background",
                                "background-color",
                                "background-image",
                                "background-repeat",
                                "background-attachment",
                                "background-position",
                                "background-size"
                            ],
                            "attributes": {
                                "id": f"wrapper_{screen_var}"
                            },
                            "components": []
                        }
                    }
                ]
            }
            pages.append(page)
    
    # If no screens found, create a default empty page
    if not pages:
        pages = [
            {
                "name": "Home",
                "frames": [
                    {
                        "component": {
                            "type": "wrapper",
                            "stylable": [
                                "background",
                                "background-color",
                                "background-image",
                                "background-repeat",
                                "background-attachment",
                                "background-position",
                                "background-size"
                            ],
                            "attributes": {
                                "id": "wrapper_home"
                            },
                            "components": []
                        }
                    }
                ]
            }
        ]
    
    # Return GrapesJS-compatible structure
    return {
        "version": "3.0.0",
        "type": "GUINoCodeDiagram",
        "size": {"width": 1400, "height": 740},
        "pages": pages,
        "styles": [],
        "assets": [],
        "symbols": []
    }


def extract_gui_section(content: str) -> str:
    """
    Extract the GUI MODEL section from BUML project content.
    
    Args:
        content: Full project BUML content
        
    Returns:
        GUI model section content
    """
    # Match GUI MODEL section
    pattern = r'# GUI MODEL #(.*?)(?:# (?:STRUCTURAL|OBJECT|AGENT|STATE MACHINE|PROJECT DEFINITION)|$)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return ""


def parse_gui_buml_content(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse BUML GUI model content and convert to JSON.
    
    Args:
        content: GUI model BUML Python code
        
    Returns:
        Dictionary with GUI model data or None if parsing fails
    """
    if not content or not content.strip():
        return None
    
    try:
        # Try to convert the BUML to JSON
        gui_json = gui_buml_to_json(content)
        return gui_json
    except Exception as e:
        print(f"Warning: Could not parse GUI BUML content: {str(e)}")
        return None

