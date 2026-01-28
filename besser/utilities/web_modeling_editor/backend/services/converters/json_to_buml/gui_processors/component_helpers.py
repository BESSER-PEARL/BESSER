"""
Helper functions for component detection and extraction.
"""

from typing import Any, Dict, List, Optional

from .constants import TEXT_TAGS


def has_menu_structure(component: Dict[str, Any]) -> bool:
    """
    Check if a component has menu-like structure (nav with links/list items).
    
    Args:
        component: GrapesJS component dict
        
    Returns:
        True if component appears to be a menu
    """
    components = component.get("components", []) or []
    if not components:
        return False
    
    # Check if it contains list structure (ul/ol with li) or direct links
    has_list = False
    has_links = False
    
    for child in components:
        child_tag = str(child.get("tagName", "")).lower()
        child_type = str(child.get("type", "")).lower()
        
        if child_tag in {"ul", "ol"}:
            has_list = True
        if child_tag == "a" or child_type in {"link", "link-button"}:
            has_links = True
        
        # Check nested items
        if child_tag == "li":
            li_children = child.get("components", []) or []
            for li_child in li_children:
                if str(li_child.get("tagName", "")).lower() == "a":
                    has_links = True
    
    return has_list or has_links


def extract_menu_items(component: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    """
    Extract menu items (labels and links) from a navigation component.
    
    Args:
        component: GrapesJS component dict
        
    Returns:
        List of menu item dicts with 'label' key
    """
    from .utils import extract_text_content
    
    menu_items = []
    components = component.get("components", []) or []
    
    def extract_from_link(link_comp):
        """Extract label from a link component."""
        label = extract_text_content(link_comp)
        if not label:
            attrs = link_comp.get("attributes", {})
            if isinstance(attrs, dict):
                label = attrs.get("title") or attrs.get("aria-label") or attrs.get("data-label")
        attrs = link_comp.get("attributes", {}) or {}
        url = None
        target = None
        rel = None
        if isinstance(attrs, dict):
            url = attrs.get("href") or attrs.get("data-href")
            target = attrs.get("target") or attrs.get("data-target")
            rel = attrs.get("rel")
        return {"label": (label or "Menu Item"), "url": url, "target": target, "rel": rel}
    
    def process_component(comp):
        """Recursively process components to find links."""
        tag = str(comp.get("tagName", "")).lower()
        comp_type = str(comp.get("type", "")).lower()
        
        # Direct link
        if tag == "a" or comp_type in {"link", "link-button"}:
            menu_items.append(extract_from_link(comp))
        
        # List item containing link
        elif tag == "li":
            children = comp.get("components", []) or []
            for child in children:
                process_component(child)
        
        # Container with children
        elif comp.get("components"):
            children = comp.get("components", []) or []
            for child in children:
                process_component(child)
    
    for comp in components:
        process_component(comp)
    
    return menu_items if menu_items else [{"label": "Menu", "url": None, "target": None, "rel": None}]


def has_data_binding(component: Dict[str, Any]) -> bool:
    """
    Check if a component has data binding attributes.
    
    Args:
        component: GrapesJS component dict
        
    Returns:
        True if component has data binding attributes
    """
    attributes = component.get("attributes", {})
    if not isinstance(attributes, dict):
        return False
    
    # Check for common data binding attributes
    data_attrs = {"data-source", "data-bind", "data-model", "data-collection", "data-entity"}
    return any(attr in attributes for attr in data_attrs)


def collect_input_fields_recursive(elements):
    """
    Recursively collect all InputField instances from a list of elements.
    
    Args:
        elements: List of ViewComponent instances
        
    Returns:
        Set of InputField instances
    """
    from besser.BUML.metamodel.gui import InputField, ViewContainer
    
    input_fields = set()
    for element in elements:
        if isinstance(element, InputField):
            input_fields.add(element)
        elif isinstance(element, ViewContainer) and hasattr(element, 'view_elements'):
            input_fields.update(collect_input_fields_recursive(element.view_elements))
    return input_fields


def extract_parameters_from_attributes(attributes: Dict[str, Any]) -> set:
    """
    Extract Parameter objects from data-param-* attributes.
    
    Args:
        attributes: Component attributes dict
        
    Returns:
        Set of Parameter objects
    """
    from besser.BUML.metamodel.gui.events_actions import Parameter
    
    parameters = set()
    if not isinstance(attributes, dict):
        return parameters
    
    for key, value in attributes.items():
        if key.startswith("data-param-") or key.startswith("param-"):
            param_name = key.replace("data-param-", "").replace("param-", "")
            parameters.add(Parameter(name=param_name, value=str(value)))
    
    return parameters
