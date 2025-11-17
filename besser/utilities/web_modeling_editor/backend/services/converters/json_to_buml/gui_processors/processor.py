"""
Main GUI diagram processor - orchestrates the conversion of GrapesJS JSON to BUML GUI model.
"""

from typing import Any, Dict, List, Optional, Set

from besser.BUML.metamodel.gui import (
    Button,
    ButtonActionType,
    GUIModel,
    Module,
    Screen,
    ViewComponent,
    ViewContainer,
)
from besser.BUML.metamodel.gui.events_actions import (
    Create,
    Delete,
    Read,
    Transition,
    Update,
)

from .chart_parsers import (
    apply_chart_colors,
    parse_bar_chart,
    parse_line_chart,
    parse_metric_card,
    parse_pie_chart,
    parse_radar_chart,
    parse_radial_bar_chart,
    parse_table_chart,
)
from .component_helpers import has_data_binding, has_menu_structure
from .component_parsers import (
    parse_button,
    parse_container,
    parse_data_list,
    parse_embedded_content,
    parse_form,
    parse_generic_component,
    parse_image,
    parse_input_field,
    parse_link,
    parse_menu,
    parse_text,
)
from .constants import CONTAINER_TAGS, CONTAINER_TYPES, INPUT_COMPONENT_TYPES, TEXT_TAGS
from .styling import build_style_map, resolve_component_styling
from .utils import sanitize_name


def process_gui_diagram(gui_diagram, class_model, domain_model):
    """
    Main entry point: Converts GrapesJS JSON and domain_model into a GUIModel instance.
    Handles style mapping, screens, and recursive component parsing.
    
    Args:
        gui_diagram: The GUI diagram data from GrapesJS
        class_model: The class model for object resolution
        domain_model: The domain metamodel for object resolution
        
    Returns:
        GUIModel instance with screens and components
    """
    gui_model_json = gui_diagram or {}
    
    # Build style map from GrapesJS styles
    style_map = build_style_map(gui_model_json.get("styles", []))
    
    # Track used names to ensure uniqueness
    used_names: Set[str] = set()

    def register_name(raw_value: Optional[str], fallback: str) -> str:
        """Register and return a unique name."""
        sanitized = sanitize_name(raw_value)
        if not sanitized:
            sanitized = sanitize_name(fallback)
        if not sanitized:
            sanitized = fallback or "Element"
        candidate = sanitized
        counter = 1
        while candidate in used_names:
            counter += 1
            candidate = f"{sanitized}_{counter}"
        used_names.add(candidate)
        return candidate

    def get_unique_name(component: Optional[Dict[str, Any]], fallback: str) -> str:
        """Extract and register a unique name from component metadata, preserving original IDs."""
        if not component:
            return register_name(fallback, fallback)

        attributes = component.get("attributes") if isinstance(component.get("attributes"), dict) else {}
        # Prefer explicit id or name from JSON
        for key in ("id", "name", "data-name", "label", "title", "chart-title"):
            if attributes.get(key):
                return register_name(attributes.get(key), fallback)

        # Fallback to component-level name or tag
        for key in ("name", "tagName", "type"):
            value = component.get(key)
            if isinstance(value, str) and value:
                return register_name(value, fallback)

        # Fallback to custom displayName
        custom = component.get("custom")
        if isinstance(custom, dict) and custom.get("displayName"):
            return register_name(custom.get("displayName"), fallback)

        return register_name(fallback, fallback)

    def attach_meta(element, meta: Dict[str, Any]) -> None:
        """
        Attach comprehensive metadata to component for code generation fidelity.
        This function is a SAFETY NET that fills in missing metadata after component parsing.
        
        IMPORTANT: This function should NOT overwrite metadata already set by specialized parsers
        (component_parsers.py). It only fills in gaps for metadata that wasn't set during 
        component-specific parsing.
        
        Design Pattern:
        1. Specialized parsers (parse_button, parse_text, etc.) set metadata via _attach_component_metadata()
        2. This function (attach_meta) is called AFTER parsing as a fallback
        3. Only sets values if they are None or empty (never overwrites existing data)
        
        This ensures ALL JSON data is preserved for perfect React regeneration.
        """
        if not element:
            return
        
        # Store tag name ONLY if not already set
        if not hasattr(element, 'tag_name') or not element.tag_name:
            element.tag_name = meta.get("tagName")
        
        # Store CSS classes (normalize to list of strings) ONLY if not already set
        if not hasattr(element, 'css_classes') or not element.css_classes:
            element.css_classes = []
            if meta.get("classList"):
                element.css_classes = [
                    cls if isinstance(cls, str) else cls.get("name", "") 
                    for cls in meta["classList"]
                ]
        
        # Store custom attributes ONLY if not already set
        if not hasattr(element, 'custom_attributes') or not element.custom_attributes:
            element.custom_attributes = {}
            if meta.get("attributes"):
                element.custom_attributes = dict(meta["attributes"]) if isinstance(meta["attributes"], dict) else {}
        
        # Store component_id ONLY if not already set
        # Check for None explicitly since hasattr returns True even if value is None
        if not hasattr(element, 'component_id') or element.component_id is None:
            if meta.get("attributes") and isinstance(meta["attributes"], dict):
                element.component_id = meta["attributes"].get("id")
            if not element.component_id and element.custom_attributes:
                element.component_id = element.custom_attributes.get("id")
        
        # Store component_type ONLY if not already set
        if not hasattr(element, 'component_type') or element.component_type is None:
            element.component_type = meta.get("type")

    def parse_component_list(components: Optional[List[Dict[str, Any]]]) -> List[ViewComponent]:
        """Parse a list of GrapesJS components into BUML ViewComponents."""
        parsed_children: List[ViewComponent] = []
        for index, child in enumerate(components or []):
            result = parse_component_node(child)
            if not result:
                continue
            if isinstance(result, list):
                for sub_index, item in enumerate(result):
                    # Preserve JSON order with display_order
                    if not hasattr(item, 'display_order'):
                        setattr(item, 'display_order', index + (sub_index * 0.01))
                    else:
                        item.display_order = index + (sub_index * 0.01)
                parsed_children.extend(result)
            else:
                # Preserve JSON order with display_order
                if not hasattr(result, 'display_order'):
                    setattr(result, 'display_order', index)
                else:
                    result.display_order = index
                parsed_children.append(result)
        return parsed_children

    def parse_component_node(component: Dict[str, Any]):
        """
        Parse a single GrapesJS component into a BUML ViewComponent.
        Dispatches to specialized parsers based on component type.
        """
        if not isinstance(component, dict):
            return None

        comp_type = str(component.get("type", "")).lower()
        tag = str(component.get("tagName", "")).lower()
        comp_name = str(component.get("name", "")).lower()

        # Skip wrapper, textnode, and comment nodes
        if comp_type == "wrapper":
            return parse_component_list(component.get("components"))
        if comp_type in ("textnode", "comment"):
            return None

        # Resolve styling
        styling = resolve_component_styling(component, style_map)
        
        # Prepare metadata - preserve ALL JSON data for fidelity
        attributes = component.get("attributes") if isinstance(component.get("attributes"), dict) else {}
        class_list = component.get("classes") or []
        meta = {
            "type": component.get("type"),  # Store original component type
            "tagName": component.get("tagName"),
            "classList": class_list,
            "attributes": attributes,
            "inlineStyle": component.get("style"),
        }
        
        # Set default tagName for known types
        if not meta["tagName"]:
            default_tags = {
                "button": "button",
                "link": "a",
                "link-button": "a",
                "checkbox": "input",
                "map": "iframe",
                "tabs": "div",
                "tab-container": "div",
                "tab-contents": "div",
                "tab-content": "div",
                "tab": "button",
                "comment": "div",
            }
            meta["tagName"] = default_tags.get(comp_type)
        
        # Handle iframe/img src attribute
        if meta["tagName"] in {"iframe", "img"} and isinstance(attributes, dict):
            attributes.setdefault("src", component.get("src"))

        # === CHART PARSERS ===
        if comp_type == "line-chart":
            element = parse_line_chart(component, class_model, domain_model)
            if element:
                if styling:  # Only assign if styling exists
                    element.styling = styling
                element.name = get_unique_name(component, element.name or "LineChart")
                apply_chart_colors(element, attributes)
                attach_meta(element, meta)
            return element

        if comp_type == "bar-chart":
            element = parse_bar_chart(component, class_model, domain_model)
            if element:
                if styling:  # Only assign if styling exists
                    element.styling = styling
                element.name = get_unique_name(component, element.name or "BarChart")
                apply_chart_colors(element, attributes)
                attach_meta(element, meta)
            return element

        if comp_type == "pie-chart":
            element = parse_pie_chart(component, class_model, domain_model)
            if element:
                if styling:  # Only assign if styling exists
                    element.styling = styling
                element.name = get_unique_name(component, element.name or "PieChart")
                apply_chart_colors(element, attributes)
                attach_meta(element, meta)
            return element

        if comp_type == "radar-chart":
            element = parse_radar_chart(component, class_model, domain_model)
            if element:
                if styling:  # Only assign if styling exists
                    element.styling = styling
                element.name = get_unique_name(component, element.name or "RadarChart")
                apply_chart_colors(element, attributes)
                attach_meta(element, meta)
            return element

        if comp_type == "radial-bar-chart":
            element = parse_radial_bar_chart(component, class_model, domain_model)
            if element:
                if styling:  # Only assign if styling exists
                    element.styling = styling
                element.name = get_unique_name(component, element.name or "RadialBarChart")
                apply_chart_colors(element, attributes)
                attach_meta(element, meta)
            return element

        if comp_type == "table-chart":
            element = parse_table_chart(component, class_model, domain_model)
            if element:
                if styling:
                    element.styling = styling
                element.name = get_unique_name(component, element.name or "TableChart")
                apply_chart_colors(element, attributes)
                attach_meta(element, meta)
            return element

        if comp_type == "metric-card":
            element = parse_metric_card(component, class_model, domain_model)
            if element:
                if styling:  # Only assign if styling exists
                    element.styling = styling
                element.name = get_unique_name(component, element.name or "MetricCard")
                attach_meta(element, meta)
            return element

        # === BUTTON PARSER ===
        if comp_type in {"button", "action-button"} or tag == "button":
            name = get_unique_name(component, "Button")
            button = parse_button(component, styling, name, meta)
            attach_meta(button, meta)
            return button

        # === INPUT FIELD PARSER ===
        if comp_type in INPUT_COMPONENT_TYPES or tag in INPUT_COMPONENT_TYPES:
            name = get_unique_name(component, "InputField")
            input_field = parse_input_field(component, styling, name, meta)
            attach_meta(input_field, meta)
            return input_field

        # === FORM PARSER ===
        if comp_type == "form" or tag == "form":
            name = get_unique_name(component, "Form")
            form = parse_form(component, styling, name, meta, parse_component_list)
            attach_meta(form, meta)
            return form

        # === TEXT PARSER ===
        if comp_type == "text" or tag in TEXT_TAGS:
            name = get_unique_name(component, "Text")
            text_element = parse_text(component, styling, name, meta)
            attach_meta(text_element, meta)
            return text_element

        # === IMAGE PARSER ===
        if tag == "img" or comp_type == "image":
            name = get_unique_name(component, "Image")
            image = parse_image(component, styling, name, meta)
            attach_meta(image, meta)
            return image

        # === LINK PARSER ===
        if comp_type in {"link", "link-button"} or tag == "a":
            name = get_unique_name(component, "Link")
            link = parse_link(component, styling, name, meta)
            attach_meta(link, meta)
            return link

        # === EMBEDDED CONTENT PARSER ===
        if comp_type == "map" or tag == "iframe":
            name = get_unique_name(component, "EmbeddedContent")
            embedded = parse_embedded_content(component, styling, name, meta)
            attach_meta(embedded, meta)
            return embedded

        # === MENU PARSER ===
        # Check for nav tags or ul/ol with link structure
        if comp_type == "menu" or (tag == "nav" and has_menu_structure(component)) or \
           (tag in {"ul", "ol"} and has_menu_structure(component)):
            name = get_unique_name(component, "Menu")
            menu = parse_menu(component, styling, name, meta)
            attach_meta(menu, meta)
            return menu

        # === DATALIST PARSER ===
        if comp_type in {"list", "data-list", "table"} or (tag in {"ul", "ol", "table"} and has_data_binding(component)):
            name = get_unique_name(component, "DataList")
            data_list = parse_data_list(component, styling, name, meta, domain_model)
            attach_meta(data_list, meta)
            return data_list

        # === CONTAINER PARSER ===
        if comp_type in CONTAINER_TYPES or tag in CONTAINER_TAGS or comp_name in CONTAINER_TYPES:
            name = get_unique_name(component, "Container")
            container = parse_container(component, styling, name, meta, parse_component_list)
            attach_meta(container, meta)
            
            # Ensure ViewContainers have stable IDs even if JSON doesn't provide one
            if not container.component_id:
                # Generate stable ID from component name or position
                container.component_id = f"container_{name.lower()}"
            
            return container

        # === GENERIC COMPONENT ===
        name = get_unique_name(component, (tag or comp_type) or "Component")
        generic = parse_generic_component(component, styling, name, meta, parse_component_list)
        attach_meta(generic, meta)

        # Preserve original metadata for React generator fidelity
        if isinstance(generic, ViewComponent):
            attributes = component.get("attributes", {})
            generic.component_id = attributes.get("id") or component.get("id")
            generic.component_type = component.get("type")
            generic.tag_name = component.get("tagName") or meta.get("tagName")
            generic.css_classes = [cls if isinstance(cls, str) else cls.get("name", "") for cls in (component.get("classes") or [])]
            generic.custom_attributes = dict(attributes) if isinstance(attributes, dict) else {}
            # Preserve layout if available
            if hasattr(generic, "layout") and hasattr(styling, "layout") and styling.layout:
                generic.layout = styling.layout

        return generic

    # Extract GUI model name
    raw_title = gui_diagram.get("title", "GUI")
    gui_name = sanitize_name(raw_title) or "GUI"

    # Create GUI model
    gui_model = GUIModel(
        name=gui_name,
        package="",
        versionCode="1.0",
        versionName="1.0",
        modules=set(),
        description=str(raw_title or "Generated GUI"),
    )

    # Parse pages/screens
    pages = gui_model_json.get("pages", [])
    screen_list: List[Screen] = []
    page_id_to_screen: Dict[str, Screen] = {}  # Map page IDs to Screen objects

    for page_index, page in enumerate(pages):
        page_id = page.get("id")  # Get the page ID for target-screen resolution
        page_name = page.get("name", f"Page_{page_index + 1}")
        frames = page.get("frames", [])
        for frame_index, frame in enumerate(frames):
            wrapper = frame.get("component", {}) or {}
            screen_fallback = page_name or f"Screen_{page_index + 1}"
            if len(frames) > 1:
                screen_fallback = f"{screen_fallback}_{frame_index + 1}"
            screen_name = get_unique_name(wrapper, screen_fallback)
            
            # Extract screen properties
            page_attrs = page.get("attributes", {}) or {}
            screen_size = "Medium"
            x_dpi = ""
            y_dpi = ""
            
            if isinstance(page_attrs, dict):
                screen_size = page_attrs.get("screen-size") or page_attrs.get("viewport-size") or "Medium"
                x_dpi = str(page_attrs.get("x-dpi") or page_attrs.get("dpi") or "")
                y_dpi = str(page_attrs.get("y-dpi") or page_attrs.get("dpi") or "")
            
            # Derive route path from page name or ID
            route_path = f"/{(page.get('name') or screen_fallback).lower().replace(' ', '-')}"
            
            screen = Screen(
                name=screen_name,
                description=str(page.get("name") or screen_fallback),
                view_elements=set(),
                is_main_page=len(screen_list) == 0,
                x_dpi=x_dpi,
                y_dpi=y_dpi,
                screen_size=screen_size,
                route_path=route_path,
            )
            styling = resolve_component_styling(wrapper, style_map)
            if styling:  # Only assign if styling exists
                screen.styling = styling
            
            # Store page metadata for code generation
            if hasattr(screen, 'component_id'):
                screen.component_id = page_id
            else:
                setattr(screen, 'page_id', page_id)
            
            # Set layout if present in styling
            if hasattr(screen, 'styling') and screen.styling and hasattr(screen.styling, 'layout') and screen.styling.layout:
                screen.layout = screen.styling.layout
            
            children = parse_component_list(wrapper.get("components"))
            screen.view_elements = set(children)
            screen_list.append(screen)
            
            # Map page ID to screen for target-screen resolution
            if page_id:
                page_id_to_screen[page_id] = screen
                # Also map with "page:" prefix that GrapesJS uses
                page_id_to_screen[f"page:{page_id}"] = screen

    # Post-processing: Resolve Action references
    def resolve_action_references(elements):
        """Recursively resolve Action references (target_screen, target_class)."""
        for element in elements:
            # Check if element is a Button with events
            if isinstance(element, Button) and hasattr(element, 'events'):
                for event in element.events:
                    for action in event.actions:
                        # Resolve Transition target_screen
                        if isinstance(action, Transition) and hasattr(action, '_target_screen_id'):
                            target_id = getattr(action, '_target_screen_id')
                            # Try to resolve by page ID first (e.g., "page:nvPCYoWq2RVUadd9w")
                            target_screen = page_id_to_screen.get(target_id)
                            # Fall back to matching by screen name if ID lookup fails
                            if not target_screen:
                                target_screen = next((s for s in screen_list if s.name == target_id), None)
                            if target_screen:
                                action.target_screen = target_screen
                        
                        # Resolve CRUD target_class
                        if isinstance(action, (Create, Read, Update, Delete)) and hasattr(action, '_target_class_name'):
                            target_name = getattr(action, '_target_class_name')
                            target_class = domain_model.get_class_by_name(target_name) if domain_model else None
                            if target_class:
                                action.target_class = target_class
            
            # Recursively process children in containers
            if isinstance(element, ViewContainer) and hasattr(element, 'view_elements'):
                resolve_action_references(element.view_elements)
    
    # Apply resolution to all screens
    for screen in screen_list:
        if hasattr(screen, 'view_elements'):
            resolve_action_references(screen.view_elements)

    # Create module and finalize
    module_name = register_name(f"{gui_name}_Module", "Module")
    dashboard_module = Module(name=module_name, screens=set(screen_list))
    gui_model.modules = {dashboard_module}
    
    # Normalize style entries for template compatibility
    # Convert dictionary selectors to strings and add proper CSS prefixes
    normalized_styles = []
    for style_entry in gui_model_json.get("styles", []):
        normalized_entry = dict(style_entry)  # Copy the entry
        selectors = normalized_entry.get("selectors", [])
        normalized_selectors = []
        for selector in selectors:
            if isinstance(selector, dict):
                # Extract name from dictionary selector (class names from GrapesJS)
                name = selector.get("name")
                if name:
                    # Add . prefix for class selectors (if not already prefixed)
                    if not name.startswith(('.', '#', '*', '[')):
                        name = f".{name}"
                    normalized_selectors.append(name)
            elif isinstance(selector, str):
                # Preserve as-is (already has #id or .class prefix)
                normalized_selectors.append(selector)
        normalized_entry["selectors"] = normalized_selectors
        # Only include entries with valid selectors
        if normalized_selectors:
            normalized_styles.append(normalized_entry)
    
    # Store normalized style entries for React generator
    setattr(gui_model, "_style_entries", normalized_styles)
    
    return gui_model
