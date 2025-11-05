"""
Styling and CSS processing for GUI components.
"""

import copy
from typing import Any, Dict

from besser.BUML.metamodel.gui import (
    Alignment,
    Color,
    Layout,
    LayoutType,
    Position,
    PositionType,
    Size,
    Styling,
    UnitSize,
)


def infer_unit(value: str) -> UnitSize:
    """
    Infer a UnitSize from a CSS value.
    
    Args:
        value: CSS value string
        
    Returns:
        Corresponding UnitSize enum
    """
    if not isinstance(value, str):
        return UnitSize.PIXELS
    lowered = value.strip().lower()
    if lowered.endswith("%"):
        return UnitSize.PERCENTAGE
    if lowered.endswith("vh"):
        return UnitSize.VH
    if lowered.endswith("vw"):
        return UnitSize.VW
    if lowered.endswith("rem"):
        return UnitSize.REM
    if lowered.endswith("em"):
        return UnitSize.EM
    if lowered == "auto":
        return UnitSize.AUTO
    return UnitSize.PIXELS


def ensure_styling_parts(styling: Styling) -> Styling:
    """
    Make sure a Styling instance always has size, position and color objects attached.
    
    Args:
        styling: Styling object to validate
        
    Returns:
        Styling object with all required parts
    """
    if styling.size is None:
        styling.size = Size()
    if styling.position is None:
        styling.position = Position()
    if styling.color is None:
        styling.color = Color()
    return styling


def styling_from_css(style_dict: Dict[str, Any]) -> Styling:
    """
    Build a Styling object from a CSS dictionary.
    
    Args:
        style_dict: Dictionary of CSS properties
        
    Returns:
        Styling object
    """
    styling = Styling(size=Size(), position=Position(), color=Color())
    merge_styling_with_overrides(styling, style_dict)
    return styling


def merge_styling_with_overrides(styling: Styling, overrides: Dict[str, Any]) -> None:
    """
    Apply CSS overrides to a Styling object.
    Handles standard CSS properties plus flexbox and grid layout.
    
    Args:
        styling: Styling object to update
        overrides: Dictionary of CSS properties to apply
    """
    styling = ensure_styling_parts(styling)
    size = styling.size
    position = styling.position
    color = styling.color
    
    # Initialize layout properties storage
    layout_props = {
        'flex_direction': None,
        'justify_content': None,
        'align_items': None,
        'flex_wrap': None,
        'grid_template_columns': None,
        'grid_template_rows': None,
        'grid_gap': None,
        'justify_items': None,
        'gap': None,
        'display': None,
    }

    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, (int, float)):
            css_value = str(value)
        else:
            css_value = str(value).strip()
        lower_key = key.lower()
        
        # Standard sizing properties
        if lower_key in {"width", "min-width"}:
            size.width = css_value
            size.unit_size = infer_unit(css_value)
        elif lower_key in {"height", "min-height"}:
            size.height = css_value
        elif lower_key == "padding":
            size.padding = css_value
        elif lower_key == "margin":
            size.margin = css_value
        elif lower_key == "font-size":
            size.font_size = css_value
        elif lower_key == "line-height":
            size.line_height = css_value
        elif lower_key == "background" or lower_key == "background-color" or lower_key == "background-image":
            color.background_color = css_value
        elif lower_key == "color":
            color.text_color = css_value
        elif lower_key == "border-color":
            color.border_color = css_value
        elif lower_key == "opacity":
            color.opacity = css_value
        elif lower_key == "text-align":
            # Map CSS text-align values to Alignment enum
            alignment_map = {
                'left': Alignment.LEFT,
                'right': Alignment.RIGHT,
                'center': Alignment.CENTER,
                'justify': Alignment.LEFT,  # Map justify to LEFT as fallback
            }
            position.alignment = alignment_map.get(css_value.lower(), Alignment.LEFT)
        elif lower_key == "top":
            position.top = css_value
        elif lower_key == "left":
            position.left = css_value
        elif lower_key == "right":
            position.right = css_value
        elif lower_key == "bottom":
            position.bottom = css_value
        elif lower_key == "z-index":
            try:
                position.z_index = int(css_value)
            except ValueError:
                pass
        elif lower_key == "position":
            mapping = {
                "static": PositionType.STATIC,
                "relative": PositionType.RELATIVE,
                "absolute": PositionType.ABSOLUTE,
                "fixed": PositionType.FIXED,
                "sticky": PositionType.STICKY,
                "inline": PositionType.INLINE,
            }
            position.p_type = mapping.get(css_value.lower(), PositionType.STATIC)
        
        # Flexbox properties
        elif lower_key == "display":
            layout_props['display'] = css_value
        elif lower_key == "flex-direction":
            layout_props['flex_direction'] = css_value
        elif lower_key == "justify-content":
            layout_props['justify_content'] = css_value
        elif lower_key == "align-items":
            layout_props['align_items'] = css_value
        elif lower_key == "flex-wrap":
            layout_props['flex_wrap'] = css_value
        elif lower_key == "gap":
            layout_props['gap'] = css_value
        
        # Grid properties
        elif lower_key == "grid-template-columns":
            layout_props['grid_template_columns'] = css_value
        elif lower_key == "grid-template-rows":
            layout_props['grid_template_rows'] = css_value
        elif lower_key == "grid-gap":
            layout_props['grid_gap'] = css_value
        elif lower_key == "justify-items":
            layout_props['justify_items'] = css_value
    
    # Create Layout object if any layout properties are present
    if any(v is not None for v in layout_props.values() if v != layout_props['display']):
        layout_type = LayoutType.FLEX if layout_props['display'] == 'flex' else LayoutType.GRID
        styling.layout = Layout(
            layout_type=layout_type,
            flex_direction=layout_props['flex_direction'],
            justify_content=layout_props['justify_content'],
            align_items=layout_props['align_items'],
            flex_wrap=layout_props['flex_wrap'],
            grid_template_columns=layout_props['grid_template_columns'],
            grid_template_rows=layout_props['grid_template_rows'],
            grid_gap=layout_props['grid_gap'],
            justify_items=layout_props['justify_items'],
            gap=layout_props['gap'] or '16px',
        )


def parse_color(value, default="#000000"):
    """
    Parses a color value from a string or dict, returns hex or rgba string.
    
    Args:
        value: Color value (string or dict with r,g,b,a keys)
        default: Default color to return if parsing fails
        
    Returns:
        Color string in hex or rgba format
    """
    if isinstance(value, dict):
        r = value.get('r', 0)
        g = value.get('g', 0)
        b = value.get('b', 0)
        a = value.get('a', 1)
        if a != 1:
            return f"rgba({r},{g},{b},{a})"
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    if isinstance(value, str):
        return value
    return default


def build_style_map(styles_list) -> Dict[str, Styling]:
    """
    Build a style map keyed by selectors (id/class/type) from GrapesJS styles.
    
    Args:
        styles_list: List of style entries from GrapesJS
        
    Returns:
        Dictionary mapping selectors to Styling objects
    """
    style_map: Dict[str, Styling] = {}
    
    for style_entry in styles_list or []:
        selectors = style_entry.get("selectors", [])
        style = style_entry.get("style", {}) or {}
        styling = styling_from_css(style)

        for selector in selectors:
            key = None
            if isinstance(selector, dict):
                key = selector.get("name")
            elif isinstance(selector, str):
                key = selector
            if key:
                style_map[key] = styling
    
    return style_map


def resolve_component_styling(component: Dict[str, Any], style_map: Dict[str, Styling]) -> Styling:
    """
    Resolve styling for a component by merging class styles, type styles, and inline styles.
    
    Args:
        component: GrapesJS component dict
        style_map: Map of selectors to Styling objects
        
    Returns:
        Resolved Styling object
    """
    from .utils import parse_style_string
    
    base = None
    attributes = component.get("attributes")

    # Try ID selector first (#id)
    if isinstance(attributes, dict):
        comp_id = attributes.get("id")
        if comp_id and f"#{comp_id}" in style_map:
            base = copy.deepcopy(style_map[f"#{comp_id}"])

    # Try class selectors (.class)
    if base is None:
        classes = component.get("classes") or []
        for cls_item in classes:
            # Handle both string and dict class names
            cls_name = None
            if isinstance(cls_item, dict):
                cls_name = cls_item.get("name")
            elif isinstance(cls_item, str):
                cls_name = cls_item
            
            if cls_name:
                for selector in (cls_name, f".{cls_name}"):
                    if selector in style_map:
                        base = copy.deepcopy(style_map[selector])
                        break
                if base:
                    break

    # Try type selector (button, div, etc.)
    if base is None:
        type_selector = component.get("type")
        if type_selector and type_selector in style_map:
            base = copy.deepcopy(style_map[type_selector])

    # DON'T create default styling objects - they pollute the output
    # Only create styling if we have actual styles from GrapesJS
    if base is None:
        # Check if component has any inline styles before creating Styling object
        has_inline = False
        inline_style = component.get("style")
        if inline_style:
            has_inline = True
        if isinstance(attributes, dict):
            if attributes.get("style"):
                has_inline = True
            # Check for direct style attributes
            for key in ("width", "height", "min-height", "padding", "margin",
                       "background", "background-color", "color", "text-align"):
                if attributes.get(key):
                    has_inline = True
                    break
        
        if not has_inline:
            # No styles found - return None to signal "no styling"
            return None
        
        # Has inline styles but no base - create minimal styling
        base = Styling(size=Size(), position=Position(), color=Color())
    else:
        base = ensure_styling_parts(base)

    # Apply inline style overrides
    overrides: Dict[str, Any] = {}
    inline_style = component.get("style")
    if inline_style:
        overrides.update(parse_style_string(inline_style))

    if isinstance(attributes, dict):
        attr_style = attributes.get("style")
        if attr_style:
            overrides.update(parse_style_string(attr_style))
        # Direct attribute overrides
        for key in (
            "width", "height", "min-height", "padding", "margin",
            "background", "background-color", "color", "text-align",
        ):
            if attributes.get(key):
                overrides[key] = attributes.get(key)

    merge_styling_with_overrides(base, overrides)
    return base
