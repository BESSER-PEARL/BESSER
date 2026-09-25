"""
Styling and CSS processing for GUI components.
"""

import copy
import re
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

        # ── Size: dimensions ──
        if lower_key == "width":
            size.width = css_value
            size.unit_size = infer_unit(css_value)
        elif lower_key == "height":
            size.height = css_value
        elif lower_key == "min-width":
            size.min_width = css_value
        elif lower_key == "max-width":
            size.max_width = css_value
        elif lower_key == "min-height":
            size.min_height = css_value
        elif lower_key == "max-height":
            size.max_height = css_value
        elif lower_key == "padding":
            size.padding = css_value
        elif lower_key == "padding-top":
            size.padding_top = css_value
        elif lower_key == "padding-right":
            size.padding_right = css_value
        elif lower_key == "padding-bottom":
            size.padding_bottom = css_value
        elif lower_key == "padding-left":
            size.padding_left = css_value
        elif lower_key == "margin":
            size.margin = css_value
        elif lower_key == "margin-top":
            size.margin_top = css_value
        elif lower_key == "margin-right":
            size.margin_right = css_value
        elif lower_key == "margin-bottom":
            size.margin_bottom = css_value
        elif lower_key == "margin-left":
            size.margin_left = css_value
        elif lower_key == "font-size":
            size.font_size = css_value
        elif lower_key == "line-height":
            size.line_height = css_value

        # ── Size: typography ──
        elif lower_key == "font-weight":
            size.font_weight = css_value
        elif lower_key == "font-family":
            size.font_family = css_value
        elif lower_key == "font-style":
            size.font_style = css_value
        elif lower_key == "text-decoration":
            size.text_decoration = css_value
        elif lower_key == "text-transform":
            size.text_transform = css_value
        elif lower_key == "letter-spacing":
            size.letter_spacing = css_value
        elif lower_key == "word-spacing":
            size.word_spacing = css_value
        elif lower_key == "white-space":
            size.white_space = css_value
        elif lower_key == "word-break":
            size.word_break = css_value

        # ── Color: backgrounds ──
        elif lower_key in {"background", "background-color"}:
            color.background_color = css_value
        elif lower_key == "background-image":
            color.background_image = css_value
        elif lower_key == "background-size":
            color.background_size = css_value
        elif lower_key == "background-position":
            color.background_position = css_value
        elif lower_key == "background-repeat":
            color.background_repeat = css_value
        elif lower_key == "color":
            color.text_color = css_value
        elif lower_key == "opacity":
            color.opacity = css_value

        # ── Color: borders ──
        elif lower_key == "border":
            color.border = css_value
        elif lower_key == "border-color":
            color.border_color = css_value
        elif lower_key == "border-radius":
            color.border_radius = css_value
        elif lower_key == "border-width":
            color.border_width = css_value
        elif lower_key == "border-style":
            color.border_style = css_value
        elif lower_key == "border-top":
            color.border_top = css_value
        elif lower_key == "border-right":
            color.border_right = css_value
        elif lower_key == "border-bottom":
            color.border_bottom = css_value
        elif lower_key == "border-left":
            color.border_left = css_value

        # ── Color: shadows ──
        elif lower_key == "box-shadow":
            color.box_shadow = css_value
        elif lower_key == "text-shadow":
            color.text_shadow = css_value

        # ── Position: placement ──
        elif lower_key == "text-align":
            alignment_map = {
                'left': Alignment.LEFT,
                'right': Alignment.RIGHT,
                'center': Alignment.CENTER,
                'justify': Alignment.LEFT,
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

        # ── Position: display and box model ──
        elif lower_key == "display":
            layout_props['display'] = css_value
            # Also store on position for non-flex/grid values (block, inline, none, etc.)
            position.display = css_value
        elif lower_key == "overflow":
            position.overflow = css_value
        elif lower_key == "overflow-x":
            position.overflow_x = css_value
        elif lower_key == "overflow-y":
            position.overflow_y = css_value
        elif lower_key == "visibility":
            position.visibility = css_value
        elif lower_key == "cursor":
            position.cursor = css_value
        elif lower_key == "box-sizing":
            position.box_sizing = css_value

        # ── Position: effects ──
        elif lower_key == "transform":
            position.transform = css_value
        elif lower_key == "transition":
            position.transition = css_value
        elif lower_key == "animation":
            position.animation = css_value
        elif lower_key == "filter":
            position.filter = css_value

        # ── Layout: flex container ──
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

        # ── Layout: grid container ──
        elif lower_key == "grid-template-columns":
            layout_props['grid_template_columns'] = css_value
        elif lower_key == "grid-template-rows":
            layout_props['grid_template_rows'] = css_value
        elif lower_key == "grid-gap":
            layout_props['grid_gap'] = css_value
        elif lower_key == "justify-items":
            layout_props['justify_items'] = css_value

        # ── Layout: flex item ──
        elif lower_key == "flex":
            layout_props['flex'] = css_value
        elif lower_key == "flex-grow":
            layout_props['flex_grow'] = css_value
        elif lower_key == "flex-shrink":
            layout_props['flex_shrink'] = css_value
        elif lower_key == "flex-basis":
            layout_props['flex_basis'] = css_value
        elif lower_key == "order":
            layout_props['order'] = css_value
        elif lower_key == "align-self":
            layout_props['align_self'] = css_value

    # Separate container properties from item properties
    container_keys = {'flex_direction', 'justify_content', 'align_items', 'flex_wrap',
                      'grid_template_columns', 'grid_template_rows', 'grid_gap',
                      'justify_items', 'gap'}
    item_keys = {'flex', 'flex_grow', 'flex_shrink', 'flex_basis', 'order', 'align_self'}

    has_container = any(layout_props.get(k) for k in container_keys)
    has_item = any(layout_props.get(k) for k in item_keys)
    display_val = layout_props['display']
    is_flex_or_grid = display_val in ('flex', 'grid')

    # Create Layout if there are container properties OR explicit display:flex/grid
    if has_container or is_flex_or_grid or has_item:
        # Only set layout_type (which emits display:flex/grid) for containers
        layout_type = None
        if is_flex_or_grid or has_container:
            layout_type = LayoutType.GRID if display_val == 'grid' else LayoutType.FLEX

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
            gap=layout_props['gap'],
            flex=layout_props.get('flex'),
            flex_grow=layout_props.get('flex_grow'),
            flex_shrink=layout_props.get('flex_shrink'),
            flex_basis=layout_props.get('flex_basis'),
            order=layout_props.get('order'),
            align_self=layout_props.get('align_self'),
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


def _selector_name(selector) -> str:
    """Return a GrapesJS selector entry as CSS text (``#id`` or ``.class``)."""
    if isinstance(selector, dict):
        name = str(selector.get("name") or "")
        if selector.get("type") == 2 and not name.startswith("#"):
            name = f"#{name}"
    else:
        name = str(selector or "")
    if not name or name[0] in ".#":
        return name
    return "." + re.sub(r"([^\w-])", r"\\\1", name)


def _is_element_rule(style_entry: Dict[str, Any]) -> bool:
    """True for a plain rule on one element id: it becomes that element's Styling."""
    selectors = style_entry.get("selectors") or []
    return (
        len(selectors) == 1
        and _selector_name(selectors[0]).startswith("#")
        and not style_entry.get("selectorsAdd")
        and not style_entry.get("state")
        and not style_entry.get("mediaText")
        and not style_entry.get("atRuleType")
    )


def build_style_map(styles_list) -> Dict[str, Styling]:
    """
    Build a style map keyed by element id selector (``#id``) from GrapesJS styles.

    Every other rule (classes, compound selectors, ``:root``, ``@media``,
    states) is kept verbatim in the GUIModel stylesheet instead, see
    :func:`build_stylesheet`.

    Args:
        styles_list: List of style entries from GrapesJS

    Returns:
        Dictionary mapping id selectors to Styling objects
    """
    style_map: Dict[str, Styling] = {}

    for style_entry in styles_list or []:
        if _is_element_rule(style_entry):
            key = _selector_name(style_entry["selectors"][0])
            style_map[key] = styling_from_css(style_entry.get("style", {}) or {})

    return style_map


def build_stylesheet(styles_list) -> str:
    """
    Render every GrapesJS rule not tied to a single element id as CSS text.

    Rules keep their order; consecutive rules under the same at-rule share one
    block. One top-level block per line.

    Args:
        styles_list: List of style entries from GrapesJS

    Returns:
        The stylesheet, or an empty string when there is none
    """
    blocks = []  # (at_rule or None, [rule text])
    for style_entry in styles_list or []:
        if not isinstance(style_entry, dict) or _is_element_rule(style_entry):
            continue
        declarations = ";".join(
            f"{prop}:{value}"
            for prop, value in (style_entry.get("style") or {}).items()
            if not str(prop).startswith("__") and value not in (None, "")
        )
        at_type = style_entry.get("atRuleType") or ("media" if style_entry.get("mediaText") else "")
        if style_entry.get("singleAtRule") and at_type:
            blocks.append((None, [f"@{at_type}{{{declarations}}}"]))
            continue
        compound = "".join(_selector_name(sel) for sel in style_entry.get("selectors") or [])
        parts = []
        if compound:
            state = style_entry.get("state")
            parts.append(f"{compound}:{state}" if state else compound)
        if style_entry.get("selectorsAdd"):
            parts.append(style_entry["selectorsAdd"])
        if not parts or not declarations:
            continue
        rule = f"{', '.join(parts)}{{{declarations}}}"
        at_rule = f"@{at_type} {style_entry.get('mediaText') or ''}".strip() if at_type else None
        if at_rule and blocks and blocks[-1][0] == at_rule:
            blocks[-1][1].append(rule)
        else:
            blocks.append((at_rule, [rule]))

    lines = [
        f"{at_rule}{{{''.join(rules)}}}" if at_rule else rules[0]
        for at_rule, rules in blocks
    ]
    return "\n".join(lines) + "\n" if lines else ""


def resolve_component_styling(component: Dict[str, Any], style_map: Dict[str, Styling]) -> Styling:
    """
    Resolve styling for a component by merging its element-id rule and inline styles.

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

    # Class rules are not merged here: they live in the GUIModel stylesheet,
    # where the browser cascades every class, state and media rule.

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
