"""
GUI Diagram converter module for BUML to JSON conversion.
Reconstructs GrapesJS-compatible JSON structures from BUML GUI models.
"""
from __future__ import annotations
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence
from besser.BUML.metamodel.gui import (
    Button,
    ButtonActionType,
    ButtonType,
    DataList,
    DataSourceElement,
    EmbeddedContent,
    Form,
    GUIModel,
    Image,
    InputField,
    Link,
    Menu,
    MenuItem,
    Module,
    Screen,
    Text,
    ViewComponent,
    ViewContainer,
)
from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import (
    BarChart,
    LineChart,
    MetricCard,
    PieChart,
    RadarChart,
    RadialBarChart,
    TableChart,
)
from besser.BUML.metamodel.gui.events_actions import (
    Create,
    Delete,
    Event,
    Read,
    Transition,
    Update,
)
from besser.BUML.metamodel.gui.style import Alignment, Layout, LayoutType, PositionType, Styling
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def gui_buml_to_json(buml_content: str) -> Dict[str, Any]:
    """
    Convert BUML GUI model Python code to GrapesJS JSON format.
    Args:
        buml_content: GUI model Python code as string
    Returns:
        Dictionary representing GrapesJS project data with pages structure
    """
    if not buml_content or not buml_content.strip():
        return _empty_gui_project()
    gui_model = _parse_gui_model(buml_content)
    if not gui_model:
        return _empty_gui_project()
    try:
        return _serialize_gui_model(gui_model)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[gui_buml_to_json] Failed to serialize GUI model: {exc}")
        return _empty_gui_project()
def extract_gui_section(content: str) -> str:
    """
    Extract the GUI MODEL section from BUML project content.
    Args:
        content: Full project BUML content
    Returns:
        GUI model section content
    """
    pattern = r"# GUI MODEL #(.*?)(?:# (?:STRUCTURAL|OBJECT|AGENT|STATE MACHINE|PROJECT DEFINITION)|$)"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else ""
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
        return gui_buml_to_json(content)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[parse_gui_buml_content] Could not parse GUI BUML content: {exc}")
        return None
# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_gui_model(content: str) -> Optional[GUIModel]:
    """Execute BUML GUI python content and return the first GUIModel found."""
    safe_globals: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "GUIModel": GUIModel,
        "Module": Module,
        "Screen": Screen,
        "ViewComponent": ViewComponent,
        "ViewContainer": ViewContainer,
        "Button": Button,
        "ButtonType": ButtonType,
        "ButtonActionType": ButtonActionType,
        "Text": Text,
        "Image": Image,
        "Link": Link,
        "InputField": InputField,
        "Form": Form,
        "Menu": Menu,
        "MenuItem": MenuItem,
        "DataList": DataList,
        "DataSourceElement": DataSourceElement,
        "EmbeddedContent": EmbeddedContent,
        "LineChart": LineChart,
        "BarChart": BarChart,
        "PieChart": PieChart,
        "RadarChart": RadarChart,
        "RadialBarChart": RadialBarChart,
        "TableChart": TableChart,
        "MetricCard": MetricCard,
        "Transition": Transition,
        "Create": Create,
        "Read": Read,
        "Update": Update,
        "Delete": Delete,
        "Event": Event,
        "DataBinding": DataBinding,
        "Styling": Styling,
        "Layout": Layout,
        "LayoutType": LayoutType,
        "Alignment": Alignment,
        "PositionType": PositionType,
        "domain_model": None,
        "set": set,
        "list": list,
        "tuple": tuple,
        "dict": dict,
    }
    local_vars: Dict[str, Any] = {}
    try:
        exec(content, safe_globals, local_vars)
    except Exception as exc:
        raise ValueError(f"Failed to execute GUI BUML content: {exc}") from exc
    gui_candidates = [
        value for value in local_vars.values() if isinstance(value, GUIModel)
    ]
    if not gui_candidates:
        gui_candidates = [
            value for value in safe_globals.values() if isinstance(value, GUIModel)
        ]
    return gui_candidates[0] if gui_candidates else None
# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _serialize_gui_model(gui_model: GUIModel) -> Dict[str, Any]:
    """Serialize a GUIModel instance into GrapesJS JSON structure."""
    pages: List[Dict[str, Any]] = []
    seen_screen_ids: set[str] = set()
    modules = sorted(gui_model.modules or [], key=lambda m: getattr(m, "name", ""))
    for module in modules:
        for screen in _sorted_screens(module.screens):
            page = _serialize_screen(screen)
            page_id = page.get("id")
            if page_id and page_id in seen_screen_ids:
                continue
            seen_screen_ids.add(page_id)
            pages.append(page)
    if not pages:
        return _empty_gui_project()
    styles = _denormalize_styles(getattr(gui_model, "_style_entries", []))
    return {
        "pages": pages,
        "styles": styles,
        "assets": [],
        "symbols": [],
        "version": "0.21.13",
    }
def _serialize_screen(screen: Screen) -> Dict[str, Any]:
    """Convert a Screen into a GrapesJS page with a wrapper component."""
    screen_id = (
        getattr(screen, "page_id", None)
        or screen.component_id
        or _slugify(screen.name)
        or _generate_id("page")
    )
    wrapper_id = getattr(screen, "component_id", None) or screen_id
    wrapper_component = {
        "type": "wrapper",
        "attributes": {"id": wrapper_id},
    }
    wrapper_style = _styling_to_css(getattr(screen, "styling", None))
    if wrapper_style:
        wrapper_component["style"] = wrapper_style
    children = []
    for child in _sorted_elements(getattr(screen, "view_elements", [])):
        serialized = _serialize_component(child)
        if serialized:
            children.append(serialized)
    if children:
        wrapper_component["components"] = children
    page_attrs = {}
    if getattr(screen, "screen_size", None):
        page_attrs["screen-size"] = screen.screen_size
    if getattr(screen, "x_dpi", None):
        page_attrs["x-dpi"] = screen.x_dpi
    if getattr(screen, "y_dpi", None):
        page_attrs["y-dpi"] = screen.y_dpi
    frame = {
        "id": _generate_id("frame"),
        "component": wrapper_component,
    }
    page = {
        "id": screen_id,
        "name": screen.description or screen.name or screen_id,
        "frames": [frame],
    }
    if page_attrs:
        page["attributes"] = page_attrs
    return page
def _serialize_component(element: ViewComponent) -> Dict[str, Any]:
    """Serialize a BUML view element into a GrapesJS component object."""
    component_type = getattr(element, "component_type", None) or _infer_component_type(
        element
    )
    tag_name = getattr(element, "tag_name", None)
    classes = _serialize_classes(getattr(element, "css_classes", None))
    attributes = _normalize_attributes(dict(getattr(element, "custom_attributes", {}) or {}))
    component_id = getattr(element, "component_id", None) or _generate_id(
        element.__class__.__name__.lower()
    )
    if component_id and "id" not in attributes:
        attributes["id"] = component_id
    _apply_component_specific_attributes(element, attributes)
    node: Dict[str, Any] = {
        "name": getattr(element, "name", None),
    }
    if component_type:
        node["type"] = component_type
    if tag_name:
        node["tagName"] = tag_name
    if classes:
        node["classes"] = classes
    if attributes:
        node["attributes"] = attributes
    style = _styling_to_css(getattr(element, "styling", None))
    if style:
        node["style"] = style
    children: List[Dict[str, Any]] = []
    if isinstance(element, ViewContainer):
        for child in _sorted_elements(getattr(element, "view_elements", [])):
            serialized_child = _serialize_component(child)
            if serialized_child:
                children.append(serialized_child)
    if isinstance(element, Text):
        content = element.content or element.description or element.name or ""
        children = [{"type": "textnode", "content": content}]
    if isinstance(element, Button):
        label = (
            getattr(element, "label", None)
            or attributes.get("button-label")
            or element.name
            or "Button"
        )
        if not children:
            children = [{"type": "textnode", "content": label}]
    elif isinstance(element, Menu) and not children:
        children = _build_menu_components(element)
    if children:
        node["components"] = children
    return _clean_dict(node)
# ---------------------------------------------------------------------------
# Attribute builders
# ---------------------------------------------------------------------------
def _apply_component_specific_attributes(element: ViewComponent, attrs: Dict[str, Any]) -> None:
    """Ensure key component traits are exposed in the attributes dictionary."""
    if isinstance(element, Button):
        _apply_button_attributes(element, attrs)
    elif isinstance(element, DataList):
        _apply_data_list_attributes(element, attrs)
    elif isinstance(element, LineChart):
        _apply_line_chart_attributes(element, attrs)
    elif isinstance(element, BarChart):
        _apply_bar_chart_attributes(element, attrs)
    elif isinstance(element, PieChart):
        _apply_pie_chart_attributes(element, attrs)
    elif isinstance(element, RadarChart):
        _apply_radar_chart_attributes(element, attrs)
    elif isinstance(element, RadialBarChart):
        _apply_radial_bar_chart_attributes(element, attrs)
    elif isinstance(element, TableChart):
        _apply_table_chart_attributes(element, attrs)
    elif isinstance(element, MetricCard):
        _apply_metric_card_attributes(element, attrs)
    elif isinstance(element, InputField):
        _apply_input_field_attributes(element, attrs)
    elif isinstance(element, Form):
        _apply_form_attributes(element, attrs)
    elif isinstance(element, Image):
        _apply_image_attributes(element, attrs)
    elif isinstance(element, Link):
        _apply_link_attributes(element, attrs)
    elif isinstance(element, EmbeddedContent):
        _apply_embedded_content_attributes(element, attrs)
def _apply_chart_data_binding_attributes(component: ViewComponent, attrs: Dict[str, Any]) -> None:
    binding = getattr(component, "data_binding", None)
    if not binding:
        return
    domain = getattr(binding, "domain_concept", None)
    if domain and getattr(domain, "name", None):
        attrs.setdefault("data-source", domain.name)
    label_name = (
        getattr(getattr(binding, "label_field", None), "name", None)
        or getattr(binding, "label_field_name", None)
    )
    data_name = (
        getattr(getattr(binding, "data_field", None), "name", None)
        or getattr(binding, "data_field_name", None)
    )
    if label_name:
        attrs.setdefault("label-field", label_name)
    if data_name:
        attrs.setdefault("data-field", data_name)
def _apply_button_attributes(button: Button, attrs: Dict[str, Any]) -> None:
    attrs.setdefault("button-label", button.label or button.name or "Button")
    action_type = attrs.get("action-type")
    if not action_type:
        action_type = _infer_button_action_type(button)
        if action_type:
            attrs["action-type"] = action_type
    if action_type == "navigate":
        target_screen_attr = attrs.get("target-screen") or attrs.get("data-target-screen")
        if not target_screen_attr:
            target_screen_attr = _resolve_target_screen_id(button)
            if target_screen_attr:
                attrs["target-screen"] = target_screen_attr
    crud_entity = attrs.get("crud-entity") or attrs.get("data-crud-entity")
    if not crud_entity:
        crud_entity = _resolve_crud_target(button)
        if crud_entity:
            attrs["crud-entity"] = crud_entity
def _apply_data_list_attributes(data_list: DataList, attrs: Dict[str, Any]) -> None:
    sources = list(getattr(data_list, "list_sources", []) or [])
    if not sources:
        return
    primary_source = next(iter(sources))
    entity_name = getattr(primary_source.dataSourceClass, "name", None)
    attrs.setdefault("data-source", entity_name or primary_source.name)
    label_field = (
        getattr(getattr(primary_source, "label_field", None), "name", None)
        or getattr(primary_source, "label_field_name", None)
    )
    value_field = (
        getattr(getattr(primary_source, "value_field", None), "name", None)
        or getattr(primary_source, "value_field_name", None)
    )
    if label_field:
        attrs.setdefault("label-field", label_field)
    if value_field:
        attrs.setdefault("data-field", value_field)
def _apply_line_chart_attributes(chart: LineChart, attrs: Dict[str, Any]) -> None:
    _apply_chart_data_binding_attributes(chart, attrs)
    attrs.setdefault("chart-title", chart.title or chart.name)
    if chart.primary_color:
        attrs.setdefault("chart-color", chart.primary_color)
    attrs.setdefault("line-width", chart.line_width)
    attrs.setdefault("show-grid", chart.show_grid)
    attrs.setdefault("show-legend", chart.show_legend)
    attrs.setdefault("show-tooltip", chart.show_tooltip)
    attrs.setdefault("curve-type", chart.curve_type)
    attrs.setdefault("animate", chart.animate)
def _apply_bar_chart_attributes(chart: BarChart, attrs: Dict[str, Any]) -> None:
    _apply_chart_data_binding_attributes(chart, attrs)
    attrs.setdefault("chart-title", chart.title or chart.name)
    if chart.primary_color:
        attrs.setdefault("chart-color", chart.primary_color)
    attrs.setdefault("bar-width", chart.bar_width)
    attrs.setdefault("orientation", chart.orientation)
    attrs.setdefault("show-grid", chart.show_grid)
    attrs.setdefault("show-legend", chart.show_legend)
    attrs.setdefault("stacked", chart.stacked)
def _apply_pie_chart_attributes(chart: PieChart, attrs: Dict[str, Any]) -> None:
    _apply_chart_data_binding_attributes(chart, attrs)
    attrs.setdefault("chart-title", chart.title or chart.name)
    attrs.setdefault("show-legend", chart.show_legend)
    attrs.setdefault("show-labels", chart.show_labels)
    attrs.setdefault("label-position", _enum_value(chart.label_position))
    attrs.setdefault("legend-position", _enum_value(chart.legend_position))
    attrs.setdefault("padding-angle", chart.padding_angle)
def _apply_radar_chart_attributes(chart: RadarChart, attrs: Dict[str, Any]) -> None:
    _apply_chart_data_binding_attributes(chart, attrs)
    attrs.setdefault("chart-title", chart.title or chart.name)
    attrs.setdefault("chart-color", chart.primary_color or "#8884d8")
    attrs.setdefault("show-grid", chart.show_grid)
    attrs.setdefault("show-tooltip", chart.show_tooltip)
    attrs.setdefault("show-radius-axis", chart.show_radius_axis)
def _apply_radial_bar_chart_attributes(chart: RadialBarChart, attrs: Dict[str, Any]) -> None:
    _apply_chart_data_binding_attributes(chart, attrs)
    attrs.setdefault("chart-title", chart.title or chart.name)
    attrs.setdefault("start-angle", chart.start_angle)
    attrs.setdefault("end-angle", chart.end_angle)
def _apply_table_chart_attributes(chart: TableChart, attrs: Dict[str, Any]) -> None:
    _apply_chart_data_binding_attributes(chart, attrs)
    attrs.setdefault("chart-title", chart.title or chart.name)
    attrs.setdefault("chart-color", chart.primary_color or "#2c3e50")
    attrs.setdefault("show-header", chart.show_header)
    attrs.setdefault("striped-rows", chart.striped_rows)
    attrs.setdefault("show-pagination", chart.show_pagination)
    attrs.setdefault("rows-per-page", chart.rows_per_page)
def _apply_metric_card_attributes(card: MetricCard, attrs: Dict[str, Any]) -> None:
    _apply_chart_data_binding_attributes(card, attrs)
    attrs.setdefault("metric-title", card.title or card.name)
    attrs.setdefault("format", getattr(card, "format", None) or "number")
    attrs.setdefault("value-color", getattr(card, "value_color", "#2c3e50"))
    attrs.setdefault("value-size", getattr(card, "value_size", 32))
    attrs.setdefault("show-trend", getattr(card, "show_trend", True))
    attrs.setdefault("positive-color", getattr(card, "positive_color", "#27ae60"))
    attrs.setdefault("negative-color", getattr(card, "negative_color", "#e74c3c"))
def _apply_input_field_attributes(field: InputField, attrs: Dict[str, Any]) -> None:
    attrs.setdefault("placeholder", field.description or "")
    field_type = getattr(field, "field_type", None)
    if field_type:
        attrs.setdefault("type", _enum_value(field_type) or "text")
def _apply_form_attributes(form: Form, attrs: Dict[str, Any]) -> None:
    attrs.setdefault("form-title", form.name or "Form")
    attrs.setdefault("method", getattr(form, "method", "post"))
def _apply_image_attributes(image: Image, attrs: Dict[str, Any]) -> None:
    if getattr(image, "source", None):
        attrs.setdefault("src", image.source)
    if getattr(image, "description", None):
        attrs.setdefault("alt", image.description)
def _apply_link_attributes(link: Link, attrs: Dict[str, Any]) -> None:
    if getattr(link, "url", None):
        attrs.setdefault("href", link.url)
    if getattr(link, "target", None):
        attrs.setdefault("target", link.target)
    if getattr(link, "rel", None):
        attrs.setdefault("rel", link.rel)
def _apply_embedded_content_attributes(content: EmbeddedContent, attrs: Dict[str, Any]) -> None:
    if getattr(content, "source", None):
        attrs.setdefault("src", content.source)
    if getattr(content, "content_type", None):
        attrs.setdefault("content-type", content.content_type)
# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _style_dict_from_layout(layout: Optional[Layout]) -> Dict[str, Any]:
    if not layout:
        return {}
    css: Dict[str, Any] = {}
    layout_type = getattr(layout, "layout_type", None)
    if layout_type == LayoutType.FLEX:
        css["display"] = "flex"
    elif layout_type == LayoutType.GRID:
        css["display"] = "grid"
    elif layout_type == LayoutType.COLUMN:
        css["display"] = "flex"
        css["flex-direction"] = "column"
    elif layout_type == LayoutType.ROW:
        css["display"] = "flex"
        css["flex-direction"] = "row"
    for attr in (
        "flex_direction",
        "justify_content",
        "align_items",
        "flex_wrap",
        "grid_template_columns",
        "grid_template_rows",
        "grid_gap",
        "gap",
    ):
        value = getattr(layout, attr, None)
        if value:
            css[attr.replace("_", "-")] = value
    return css
def _styling_to_css(styling: Optional[Styling]) -> Dict[str, Any]:
    if not styling:
        return {}
    css: Dict[str, Any] = {}
    size = getattr(styling, "size", None)
    if size:
        if getattr(size, "width", None) not in (None, "", "auto"):
            css["width"] = size.width
        if getattr(size, "height", None) not in (None, "", "auto"):
            css["height"] = size.height
        if getattr(size, "padding", None):
            css["padding"] = size.padding
        if getattr(size, "margin", None):
            css["margin"] = size.margin
        if getattr(size, "font_size", None):
            css["font-size"] = size.font_size
        if getattr(size, "line_height", None):
            css["line-height"] = size.line_height
    position = getattr(styling, "position", None)
    if position:
        p_type = getattr(position, "p_type", None)
        if p_type and isinstance(p_type, PositionType):
            css["position"] = p_type.value
        for attr in ("top", "left", "right", "bottom"):
            value = getattr(position, attr, None)
            if value not in (None, "", "auto"):
                css[attr] = value
        alignment = getattr(position, "alignment", None)
        if alignment:
            css["text-align"] = _enum_value(alignment) or alignment
        if getattr(position, "z_index", None) not in (None, 0):
            css["z-index"] = position.z_index
    color = getattr(styling, "color", None)
    if color:
        if getattr(color, "background_color", None):
            css["background-color"] = str(color.background_color)
        if getattr(color, "text_color", None):
            css["color"] = str(color.text_color)
        if getattr(color, "border_color", None):
            css["border-color"] = str(color.border_color)
        if getattr(color, "opacity", None):
            css["opacity"] = color.opacity
        if getattr(color, "box_shadow", None):
            css["box-shadow"] = color.box_shadow
    layout = getattr(styling, "layout", None)
    css.update(_style_dict_from_layout(layout))
    return {k: v for k, v in css.items() if v not in (None, "", "auto")}
def _denormalize_styles(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert normalized style entries back into GrapesJS selector objects."""
    denormalized: List[Dict[str, Any]] = []
    for entry in entries or []:
        selectors: List[Any] = []
        for selector in entry.get("selectors", []):
            if isinstance(selector, str) and selector.startswith("."):
                selectors.append({"name": selector[1:], "private": 1})
            else:
                selectors.append(selector)
        style = {
            _to_kebab_case(k): v for k, v in (entry.get("style") or {}).items()
        }
        normalized_entry: Dict[str, Any] = {"selectors": selectors, "style": style}
        if entry.get("selectorsAdd"):
            normalized_entry["selectorsAdd"] = entry["selectorsAdd"]
        denormalized.append(normalized_entry)
    return denormalized
def _sorted_elements(elements: Iterable[ViewComponent]) -> List[ViewComponent]:
    return sorted(
        elements or [],
        key=lambda elem: (
            getattr(elem, "display_order", 0),
            getattr(elem, "name", ""),
        ),
    )
def _sorted_screens(screens: Iterable[Screen]) -> List[Screen]:
    return sorted(
        screens or [],
        key=lambda scr: getattr(scr, "name", ""),
    )
def _serialize_classes(classes: Optional[Iterable[str]]) -> Optional[List[Dict[str, Any]]]:
    if not classes:
        return None
    normalized = []
    for cls in classes:
        if cls:
            normalized.append({"name": cls})
    return normalized or None
def _normalize_attributes(attrs: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in (attrs or {}).items():
        normalized[key] = _normalize_value(value)
    return normalized
def _normalize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    if isinstance(value, set):
        return [_normalize_value(v) for v in value]
    if hasattr(value, "name"):
        return value.name
    if hasattr(value, "value"):
        return value.value
    return str(value)
def _clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v not in (None, "", [], {})}
def _slugify(name: Optional[str]) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip())
    return cleaned.strip("-").lower()
def _generate_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"
def _enum_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "name"):
        return value.name
    return str(value)
def _to_kebab_case(value: str) -> str:
    if not value:
        return value
    return re.sub(r"(?<!^)(?=[A-Z])", "-", value).replace("_", "-").lower()
def _infer_component_type(element: ViewComponent) -> Optional[str]:
    mapping = {
        Text: "text",
        Image: "image",
        Link: "link",
        Button: "action-button",
        InputField: "input",
        Form: "form",
        Menu: "menu",
        DataList: "data-list",
        EmbeddedContent: "embedded-content",
        LineChart: "line-chart",
        BarChart: "bar-chart",
        PieChart: "pie-chart",
        RadarChart: "radar-chart",
        RadialBarChart: "radial-bar-chart",
        TableChart: "table-chart",
        MetricCard: "metric-card",
    }
    for cls, comp_type in mapping.items():
        if isinstance(element, cls):
            return comp_type
    if isinstance(element, ViewContainer):
        return "div"
    return None
def _infer_button_action_type(button: Button) -> Optional[str]:
    events = getattr(button, "events", None)
    if not events:
        return None
    for event in events:
        for action in getattr(event, "actions", []):
            if isinstance(action, Transition):
                return "navigate"
            if isinstance(action, Create):
                return "create"
            if isinstance(action, Read):
                return "read"
            if isinstance(action, Update):
                return "update"
            if isinstance(action, Delete):
                return "delete"
    return None
def _resolve_target_screen_id(button: Button) -> Optional[str]:
    events = getattr(button, "events", None)
    if not events:
        return None
    for event in events:
        for action in getattr(event, "actions", []):
            if isinstance(action, Transition):
                target = getattr(action, "target_screen", None)
                if target:
                    return (
                        getattr(target, "page_id", None)
                        or getattr(target, "component_id", None)
                        or getattr(target, "name", None)
                    )
    return None
def _resolve_crud_target(button: Button) -> Optional[str]:
    events = getattr(button, "events", None)
    if not events:
        return None
    for event in events:
        for action in getattr(event, "actions", []):
            if isinstance(action, (Create, Read, Update, Delete)):
                target = getattr(action, "target_class", None)
                if target and getattr(target, "name", None):
                    return target.name
    return None
def _empty_gui_project() -> Dict[str, Any]:
    """Return a minimal valid GrapesJS project payload."""
    return {
        "pages": [
            {
                "id": _generate_id("page"),
                "name": "Home",
                "frames": [
                    {
                        "id": _generate_id("frame"),
                        "component": {
                            "type": "wrapper",
                            "attributes": {"id": _generate_id("wrapper")},
                            "components": [],
                        },
                    }
                ],
            }
        ],
        "styles": [],
        "assets": [],
        "symbols": [],
        "version": "0.21.13",
    }
def _build_menu_components(menu: Menu) -> List[Dict[str, Any]]:
    items = sorted(
        getattr(menu, "menuItems", []) or [],
        key=lambda item: getattr(item, "label", ""),
    )
    components: List[Dict[str, Any]] = []
    for item in items:
        label = getattr(item, "label", None) or "Menu Item"
        link_attrs = {
            "href": getattr(item, "url", None) or "#",
            "id": _generate_id("menu-item"),
        }
        if getattr(item, "target", None):
            link_attrs["target"] = item.target
        if getattr(item, "rel", None):
            link_attrs["rel"] = item.rel
        components.append(
            {
                "type": "link",
                "attributes": link_attrs,
                "components": [{"type": "textnode", "content": label}],
            }
        )
    return components