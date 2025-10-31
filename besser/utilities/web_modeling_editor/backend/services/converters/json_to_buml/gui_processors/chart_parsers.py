"""
Chart component parsers for GUI diagrams.
"""

from typing import Dict, Any
from besser.BUML.metamodel.gui import (
    Alignment, BarChart, DataBinding, LineChart,
    PieChart, RadarChart, RadialBarChart,
    Color, Position, Size, Styling,
)
from .styling import ensure_styling_parts
from .utils import clean_attribute_name, get_element_by_id, parse_bool


def parse_line_chart(view_comp: Dict[str, Any], class_model, domain_model) -> LineChart:
    """
    Parses a line chart component, resolving data binding from class_model and domain_model.
    
    Args:
        view_comp: Component dict from GrapesJS
        class_model: Class model for object resolution
        domain_model: Domain metamodel for object resolution
        
    Returns:
        LineChart instance
    """
    attrs = view_comp.get('attributes', {})

    # Resolve data binding elements
    data_source_el = get_element_by_id(class_model, attrs.get('data-source'))
    label_field_el = get_element_by_id(class_model, attrs.get('label-field'))
    data_field_el = get_element_by_id(class_model, attrs.get('data-field'))

    data_source_name = data_source_el.get('name') if data_source_el else None
    label_field_name = label_field_el.get('name') if label_field_el else None
    data_field_name = data_field_el.get('name') if data_field_el else None

    if label_field_name:
        label_field_name = clean_attribute_name(label_field_name)
    if data_field_name:
        data_field_name = clean_attribute_name(data_field_name)

    # Resolve domain class and fields
    domain_class = domain_model.get_class_by_name(data_source_name) if data_source_name else None
    label_field = None
    data_field = None

    if domain_class:
        if label_field_name:
            label_field = next((a for a in domain_class.attributes if a.name == label_field_name), None)
        if data_field_name:
            data_field = next((a for a in domain_class.attributes if a.name == data_field_name), None)

    raw_title = attrs.get('chart-title')
    title_value = raw_title.strip() if isinstance(raw_title, str) else None
    name_seed = raw_title if isinstance(raw_title, str) else 'LineChart'
    chart_title = name_seed.replace(' ', '_') if isinstance(name_seed, str) else name_seed
    if not title_value:
        if isinstance(chart_title, str):
            title_value = chart_title.replace('_', ' ')
        else:
            title_value = str(chart_title)

    primary_color = attrs.get('chart-color')
    if not isinstance(primary_color, str) or not primary_color.strip():
        primary_color = None

    # Create data binding
    data_binding = DataBinding(
        name=(title_value or "LineChart") + "DataBinding",
        domain_concept=domain_class,
        label_field=label_field,
        data_field=data_field
    )

    chart_title_attr = attrs.get('chart-title', 'LineChart')
    chart_title = chart_title_attr.replace(' ', '_') if isinstance(chart_title_attr, str) else chart_title_attr
    display_title = chart_title_attr if isinstance(chart_title_attr, str) else None
    primary_color = attrs.get('chart-color') or attrs.get('color')

    # Parse enhanced line chart properties
    line_chart = LineChart(
        name=chart_title,
        title=display_title,
        primary_color=primary_color,
        line_width=int(attrs.get('line-width', 2)),
        show_grid=parse_bool(attrs.get('show-grid'), True),
        show_legend=parse_bool(attrs.get('show-legend'), True),
        show_tooltip=parse_bool(attrs.get('show-tooltip'), True),
        curve_type=attrs.get('curve-type', 'monotone'),
        animate=parse_bool(attrs.get('animate'), True),
        legend_position=attrs.get('legend-position', 'top'),
        grid_color=attrs.get('grid-color', '#e0e0e0'),
        dot_size=int(attrs.get('dot-size', 5)),
        title=title_value,
        primary_color=primary_color
    )
    line_chart.data_binding = data_binding
    return line_chart


def parse_bar_chart(view_comp: Dict[str, Any], class_model, domain_model) -> BarChart:
    """
    Parses a bar chart component, resolving data binding from class_model and domain_model.
    
    Args:
        view_comp: Component dict from GrapesJS
        class_model: Class model for object resolution
        domain_model: Domain metamodel for object resolution
        
    Returns:
        BarChart instance
    """
    attrs = view_comp.get('attributes', {})

    # Resolve data binding elements
    data_source_el = get_element_by_id(class_model, attrs.get('data-source'))
    label_field_el = get_element_by_id(class_model, attrs.get('label-field'))
    data_field_el = get_element_by_id(class_model, attrs.get('data-field'))

    data_source_name = data_source_el.get('name') if data_source_el else None
    label_field_name = label_field_el.get('name') if label_field_el else None
    data_field_name = data_field_el.get('name') if data_field_el else None

    if label_field_name:
        label_field_name = clean_attribute_name(label_field_name)
    if data_field_name:
        data_field_name = clean_attribute_name(data_field_name)

    # Resolve domain class and fields
    domain_class = domain_model.get_class_by_name(data_source_name) if data_source_name else None
    label_field = None
    data_field = None

    if domain_class:
        if label_field_name:
            label_field = next((a for a in domain_class.attributes if a.name == label_field_name), None)
        if data_field_name:
            data_field = next((a for a in domain_class.attributes if a.name == data_field_name), None)

    # Create data binding
    data_binding = DataBinding(
        name=attrs.get('chart-title', 'BarChart') + "DataBinding",
        domain_concept=domain_class,
        label_field=label_field,
        data_field=data_field
    )

    chart_title_attr = attrs.get('chart-title', 'BarChart')
    chart_title = chart_title_attr.replace(' ', '_') if isinstance(chart_title_attr, str) else chart_title_attr
    display_title = chart_title_attr if isinstance(chart_title_attr, str) else None
    primary_color = attrs.get('chart-color') or attrs.get('color')

    # Parse enhanced bar chart properties
    bar_chart = BarChart(
        name=chart_title,
        title=display_title,
        primary_color=primary_color,
        bar_width=int(attrs.get('bar-width', 30)),
        orientation=attrs.get('orientation', 'vertical'),
        show_grid=parse_bool(attrs.get('show-grid'), True),
        show_legend=parse_bool(attrs.get('show-legend'), True),
        show_tooltip=parse_bool(attrs.get('show-tooltip'), True),
        stacked=parse_bool(attrs.get('stacked'), False),
        animate=parse_bool(attrs.get('animate'), True),
        legend_position=attrs.get('legend-position', 'top'),
        grid_color=attrs.get('grid-color', '#e0e0e0'),
        bar_gap=int(attrs.get('bar-gap', 4))
    )
    bar_chart.data_binding = data_binding
    return bar_chart


def parse_pie_chart(view_comp: Dict[str, Any], class_model, domain_model) -> PieChart:
    """
    Parses a pie chart component, resolving data binding from class_model and domain_model.
    
    Args:
        view_comp: Component dict from GrapesJS
        class_model: Class model for object resolution
        domain_model: Domain metamodel for object resolution
        
    Returns:
        PieChart instance
    """
    attrs = view_comp.get('attributes', {})

    # Resolve data binding elements
    data_source_el = get_element_by_id(class_model, attrs.get('data-source'))
    label_field_el = get_element_by_id(class_model, attrs.get('label-field'))
    data_field_el = get_element_by_id(class_model, attrs.get('data-field'))

    data_source_name = data_source_el.get('name') if data_source_el else None
    label_field_name = label_field_el.get('name') if label_field_el else None
    data_field_name = data_field_el.get('name') if data_field_el else None

    if label_field_name:
        label_field_name = clean_attribute_name(label_field_name)
    if data_field_name:
        data_field_name = clean_attribute_name(data_field_name)

    # Resolve domain class and fields
    domain_class = domain_model.get_class_by_name(data_source_name) if data_source_name else None
    label_field = None
    data_field = None

    if domain_class:
        if label_field_name:
            label_field = next((a for a in domain_class.attributes if a.name == label_field_name), None)
        if data_field_name:
            data_field = next((a for a in domain_class.attributes if a.name == data_field_name), None)

    # Create data binding
    data_binding = DataBinding(
        name=attrs.get('chart-title', 'PieChart') + "DataBinding",
        domain_concept=domain_class,
        label_field=label_field,
        data_field=data_field
    )

    chart_title_attr = attrs.get('chart-title', 'PieChart')
    chart_title = chart_title_attr.replace(' ', '_') if isinstance(chart_title_attr, str) else chart_title_attr
    display_title = chart_title_attr if isinstance(chart_title_attr, str) else None
    primary_color = attrs.get('chart-color') or attrs.get('color')

    # Parse enhanced pie chart properties
    show_legend = parse_bool(attrs.get('show-legend'), True)
    show_labels = parse_bool(attrs.get('show-labels'), True)

    # Map string positions to Alignment enum
    legend_pos_map = {
        'top': Alignment.TOP,
        'right': Alignment.RIGHT,
        'bottom': Alignment.BOTTOM,
        'left': Alignment.LEFT
    }
    label_pos_map = {
        'inside': Alignment.INSIDE,
        'outside': Alignment.OUTSIDE
    }

    legend_position_str = attrs.get('legend-position', 'left')
    label_position_str = attrs.get('label-position', 'inside')

    legend_position = legend_pos_map.get(legend_position_str, Alignment.LEFT)
    label_position = label_pos_map.get(label_position_str, Alignment.INSIDE)

    pie_chart = PieChart(
        name=chart_title,
        title=display_title,
        primary_color=primary_color,
        show_legend=show_legend,
        legend_position=legend_position,
        show_labels=show_labels,
        label_position=label_position,
        padding_angle=int(attrs.get('padding-angle', 0)),
        inner_radius=int(attrs.get('inner-radius', 0)),
        outer_radius=int(attrs.get('outer-radius', 80)),
        start_angle=int(attrs.get('start-angle', 0)),
        end_angle=int(attrs.get('end-angle', 360))
    )
    pie_chart.data_binding = data_binding
    return pie_chart


def parse_radar_chart(view_comp: Dict[str, Any], _, domain_model) -> RadarChart:
    """
    Parses a radar chart component, resolving data binding from domain_model.
    
    Args:
        view_comp: Component dict from GrapesJS
        domain_model: Domain metamodel for object resolution
        
    Returns:
        RadarChart instance
    """
    attrs = view_comp.get('attributes', {})

    data_source_name = attrs.get('data-source')
    label_field_name = attrs.get('label-field')
    data_field_name = attrs.get('data-field')

    # Resolve domain class and fields
    domain_class = domain_model.get_class_by_name(data_source_name) if data_source_name else None
    label_field = None
    data_field = None

    if domain_class:
        if label_field_name:
            label_field = next((a for a in domain_class.attributes if a.name == label_field_name), None)
        if data_field_name:
            data_field = next((a for a in domain_class.attributes if a.name == data_field_name), None)

    # Create data binding
    data_binding = DataBinding(
        name=attrs.get('chart-title', 'RadarChart') + "DataBinding",
        domain_concept=domain_class,
        label_field=label_field,
        data_field=data_field
    )

    chart_title_attr = attrs.get('chart-title', 'RadarChart')
    chart_title = chart_title_attr.replace(' ', '_') if isinstance(chart_title_attr, str) else chart_title_attr
    display_title = chart_title_attr if isinstance(chart_title_attr, str) else None
    primary_color = attrs.get('chart-color') or attrs.get('color')

    # Parse enhanced radar chart properties
    radar_chart = RadarChart(
        name=chart_title,
        title=display_title,
        primary_color=primary_color,
        show_grid=parse_bool(attrs.get('show-grid'), True),
        show_tooltip=parse_bool(attrs.get('show-tooltip'), True),
        show_radius_axis=parse_bool(attrs.get('show-radius-axis'), True),
        show_legend=parse_bool(attrs.get('show-legend'), True),
        legend_position=attrs.get('legend-position', 'top'),
        dot_size=int(attrs.get('dot-size', 3)),
        grid_type=attrs.get('grid-type', 'polygon'),
        stroke_width=int(attrs.get('stroke-width', 2))
    )
    radar_chart.data_binding = data_binding
    return radar_chart


def parse_radial_bar_chart(view_comp: Dict[str, Any], class_model, domain_model) -> RadialBarChart:
    """
    Parses a radial bar chart component, mapping features/values to label/data fields.
    
    Args:
        view_comp: Component dict from GrapesJS
        class_model: Class model for object resolution
        domain_model: Domain metamodel for object resolution
        
    Returns:
        RadialBarChart instance
    """
    attrs = view_comp.get('attributes', {})

    # Resolve data binding elements (features = labels, values = data)
    data_source_el = get_element_by_id(class_model, attrs.get('data-source'))
    features_el = get_element_by_id(class_model, attrs.get('features'))
    values_el = get_element_by_id(class_model, attrs.get('values'))

    data_source_name = data_source_el.get('name') if data_source_el else None
    features_name = features_el.get('name') if features_el else None
    values_name = values_el.get('name') if values_el else None

    if features_name:
        features_name = clean_attribute_name(features_name)
    if values_name:
        values_name = clean_attribute_name(values_name)

    # Resolve domain class and fields
    domain_class = domain_model.get_class_by_name(data_source_name) if data_source_name else None
    label_field = None
    data_field = None

    if domain_class:
        if features_name:
            label_field = next((a for a in domain_class.attributes if a.name == features_name), None)
        if values_name:
            data_field = next((a for a in domain_class.attributes if a.name == values_name), None)

    # Create data binding
    data_binding = DataBinding(
        name=attrs.get('chart-title', 'RadialBarChart') + "DataBinding",
        domain_concept=domain_class,
        label_field=label_field,
        data_field=data_field
    )

    chart_title_attr = attrs.get('chart-title', 'RadialBarChart')
    chart_title = chart_title_attr.replace(' ', '_') if isinstance(chart_title_attr, str) else chart_title_attr
    display_title = chart_title_attr if isinstance(chart_title_attr, str) else None
    primary_color = attrs.get('chart-color') or attrs.get('color')

    # Parse enhanced radial bar chart properties
    radial_bar_chart = RadialBarChart(
        name=chart_title,
        title=display_title,
        primary_color=primary_color,
        start_angle=int(attrs.get('start-angle', 0)),
        end_angle=int(attrs.get('end-angle', 360)),
        inner_radius=int(attrs.get('inner-radius', 30)),
        outer_radius=int(attrs.get('outer-radius', 80)),
        show_legend=parse_bool(attrs.get('show-legend'), True),
        legend_position=attrs.get('legend-position', 'top'),
        show_tooltip=parse_bool(attrs.get('show-tooltip'), True)
    )
    radial_bar_chart.data_binding = data_binding
    return radial_bar_chart


def apply_chart_colors(element, attributes: Dict[str, Any]) -> None:
    """
    Apply chart-specific color styling based on chart type.
    
    Args:
        element: Chart ViewComponent instance
        attributes: Component attributes dict
    """

    color_value = None
    if isinstance(attributes, dict):
        color_value = attributes.get("chart-color") or attributes.get("color")
    if not color_value:
        return

    element.styling = ensure_styling_parts(
        element.styling or Styling(size=Size(), position=Position(), color=Color())
    )

    if isinstance(element, LineChart):
        element.styling.color.line_color = color_value
    elif isinstance(element, BarChart):
        element.styling.color.bar_color = color_value
    elif isinstance(element, PieChart):
        element.styling.color.color_palette = color_value
    elif isinstance(element, RadarChart):
        element.styling.color.line_color = color_value
    elif isinstance(element, RadialBarChart):
        element.styling.color.bar_color = color_value
