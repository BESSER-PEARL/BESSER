"""
Chart component parsers for GUI diagrams.
"""

from typing import Dict, Any
from besser.BUML.metamodel.gui import (
    Alignment, BarChart, DataBinding, LineChart,
    PieChart, RadarChart, RadialBarChart, TableChart, ViewComponent,
    Color, Position, Size, Styling, DataAggregation, MetricCard,
)
from .styling import ensure_styling_parts
from .utils import clean_attribute_name, get_element_by_id, parse_bool, sanitize_name

# TODO: Uncomment when backend aggregation is ready
# def _parse_aggregation(aggregation_str: str) -> DataAggregation:
#     """
#     Parse aggregation string from JSON attributes to DataAggregation enum.
    
#     Args:
#         aggregation_str: String value like "sum", "average", "count", etc.
        
#     Returns:
#         DataAggregation enum value or None if invalid
#     """
#     if not aggregation_str or not isinstance(aggregation_str, str):
#         return None
    
#     aggregation_map = {
#         'sum': DataAggregation.SUM,
#         'avg': DataAggregation.AVG,
#         'average': DataAggregation.AVG,
#         'count': DataAggregation.COUNT,
#         'min': DataAggregation.MIN,
#         'minimum': DataAggregation.MIN,
#         'max': DataAggregation.MAX,
#         'maximum': DataAggregation.MAX,
#         'median': DataAggregation.MEDIAN,
#         'first': DataAggregation.FIRST,
#         'last': DataAggregation.LAST,
#     }
    
#     return aggregation_map.get(aggregation_str.lower())


def _parse_chart_data_binding(attrs: Dict[str, Any], class_model, domain_model) -> tuple:
    """
    Parse common chart data binding attributes.
    
    Returns:
        Tuple of (domain_class, label_field, data_field, aggregation, group_by_field)
    """
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

    return domain_class, label_field, data_field


def _attach_chart_metadata(chart, component: Dict[str, Any]) -> None:
    """Helper to attach GrapesJS metadata to chart component for code generation fidelity."""
    if not chart:
        return
    
    attributes = component.get("attributes", {})
    if isinstance(attributes, dict):
        chart.component_id = attributes.get("id") or component.get("id")
    else:
        chart.component_id = component.get("id")
    
    chart.component_type = component.get("type")
    chart.tag_name = component.get("tagName")
    chart.css_classes = [cls if isinstance(cls, str) else cls.get("name", "") for cls in (component.get("classes") or [])]
    chart.custom_attributes = dict(attributes) if isinstance(attributes, dict) else {}


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

    # Parse common data binding fields
    domain_class, label_field, data_field = _parse_chart_data_binding(
        attrs, class_model, domain_model
    )

    raw_title = attrs.get('chart-title')
    title_value = raw_title.strip() if isinstance(raw_title, str) else None
    name_seed = raw_title if isinstance(raw_title, str) else 'LineChart'
    # Use sanitize_name for proper name handling (removes special chars, handles typos)
    chart_title = sanitize_name(name_seed) if name_seed else sanitize_name('LineChart')
    if not chart_title:
        chart_title = 'LineChart'
    if not title_value:
        title_value = chart_title.replace('_', ' ').title()

    primary_color = attrs.get('chart-color')
    if not isinstance(primary_color, str) or not primary_color.strip():
        primary_color = None

    # Create data binding (only if domain_class exists)
    data_binding = None
    if domain_class:
        data_binding = DataBinding(
            domain_concept=domain_class,
            label_field=label_field,
            data_field=data_field
        )

    # Parse enhanced line chart properties
    line_chart = LineChart(
        name=chart_title,
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
    _attach_chart_metadata(line_chart, view_comp)
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

    # Parse common data binding fields
    domain_class, label_field, data_field = _parse_chart_data_binding(
        attrs, class_model, domain_model
    )

    raw_title = attrs.get('chart-title')
    title_value = raw_title.strip() if isinstance(raw_title, str) else None
    name_seed = raw_title if isinstance(raw_title, str) else 'BarChart'
    # Use sanitize_name for proper name handling (removes special chars, handles typos)
    chart_title = sanitize_name(name_seed) if name_seed else sanitize_name('BarChart')
    if not chart_title:
        chart_title = 'BarChart'
    if not title_value:
        title_value = chart_title.replace('_', ' ').title()

    primary_color = attrs.get('chart-color')
    if not isinstance(primary_color, str) or not primary_color.strip():
        primary_color = None

    # Create data binding (only if domain_class exists)
    data_binding = None
    if domain_class:
        data_binding = DataBinding(
            domain_concept=domain_class,
            label_field=label_field,
            data_field=data_field
        )

    # Parse enhanced bar chart properties
    bar_chart = BarChart(
        name=chart_title,
        bar_width=int(attrs.get('bar-width', 30)),
        orientation=attrs.get('orientation', 'vertical'),
        show_grid=parse_bool(attrs.get('show-grid'), True),
        show_legend=parse_bool(attrs.get('show-legend'), True),
        show_tooltip=parse_bool(attrs.get('show-tooltip'), True),
        stacked=parse_bool(attrs.get('stacked'), False),
        animate=parse_bool(attrs.get('animate'), True),
        legend_position=attrs.get('legend-position', 'top'),
        grid_color=attrs.get('grid-color', '#e0e0e0'),
        bar_gap=int(attrs.get('bar-gap', 4)),
        title=title_value,
        primary_color=primary_color
    )
    bar_chart.data_binding = data_binding
    _attach_chart_metadata(bar_chart, view_comp)
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

    # Parse common data binding fields
    domain_class, label_field, data_field = _parse_chart_data_binding(
        attrs, class_model, domain_model
    )

    raw_title = attrs.get('chart-title')
    title_value = raw_title.strip() if isinstance(raw_title, str) else None
    name_seed = raw_title if isinstance(raw_title, str) else 'PieChart'
    # Use sanitize_name for proper name handling (removes special chars, handles typos)
    chart_title = sanitize_name(name_seed) if name_seed else sanitize_name('PieChart')
    if not chart_title:
        chart_title = 'PieChart'
    if not title_value:
        title_value = chart_title.replace('_', ' ').title()

    primary_color = attrs.get('chart-color')
    if not isinstance(primary_color, str) or not primary_color.strip():
        primary_color = None

    # Create data binding (only if domain_class exists)
    data_binding = None
    if domain_class:
        data_binding = DataBinding(
            domain_concept=domain_class,
            label_field=label_field,
            data_field=data_field
        )

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
        show_legend=show_legend,
        legend_position=legend_position,
        show_labels=show_labels,
        label_position=label_position,
        padding_angle=int(attrs.get('padding-angle', 0)),
        inner_radius=int(attrs.get('inner-radius', 0)),
        outer_radius=int(attrs.get('outer-radius', 80)),
        start_angle=int(attrs.get('start-angle', 0)),
        end_angle=int(attrs.get('end-angle', 360)),
        title=title_value,
        primary_color=primary_color
    )
    pie_chart.data_binding = data_binding
    _attach_chart_metadata(pie_chart, view_comp)
    return pie_chart


def parse_radar_chart(view_comp: Dict[str, Any], class_model, domain_model) -> RadarChart:
    """
    Parses a radar chart component, resolving data binding from class_model and domain_model.
    
    Args:
        view_comp: Component dict from GrapesJS
        class_model: Class model for object resolution
        domain_model: Domain metamodel for object resolution
        
    Returns:
        RadarChart instance
    """
    attrs = view_comp.get('attributes', {})

    # Parse common data binding fields
    domain_class, label_field, data_field = _parse_chart_data_binding(
        attrs, class_model, domain_model
    )

    raw_title = attrs.get('chart-title')
    title_value = raw_title.strip() if isinstance(raw_title, str) else None
    name_seed = raw_title if isinstance(raw_title, str) else 'RadarChart'
    # Use sanitize_name for proper name handling (removes special chars, handles typos)
    chart_title = sanitize_name(name_seed) if name_seed else sanitize_name('RadarChart')
    if not chart_title:
        chart_title = 'RadarChart'
    if not title_value:
        title_value = chart_title.replace('_', ' ').title()

    primary_color = attrs.get('chart-color')
    if not isinstance(primary_color, str) or not primary_color.strip():
        primary_color = None

    # Create data binding (only if domain_class exists)
    data_binding = None
    if domain_class:
        data_binding = DataBinding(
            domain_concept=domain_class,
            label_field=label_field,
            data_field=data_field
        )

    # Parse enhanced radar chart properties
    radar_chart = RadarChart(
        name=chart_title,
        show_grid=parse_bool(attrs.get('show-grid'), True),
        show_tooltip=parse_bool(attrs.get('show-tooltip'), True),
        show_radius_axis=parse_bool(attrs.get('show-radius-axis'), True),
        show_legend=parse_bool(attrs.get('show-legend'), True),
        legend_position=attrs.get('legend-position', 'top'),
        dot_size=int(attrs.get('dot-size', 3)),
        grid_type=attrs.get('grid-type', 'polygon'),
        stroke_width=int(attrs.get('stroke-width', 2)),
        title=title_value,
        primary_color=primary_color
    )
    radar_chart.data_binding = data_binding
    _attach_chart_metadata(radar_chart, view_comp)
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

    # For radial bar, parse using special 'features' and 'values' attrs
    # but also check for standard fields with aggregation support
    data_source_el = get_element_by_id(class_model, attrs.get('data-source'))
    features_el = get_element_by_id(class_model, attrs.get('features')) or get_element_by_id(class_model, attrs.get('label-field'))
    values_el = get_element_by_id(class_model, attrs.get('values')) or get_element_by_id(class_model, attrs.get('data-field'))

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

    raw_title = attrs.get('chart-title')
    title_value = raw_title.strip() if isinstance(raw_title, str) else None
    name_seed = raw_title if isinstance(raw_title, str) else 'RadialBarChart'
    # Use sanitize_name for proper name handling (removes special chars, handles typos)
    chart_title = sanitize_name(name_seed) if name_seed else sanitize_name('RadialBarChart')
    if not chart_title:
        chart_title = 'RadialBarChart'
    if not title_value:
        title_value = chart_title.replace('_', ' ').title()

    primary_color = attrs.get('chart-color')
    if not isinstance(primary_color, str) or not primary_color.strip():
        primary_color = None

    # Create data binding (only if domain_class exists)
    data_binding = None
    if domain_class:
        data_binding = DataBinding(
            domain_concept=domain_class,
            label_field=label_field,
            data_field=data_field
        )

    # Parse enhanced radial bar chart properties
    radial_bar_chart = RadialBarChart(
        name=chart_title,
        start_angle=int(attrs.get('start-angle', 0)),
        end_angle=int(attrs.get('end-angle', 360)),
        inner_radius=int(attrs.get('inner-radius', 30)),
        outer_radius=int(attrs.get('outer-radius', 80)),
        show_legend=parse_bool(attrs.get('show-legend'), True),
        legend_position=attrs.get('legend-position', 'top'),
        show_tooltip=parse_bool(attrs.get('show-tooltip'), True),
        title=title_value,
        primary_color=primary_color
    )
    radial_bar_chart.data_binding = data_binding
    _attach_chart_metadata(radial_bar_chart, view_comp)
    return radial_bar_chart


def parse_table_chart(view_comp: Dict[str, Any], class_model, domain_model) -> TableChart:
    """
    Parses a table chart component, resolving data binding and presentation options.
    """
    attrs = view_comp.get('attributes', {})

    # Parse common data binding fields
    domain_class, label_field, data_field = _parse_chart_data_binding(
        attrs, class_model, domain_model
    )

    raw_title = attrs.get('chart-title')
    title_value = raw_title.strip() if isinstance(raw_title, str) else None
    name_seed = raw_title if isinstance(raw_title, str) else 'TableChart'
    chart_title = sanitize_name(name_seed) if name_seed else sanitize_name('TableChart')
    if not chart_title:
        chart_title = 'TableChart'
    if not title_value:
        title_value = chart_title.replace('_', ' ').title()

    primary_color = attrs.get('chart-color')
    if not isinstance(primary_color, str) or not primary_color.strip():
        primary_color = None

    data_binding = None
    if domain_class:
        data_binding = DataBinding(
            domain_concept=domain_class,
            label_field=label_field,
            data_field=data_field
        )

    def _bool_attr(key: str, default: bool) -> bool:
        return parse_bool(attrs.get(key), default)

    def _int_attr(key: str, default: int) -> int:
        try:
            return int(attrs.get(key, default))
        except (TypeError, ValueError):
            return default

    table_chart = TableChart(
        name=chart_title,
        show_header=_bool_attr('show-header', True),
        striped_rows=_bool_attr('striped-rows', False),
        show_pagination=_bool_attr('show-pagination', True),
        rows_per_page=_int_attr('rows-per-page', 5),
        title=title_value,
        primary_color=primary_color
    )

    if domain_class:
        column_names = []
        for attr in getattr(domain_class, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if isinstance(attr_name, str) and attr_name:
                column_names.append(attr_name)
        # Attempt to include association end roles if available
        association_iterables = []
        for candidate in ("association_ends", "associationEnds"):
            attr_value = getattr(domain_class, candidate, None)
            if not attr_value:
                continue
            try:
                association_data = attr_value() if callable(attr_value) else attr_value
            except TypeError:
                association_data = attr_value

            if association_data is None:
                continue
            if isinstance(association_data, dict):
                association_iterables.extend(association_data.values())
            elif isinstance(association_data, (list, tuple, set)):
                association_iterables.extend(list(association_data))
            else:
                association_iterables.append(association_data)

        for association_end in association_iterables:
            if association_end is None:
                continue
            end_name = None
            if hasattr(association_end, "name"):
                end_name = getattr(association_end, "name", None)
            if not end_name and hasattr(association_end, "role"):
                end_name = getattr(association_end, "role", None)
            if isinstance(end_name, str) and end_name:
                column_names.append(end_name)

        if column_names:
            table_chart.columns = column_names

    table_chart.data_binding = data_binding
    _attach_chart_metadata(table_chart, view_comp)
    return table_chart


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
    elif isinstance(element, TableChart):
        element.styling.color.background_color = color_value


def parse_metric_card(view_comp: Dict[str, Any], class_model, domain_model) -> MetricCard:
    """
    Parse a metric card component from JSON to BUML MetricCard.
    
    Args:
        view_comp: Component dictionary from GrapesJS JSON
        class_model: Class diagram model
        domain_model: Domain model containing classes
        
    Returns:
        MetricCard instance with data binding
    """
    attrs = view_comp.get("attributes", {})
    
    # Parse basic metric card properties
    metric_title = attrs.get('metric-title', 'Metric Title')
    format_value = attrs.get('format', 'number')
    value_color = attrs.get('value-color', '#2c3e50')
    value_size = int(attrs.get('value-size', 32))
    show_trend = parse_bool(attrs.get('show-trend'), True)
    positive_color = attrs.get('positive-color', '#27ae60')
    negative_color = attrs.get('negative-color', '#e74c3c')
    
    # Parse data binding
    domain_class, _, data_field = _parse_chart_data_binding(attrs, class_model, domain_model)
    
    # Create data binding if we have a domain class
    data_binding = None
    if domain_class:
        # Use sanitized name for the DataBinding
        binding_name = sanitize_name(f"{metric_title}_binding")
        data_binding = DataBinding(
            name=binding_name,
            domain_concept=domain_class,
            label_field=None,  # Metric cards don't need label field
            data_field=data_field,

        )
    
    # Create metric card
    metric_card = MetricCard(
        name=sanitize_name(metric_title),
        metric_title=metric_title,
        format=format_value,
        value_color=value_color,
        value_size=value_size,
        show_trend=show_trend,
        positive_color=positive_color,
        negative_color=negative_color,
        title=None,
        primary_color=value_color
    )
    
    metric_card.data_binding = data_binding
    _attach_chart_metadata(metric_card, view_comp)
    
    return metric_card
