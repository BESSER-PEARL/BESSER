import re
import json
from besser.BUML.metamodel.gui import (
    GUIModel, Screen, Module, LineChart, Styling, Size,
    Position, Color, UnitSize, PositionType, ViewContainer,
    Text, BarChart, PieChart, Alignment, RadarChart, RadialBarChart,
    ViewComponent, DataBinding
)

# Helper to resolve id to element in class_model
def get_element_by_id(class_model, element_id):
    """
    Resolve an element by its ID from the class_model (dict or list).
    Returns the element dict or None if not found.
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

    for el in class_model:
        if el.get('id') == element_id:
            return el
    return None

def process_gui_diagram(gui_diagram, class_model, domain_model):
    """
    Main entry point: Converts GrapesJS JSON and domain_model into a GUIModel instance.
    Handles style mapping, screens, and recursive component parsing.
    Args:
        gui_diagram: The GUI diagram data.
        class_model: The class model for object resolution.
        domain_model: The domain metamodel for object resolution.
    Returns: GUIModel instance.
    """

    # Save the input gui_diagram to a file for debugging
    with open("debug_gui_input.json", "w", encoding="utf-8") as f:
        json.dump(gui_diagram, f, ensure_ascii=False, indent=2)

    # --- Style mapping ---
    gui_model_json = gui_diagram
    style_map = {}
    for style_entry in gui_model_json.get('styles', []):
        selectors = style_entry.get('selectors', [])
        style = style_entry.get('style', {})

        # Build Styling object from style dict
        # Size
        width = style.get('width', 'auto')
        height = style.get('min-height', style.get('height', 'auto'))
        padding = style.get('padding', '0')
        margin = style.get('margin', '0')
        font_size = style.get('font-size', None)
        unit_size = UnitSize.PERCENTAGE if isinstance(width, str) and '%' in width else UnitSize.PIXELS
        size = Size(width=width, height=height, padding=padding, margin=margin, font_size=font_size, unit_size=unit_size)

        # Color
        color_val = style.get('color', None)
        background_color = style.get('background', style.get('background-color', '#FFFFFF'))
        border_color = style.get('border-color', '#000000')
        color = Color(background_color=background_color, text_color=color_val or '#000000', border_color=border_color)

        # Position
        pos_type = style.get('position', None)
        if pos_type:
            pos_type_enum = PositionType.RELATIVE if pos_type == 'relative' else PositionType.ABSOLUTE if pos_type == 'absolute' else PositionType.STATIC
        else:
            pos_type_enum = PositionType.STATIC
        top = style.get('top', 'auto')
        left = style.get('left', 'auto')
        right = style.get('right', 'auto')
        bottom = style.get('bottom', 'auto')
        alignment = style.get('text-align', None)
        position = Position(p_type=pos_type_enum, top=top, left=left, right=right, bottom=bottom, alignment=alignment)

        styling = Styling(size=size, position=position, color=color)

        for selector in selectors:
            if isinstance(selector, dict):
                key = selector.get('name')
                if key:
                    style_map[key] = styling
            elif isinstance(selector, str):
                style_map[selector] = styling

    def get_style_for_component(component):
        """
        Returns the style dict for a component by id, class, or type from the style_map.
        """
        # Try by id
        comp_id = component.get('attributes', {}).get('id')
        if comp_id and f"#{comp_id}" in style_map:
            return style_map[f"#{comp_id}"]
        # Try by class
        for cls in component.get('classes', []):
            if cls in style_map:
                return style_map[cls]
            if f".{cls}" in style_map:
                return style_map[f".{cls}"]
        # Try by type
        comp_type = component.get('type')
        if comp_type and comp_type in style_map:
            return style_map[comp_type]

        size = Size()
        position = Position()
        color = Color()
        styling = Styling(size=size, position=position, color=color)
        return styling

    def sanitize_name(name):
        if not isinstance(name, str):
            return name
        return name.replace(' ', '_')

    title = gui_diagram.get('title', 'GUI')
    title = sanitize_name(title)

    gui_model = GUIModel(
        name=title,
        package="ai.factories",
        versionCode="1.0",
        versionName="1.0",
        modules=set(),
        description="Test GUI Model"
    )

    # --- Parse pages/frames/components ---
    pages = gui_model_json.get('pages', [])
    screens = set()
    def parse_components(components, class_model):
        """
        Recursively parses a list of components into GUI metamodel elements.
        Args: components (list), class_model (dict or list)
        Returns: set of metamodel elements
        """
        elements = set()
        for c in components:
            # Get Styling object
            styling = get_style_for_component(c)

            # Dispatch to real element parsers
            c_type = c.get('type', '').lower()
            el = None
            if c_type == 'text':
                el = parse_text(c, get_style_for_component)
                el.styling = styling
            elif c_type == 'line-chart':
                el = parse_line_chart(c, class_model, domain_model)
                el.styling = styling
            elif c_type == 'bar-chart':
                el = parse_bar_chart(c, class_model, domain_model)
                el.styling = styling
            elif c_type == 'pie-chart':
                el = parse_pie_chart(c, class_model, domain_model)
                el.styling = styling
            elif c_type == 'radar-chart':
                el = parse_radar_chart(c, class_model, domain_model)
                el.styling = styling
            elif c_type == 'radial-bar-chart':
                el = parse_radial_bar_chart(c, class_model, domain_model)
                el.styling = styling
            else:
                el = ViewComponent(name=c.get('type', 'Component'), styling=styling)
            if el:
                elements.add(el)
        return elements

    for page in pages:
        screen_name = page.get('name', 'MainScreen')
        frames = page.get('frames', [])
        for frame in frames:
            comp = frame.get('component', {})
            # Compose screen
            screen = Screen(
                name=screen_name,
                description="This is the main screen",
                view_elements=set(),
                is_main_page=True,
            )
            # Screen styling (from style or default)

            styling = get_style_for_component(comp)
            if styling is None:
                color = Color(background_color="#FFFFFF")
                size = Size(width="100%", height="400px", unit_size=UnitSize.PERCENTAGE)
                position = Position(p_type=PositionType.RELATIVE)
                styling = Styling(size=size, position=position, color=color)
            screen.styling = styling

            children = comp.get('components', [])
            screen.view_elements = parse_components(children, class_model)
            screens.add(screen)

    dashboard_module = Module(name="DashboardModule", screens=screens)
    gui_model.modules = {dashboard_module}
    return gui_model


# ------------------------
# Helpers
# ------------------------

def parse_element(view_comp, view_elements, class_model, domain_model):
    """
    Dispatches to the correct parser for a view component based on its resolvedName.
    """
    resolved_name = view_comp.get('type', {}).get('resolvedName')
    if resolved_name == 'Text':
        return parse_text(view_comp, get_style_for_component)
    elif resolved_name == 'LineChart':
        return parse_line_chart(view_comp, class_model, domain_model)
    elif resolved_name == 'BarChart':
        return parse_bar_chart(view_comp, class_model, domain_model)
    elif resolved_name == 'PieChart':
        return parse_pie_chart(view_comp, class_model, domain_model)
    elif resolved_name == 'RadarChart':
        return parse_radar_chart(view_comp, class_model, domain_model)
    elif resolved_name == 'Container':
        return parse_container(view_comp, view_elements, class_model, domain_model)
    elif resolved_name == 'RadialBarChart':
        return parse_radial_bar_chart(view_comp, class_model, domain_model)
    return None


def clean_attribute_name(attr_text):
    """
    Cleans attribute names by removing visibility and type annotations.
    """
    text = re.sub(r'^[\+\-\#]\s*', '', attr_text)       # remove + - #
    text = re.sub(r'\s*:\s*.*$', '', text)              # remove type annotation
    return text.strip()


def parse_numeric_value(value, default):
    """
    Parses a numeric value from a string (supports %, px, or plain numbers).
    Returns float or default.
    """
    if isinstance(value, str):
        value = value.strip()
        # Case: percentage (e.g. "50%")
        if value.endswith('%'):
            try:
                return float(value.strip('%')) / 100
            except ValueError:
                return default
        # Case: pixels (e.g. "300px")
        if value.endswith('px'):
            try:
                return float(value.strip('px'))
            except ValueError:
                return default
        # Case: numeric string (e.g. "42" or "42.5")
        try:
            return float(value)
        except ValueError:
            return default

    return value if value is not None else default

def parse_color(value, default="#000000"):
    """
    Parses a color value from a string or dict, returns hex or rgba string.
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


# ------------------------
# Parsers
# ------------------------

def parse_line_chart(view_comp, class_model, domain_model):
    """
    Parses a line chart component, resolving data binding from class_model and domain_model.
    Args: view_comp (dict), class_model, domain_model
    Returns: LineChart instance
    """
    attrs = view_comp.get('attributes', {})
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
    domain_class = domain_model.get_class_by_name(data_source_name) if data_source_name else None
    label_field = None
    data_field = None
    if domain_class:
        if label_field_name:
            label_field = next((a for a in domain_class.attributes if a.name == label_field_name), None)
        if data_field_name:
            data_field = next((a for a in domain_class.attributes if a.name == data_field_name), None)
    data_binding = DataBinding(
        name=attrs.get('chart-title', 'LineChart') + "DataBinding",
        domain_concept=domain_class,
        label_field=label_field,
        data_field=data_field
    )
    chart_title = attrs.get('chart-title', 'LineChart')
    chart_title = chart_title.replace(' ', '_') if isinstance(chart_title, str) else chart_title
    line_chart = LineChart(
        name=chart_title,
        line_width=2
    )
    line_chart.data_binding = data_binding
    return line_chart

def parse_bar_chart(view_comp, class_model, domain_model):
    """
    Parses a bar chart component, resolving data binding from class_model and domain_model.
    Args: view_comp (dict), class_model, domain_model
    Returns: BarChart instance
    """
    attrs = view_comp.get('attributes', {})
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
    domain_class = domain_model.get_class_by_name(data_source_name) if data_source_name else None
    label_field = None
    data_field = None
    if domain_class:
        if label_field_name:
            label_field = next((a for a in domain_class.attributes if a.name == label_field_name), None)
        if data_field_name:
            data_field = next((a for a in domain_class.attributes if a.name == data_field_name), None)
    data_binding = DataBinding(
        name=attrs.get('chart-title', 'BarChart') + "DataBinding",
        domain_concept=domain_class,
        label_field=label_field,
        data_field=data_field
    )
    chart_title = attrs.get('chart-title', 'BarChart')
    chart_title = chart_title.replace(' ', '_') if isinstance(chart_title, str) else chart_title
    bar_chart = BarChart(
        name=chart_title,
        bar_width=30
    )
    bar_chart.data_binding = data_binding
    return bar_chart


def parse_radial_bar_chart(view_comp, class_model, domain_model):
    """
    Parses a radial bar chart component, mapping features/values to label/data fields.
    Args: view_comp (dict), class_model, domain_model
    Returns: RadialBarChart instance
    """
    attrs = view_comp.get('attributes', {})
    data_source_el = get_element_by_id(class_model, attrs.get('data-source'))
    features_el = get_element_by_id(class_model, attrs.get('features'))
    values_el = get_element_by_id(class_model, attrs.get('values'))
    data_source_name = data_source_el.get('name') if data_source_el else None
    features_name = features_el.get('name') if features_el else None
    values_name = values_el.get('name') if values_el else None
    data_source_name = data_source_el.get('name') if data_source_el else None
    features_name = features_el.get('name') if features_el else None
    values_name = values_el.get('name') if values_el else None
    if features_name:
        features_name = clean_attribute_name(features_name)
    if values_name:
        values_name = clean_attribute_name(values_name)
    domain_class = domain_model.get_class_by_name(data_source_name) if data_source_name else None
    label_field = None
    data_field = None
    if domain_class:
        if features_name:
            label_field = next((a for a in domain_class.attributes if a.name == features_name), None)
        if values_name:
            data_field = next((a for a in domain_class.attributes if a.name == values_name), None)
    data_binding = DataBinding(
        name=attrs.get('chart-title', 'RadialBarChart') + "DataBinding",
        domain_concept=domain_class,
        label_field=label_field,
        data_field=data_field
    )
    chart_title = attrs.get('chart-title', 'RadialBarChart')
    chart_title = chart_title.replace(' ', '_') if isinstance(chart_title, str) else chart_title
    radial_bar_chart = RadialBarChart(
        name=chart_title,
        start_angle=attrs.get('start-angle', 90),
        end_angle=attrs.get('end-angle', 450)
    )
    radial_bar_chart.data_binding = data_binding
    # Styling
    color = Color(
        color_palette=view_comp.get('props', {}).get('colorPalette', 'default')
    )
    size = Size(
        width=parse_numeric_value(view_comp.get('props', {}).get('width', 350), 350),
        height=parse_numeric_value(view_comp.get('props', {}).get('height', 350), 350),
        unit_size=UnitSize.PIXELS
    )
    position = Position(
        p_type=PositionType.ABSOLUTE,
        top=parse_numeric_value(view_comp.get('props', {}).get('y', 0), 0),
        left=parse_numeric_value(view_comp.get('props', {}).get('x', 0), 0)
    )
    radial_bar_chart.styling = Styling(size=size, position=position, color=color)
    return radial_bar_chart

def parse_radar_chart(view_comp, _, domain_model):
    """
    Parses a radar chart component, resolving data binding from domain_model.
    Args: view_comp (dict), domain_model
    Returns: RadarChart instance
    """
    attrs = view_comp.get('attributes', {})
    data_source_name = attrs.get('data-source')
    label_field_name = attrs.get('label-field')
    data_field_name = attrs.get('data-field')
    domain_class = domain_model.get_class_by_name(data_source_name) if data_source_name else None
    label_field = None
    data_field = None
    if domain_class:
        if label_field_name:
            label_field = next((a for a in domain_class.attributes if a.name == label_field_name), None)
        if data_field_name:
            data_field = next((a for a in domain_class.attributes if a.name == data_field_name), None)
    data_binding = DataBinding(
        name=attrs.get('chart-title', 'RadarChart') + "DataBinding",
        domain_concept=domain_class,
        label_field=label_field,
        data_field=data_field
    )
    chart_title = attrs.get('chart-title', 'RadarChart')
    chart_title = chart_title.replace(' ', '_') if isinstance(chart_title, str) else chart_title
    radar_chart = RadarChart(
        name=chart_title,
        show_grid=True,
        show_tooltip=True,
        show_radius_axis=True
    )
    radar_chart.data_binding = data_binding
    return radar_chart

def parse_text(view_comp, get_style_for_component):
    """
    Parses a text component and its styling from the view_comp dict.
    Returns: Text instance
    """
    content = None
    components = view_comp.get('components', [])
    if components and isinstance(components, list):
        first_child = components[0]
        if isinstance(first_child, dict) and first_child.get('type') == 'textnode':
            content = first_child.get('content', 'Sample Text')

    text_id = view_comp.get('attributes', {}).get('id', 'Text')
    text_id = text_id.replace(' ', '_') if isinstance(text_id, str) else text_id
    text_el = Text(
        name=text_id,
        content=content
    )

    #styling = get_style_for_component(view_comp)
    #text_el.styling = styling
    return text_el


def parse_container(view_comp, view_elements, class_model, domain_model):
    """
    Parses a container component and its children recursively.
    Args: view_comp (dict), view_elements (dict), class_model, domain_model
    Returns: ViewContainer instance
    """
    container = ViewContainer(
        name=view_comp.get('custom').get('displayName', 'Container'),
        description="A container element",
        view_elements=set()
    )

    # Styling
    color = Color(
        background_color=parse_color(view_comp.get('props').get('background', '#FFFFFF')),
        border_color=parse_color(view_comp.get('props').get('borderColor', '#000000'))
    )
    size = Size(
        width=parse_numeric_value(view_comp.get('props').get('width', 350), 350),
        height=parse_numeric_value(view_comp.get('props').get('height', 350), 350),
        unit_size=UnitSize.PERCENTAGE
    )
    position = Position(
        p_type=PositionType.ABSOLUTE,
        top=parse_numeric_value(view_comp.get('props').get('y', 0), 0),
        left=parse_numeric_value(view_comp.get('props').get('x', 0), 0)
    )
    container.styling = Styling(size=size, position=position, color=color)

    # Children
    children = set()
    for child_id in view_comp.get('nodes', []):
        child_comp = view_elements.get(child_id, {})
        if child_comp:
            parsed_child = parse_element(child_comp, view_elements, class_model, domain_model)
            if parsed_child:
                children.add(parsed_child)

    container.view_elements = children
    return container

def parse_pie_chart(view_comp, class_model, domain_model):
    """
    Parses a pie chart component, resolving data binding from class_model and domain_model.
    Args: view_comp (dict), class_model, domain_model
    Returns: PieChart instance
    """
    attrs = view_comp.get('attributes', {})
    data_source_id = attrs.get('data-source')
    label_field_id = attrs.get('label-field')
    data_field_id = attrs.get('data-field')
    data_source_el = get_element_by_id(class_model, data_source_id)
    label_field_el = get_element_by_id(class_model, label_field_id)
    data_field_el = get_element_by_id(class_model, data_field_id)
    data_source_name = data_source_el.get('name') if data_source_el else None
    label_field_name = label_field_el.get('name') if label_field_el else None
    data_field_name = data_field_el.get('name') if data_field_el else None
    if label_field_name:
        label_field_name = clean_attribute_name(label_field_name)
    if data_field_name:
        data_field_name = clean_attribute_name(data_field_name)
    domain_class = domain_model.get_class_by_name(data_source_name) if data_source_name else None
    label_field = None
    data_field = None
    if domain_class:
        if label_field_name:
            label_field = next((a for a in domain_class.attributes if a.name == label_field_name), None)
        if data_field_name:
            data_field = next((a for a in domain_class.attributes if a.name == data_field_name), None)
    data_binding = DataBinding(
        name=attrs.get('chart-title', 'PieChart') + "DataBinding",
        domain_concept=domain_class,
        label_field=label_field,
        data_field=data_field
    )
    chart_title = attrs.get('chart-title', 'PieChart')
    chart_title = chart_title.replace(' ', '_') if isinstance(chart_title, str) else chart_title
    pie_chart = PieChart(
        name=chart_title,
        show_legend=True,
        legend_position=Alignment.LEFT,
        show_labels=True,
        label_position=Alignment.INSIDE,
        padding_angle=0
    )
    pie_chart.data_binding = data_binding
    return pie_chart
