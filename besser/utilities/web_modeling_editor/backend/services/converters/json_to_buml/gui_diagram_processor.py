import re
from besser.BUML.metamodel.gui import (
    GUIModel, Screen, Module, LineChart, Styling, Size,
    Position, Color, UnitSize, PositionType, ViewContainer,
    Text, BarChart
)

def process_gui_diagram(json_data, domain_model):
    gui_diagram = json_data.get('referenceDiagramData')
    class_model = json_data.get('model')

    title = gui_diagram.get('title', '')
    if ' ' in title:
        title = title.replace(' ', '_')

    gui_model = GUIModel(
        name=title,
        package="ai.factories",
        versionCode="1.0",
        versionName="1.0",
        modules=set(),
        description="Test GUI Model"
    )

    view_elements = gui_diagram.get('model', {})
    main_screen = view_elements.get('ROOT', {})

    dashboard_screen = Screen(
        name=main_screen.get('displayName', 'MainScreen'),
        description="This is the main screen",
        view_elements=set(),
        is_main_page=True,
    )

    # Screen styling
    color = Color(background_color=parse_color(main_screen.get('props', {}).get('background', '#FFFFFF')))
    size = Size(
        width=parse_numeric_value(main_screen.get('props', {}).get('width', 900), 900),
        height=parse_numeric_value(main_screen.get('props', {}).get('height', 1000), 1000),
        unit_size=UnitSize.PIXELS
    )
    position = Position(p_type=PositionType.RELATIVE)
    dashboard_screen.styling = Styling(size=size, position=position, color=color)

    screen_elements = set()
    for view_comp_id in main_screen.get('nodes', []):
        view_comp = view_elements.get(view_comp_id, {})
        if view_comp:
            parsed = parse_element(view_comp, view_elements, class_model, domain_model)
            if parsed:
                screen_elements.add(parsed)

    dashboard_screen.view_elements = screen_elements
    dashboard_module = Module(name="DashboardModule", screens={dashboard_screen})
    gui_model.modules = {dashboard_module}
    return gui_model


# ------------------------
# Helpers
# ------------------------

def parse_element(view_comp, view_elements, class_model, domain_model):
    """Dispatcher that selects the right parser based on type."""
    resolved_name = view_comp.get('type', {}).get('resolvedName')
    if resolved_name == 'Text':
        return parse_text(view_comp)
    elif resolved_name == 'LineChart':
        return parse_line_chart(view_comp, class_model, domain_model)
    elif resolved_name == 'BarChart':
        return parse_bar_chart(view_comp, class_model, domain_model)
    elif resolved_name == 'Container':
        return parse_container(view_comp, view_elements, class_model, domain_model)
    return None


def search_attribute(class_model, class_id, attr_id, domain_model):
    class_element = get_element_by_id(class_model, class_id)
    attr_element = get_element_by_id(class_model, attr_id)
    attr_element_name = clean_attribute_name(attr_element.get('name'))
    class_domain = domain_model.get_class_by_name(class_element.get('name'))
    for attr in class_domain.attributes:
        if attr.name == attr_element_name:
            return attr
    raise ValueError(f"{class_id} Error parsing GUI model: attribute <<{attr_id}>> not found")


def get_element_by_id(class_model, element_id):
    elements = class_model.get('elements', {})
    return elements.get(element_id)


def clean_attribute_name(attr_text):
    text = re.sub(r'^[\+\-\#]\s*', '', attr_text)       # remove + - #
    text = re.sub(r'\s*:\s*.*$', '', text)              # remove type annotation
    return text.strip()


def parse_numeric_value(value, default):
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
    if isinstance(value, dict):
        r = value.get('r', 0)
        g = value.get('g', 0)
        b = value.get('b', 0)
        a = value.get('a', 1)
        if a != 1:
            return f"rgba({r},{g},{b},{a})"
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    elif isinstance(value, str):
        return value
    else:
        return default


# ------------------------
# Parsers
# ------------------------

def parse_line_chart(view_comp, class_model, domain_model):
    line_chart = LineChart(
        name=view_comp.get('custom').get('displayName', 'LineChart'),
        x_axis=search_attribute(
            class_model,
            view_comp.get('props').get('class'),
            view_comp.get('props').get('attributeX'),
            domain_model
        ),
        y_axis=search_attribute(
            class_model,
            view_comp.get('props').get('class'),
            view_comp.get('props').get('attributeY'),
            domain_model
        ),
        line_width=view_comp.get('props').get('lineWidth', 2)
    )

    # Styling
    color = Color(
        line_color=parse_color(view_comp.get('props').get('lineColor', '#000000')),
        grid_color=parse_color(view_comp.get('props').get('gridColor', '#FFFFFF')),
        axis_color=parse_color(view_comp.get('props').get('axisColor', '#000FFF'))
    )
    size = Size(
        width=parse_numeric_value(view_comp.get('props').get('width', 300), 200),
        height=parse_numeric_value(view_comp.get('props').get('height', 300), 300),
        font_size=view_comp.get('props').get('fontSize', 12),
        unit_size=UnitSize.PERCENTAGE
    )
    position = Position(
        p_type=PositionType.ABSOLUTE,
        top=parse_numeric_value(view_comp.get('props').get('y', 0), 0),
        left=parse_numeric_value(view_comp.get('props').get('x', 0), 0)
    )
    line_chart.styling = Styling(size=size, position=position, color=color)

    return line_chart

def parse_bar_chart(view_comp, class_model, domain_model):
    bar_chart = BarChart(
        name=view_comp.get('custom').get('displayName', 'BarChart'),
        x_axis=search_attribute(
            class_model,
            view_comp.get('props').get('class'),
            view_comp.get('props').get('attributeX'),
            domain_model
        ),
        y_axis=search_attribute(
            class_model,
            view_comp.get('props').get('class'),
            view_comp.get('props').get('attributeY'),
            domain_model
        ),
        bar_width=view_comp.get('props').get('barWidth', 30)
    )

    # Styling
    color = Color(
        bar_color=parse_color(view_comp.get('props').get('barColor', '#000000')),
        grid_color=parse_color(view_comp.get('props').get('gridColor', '#FFFFFF')),
        axis_color=parse_color(view_comp.get('props').get('axisColor', '#000FFF'))
    )
    size = Size(
        width=parse_numeric_value(view_comp.get('props').get('width', 300), 200),
        height=parse_numeric_value(view_comp.get('props').get('height', 300), 300),
        font_size=view_comp.get('props').get('fontSize', 12),
        unit_size=UnitSize.PERCENTAGE
    )
    position = Position(
        p_type=PositionType.ABSOLUTE,
        top=parse_numeric_value(view_comp.get('props').get('y', 0), 0),
        left=parse_numeric_value(view_comp.get('props').get('x', 0), 0)
    )
    bar_chart.styling = Styling(size=size, position=position, color=color)

    return bar_chart

def parse_text(view_comp):
    text_el = Text(
        name=view_comp.get('custom').get('displayName', 'Text'),
        content=view_comp.get('props').get('text', 'Sample Text')
    )

    # Styling
    color = Color(text_color=parse_color(view_comp.get('props').get('color', '#00FFFF')))
    size = Size(
        width=parse_numeric_value(view_comp.get('props').get('width', 100), 100),
        height=parse_numeric_value(view_comp.get('props').get('height', 50), 50),
        font_size=view_comp.get('props').get('fontSize', 12),
        unit_size=UnitSize.PERCENTAGE
    )
    position = Position(
        p_type=PositionType.ABSOLUTE,
        top=parse_numeric_value(view_comp.get('props').get('y', 0), 0),
        left=parse_numeric_value(view_comp.get('props').get('x', 0), 0)
    )
    text_el.styling = Styling(size=size, position=position, color=color)

    return text_el


def parse_container(view_comp, view_elements, class_model, domain_model):
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
