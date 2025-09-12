import re
from besser.BUML.metamodel.gui import (
    GUIModel, Screen, Module, LineChart, Styling, Size,
    Position, Color, UnitSize, PositionType, ViewContainer,
    Text
)

def process_gui_diagram(json_data, domain_model):
    gui_diagram = json_data.get('referenceDiagramData')
    class_model = json_data.get('model')

    title = gui_diagram.get('title', '')
    if ' ' in title:
        title = title.replace(' ', '_')

    gui_model = GUIModel(name=title,
                        package="ai.factories",
                        versionCode="1.0",
                        versionName="1.0",
                        modules=set(),
                        description="Test GUI Model")

    view_elements = gui_diagram.get('model', {})
    main_screen = view_elements.get('ROOT', {})

    dashboard_screen = Screen(name=main_screen.get('displayName', 'MainScreen'),
                        description="This is the main screen",
                        view_elements=set(),
                        is_main_page=True,
                        )

    screen_elements = set()
    for view_comp_id in main_screen.get('nodes', []):
        view_comp = view_elements.get(view_comp_id, {})
        if view_comp:
            if view_comp.get('type').get('resolvedName') == 'LineChart':
                line_chart = parse_line_chart(view_comp, class_model, domain_model)
                screen_elements.add(line_chart)

            if view_comp.get('type').get('resolvedName') == 'Text':
                text_el = parse_text(view_comp)
                screen_elements.add(text_el)

            if view_comp.get('type').get('resolvedName') == 'Container':
                container = parse_container(view_comp, view_elements, class_model, domain_model)
                screen_elements.add(container)

    dashboard_screen.view_elements = screen_elements
    dashboard_module = Module(name="DashboardModule", screens={dashboard_screen})
    gui_model.modules = {dashboard_module}
    return gui_model

def search_attribute(class_model, class_id, attr_id, domain_model):
    class_element = get_element_by_id(class_model, class_id)
    attr_element = get_element_by_id(class_model, attr_id)
    attr_element_name = clean_attribute_name(attr_element.get('name'))
    class_domain = domain_model.get_class_by_name(class_element.get('name'))
    for attr in class_domain.attributes:
        if attr.name == attr_element_name:
            attr_domain = attr
            return attr_domain
    raise ValueError(f"{class_id} Error parsing GUI model: attribute <<{attr_id}>> not found")

def get_element_by_id(class_model, element_id):
    elements = class_model.get('elements', {})
    return elements.get(element_id)

def clean_attribute_name(attr_text):
    # Remove visibility (+, -, #) at the start
    text = re.sub(r'^[\+\-\#]\s*', '', attr_text)
    # Remove type annotation (colon and everything after)
    text = re.sub(r'\s*:\s*.*$', '', text)
    # Remove leading/trailing spaces
    return text.strip()

def parse_dimension(value, default):
    if isinstance(value, str) and value.endswith('%'):
        try:
            return float(value.strip('%')) / 100
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

def parse_line_chart(view_comp, class_model, domain_model):
    line_chart = LineChart(name=view_comp.get('custom').get('displayName', 'LineChart'),
                           x_axis=search_attribute(class_model,
                                                   view_comp.get('props').get('class'),
                                                   view_comp.get('props').get('attributeX'),
                                                   domain_model),
                           y_axis=search_attribute(class_model,
                                                   view_comp.get('props').get('class'),
                                                   view_comp.get('props').get('attributeY'),
                                                   domain_model))

    # Styling
    # Color
    color = Color(line_color=parse_color(view_comp.get('props').get('lineColor', '#000000')),
                    grid_color=parse_color(view_comp.get('props').get('gridColor', '#FFFFFF')),
                    axis_color=parse_color(view_comp.get('props').get('axisColor', '#000FFF')))
    # Size
    size = Size(width=parse_dimension(view_comp.get('props').get('width', "30%"), "30%"),
                height=parse_dimension(view_comp.get('props').get('height', "20%"), "20%"),
                unit_size=UnitSize.PERCENTAGE)
    # Position
    position = Position(p_type=PositionType.ABSOLUTE,
                        top=view_comp.get('props').get('y', 0),
                        left=view_comp.get('props').get('x', 0))
    styling = Styling(size=size, position=position, color=color)

    line_chart.styling = styling

    return line_chart

def parse_text(view_comp):
    text_el = Text(name=view_comp.get('custom').get('displayName', 'Text'),
                   content=view_comp.get('props').get('text', 'Sample Text'))

    # Styling
    # Color
    color = Color(text_color=parse_color(view_comp.get('props').get('color', '#00FFFF')))
    # Size
    size = Size(width=parse_dimension(view_comp.get('props').get('width', "10%"), "10%"),
                height=parse_dimension(view_comp.get('props').get('height', "5%"), "5%"),
                font_size=view_comp.get('props').get('fontSize', 12),
                unit_size=UnitSize.PERCENTAGE)
    # Position
    position = Position(p_type=PositionType.ABSOLUTE,
                        top=view_comp.get('props').get('y', 0),
                        left=view_comp.get('props').get('x', 0))
    styling = Styling(size=size, position=position, color=color)

    text_el.styling = styling
    return text_el

def parse_container(view_comp, view_elements, class_model, domain_model):
    container = ViewContainer(name=view_comp.get('custom').get('displayName', 'Container'),
                              description="A container element",
                              view_elements=set())

    container_elements = set()

    # Styling
    # Color
    color = Color(background_color=parse_color(view_comp.get('props').get('background', '#FFFFFF')),
                    border_color=parse_color(view_comp.get('props').get('borderColor', '#000000')))
    # Size
    size = Size(width=parse_dimension(view_comp.get('props').get('width', "30%"), "30%"),
                height=parse_dimension(view_comp.get('props').get('height', "30%"), "30%"),
                unit_size=UnitSize.PERCENTAGE)
    # Position
    position = Position(p_type=PositionType.ABSOLUTE,
                        top=view_comp.get('props').get('y', 0),
                        left=view_comp.get('props').get('x', 0))
    styling = Styling(size=size, position=position, color=color)

    container.styling = styling

    for child in view_comp.get('nodes', []):
        child_comp = view_elements.get(child, {})
        if child_comp:
            if child_comp.get('type').get('resolvedName') == 'Text':
                text_el = parse_text(child_comp)
                container_elements.add(text_el)
            if child_comp.get('type').get('resolvedName') == 'LineChart':
                line_chart = parse_line_chart(child_comp, class_model, domain_model)
                container_elements.add(line_chart)
            if child_comp.get('type').get('resolvedName') == 'Container':
                sub_container = parse_container(child_comp, view_elements, class_model, domain_model)
                container_elements.add(sub_container)

    container.view_elements = container_elements

    return container
