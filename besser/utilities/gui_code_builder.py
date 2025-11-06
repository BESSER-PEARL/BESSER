"""
GUI BUML Code Builder

This module generates Python code from BUML GUI models.
It creates executable Python code that can recreate the GUI model programmatically.
"""

import os
from besser.BUML.metamodel.gui import GUIModel
from besser.BUML.metamodel.gui.graphical_ui import (
    ViewComponent,
    ViewContainer,
    Screen,
    Module,
    Button,
    Text,
    Image,
    InputField,
    Form,
    Menu,
    MenuItem,
    DataList,
    Link,
    EmbeddedContent,
    DataSourceElement,
)
from besser.BUML.metamodel.gui.dashboard import (
    LineChart, BarChart, PieChart, RadarChart, RadialBarChart, TableChart
)
from besser.BUML.metamodel.gui.events_actions import Event, Transition, Create, Read, Update, Delete
from besser.utilities.buml_code_builder import domain_model_to_code


def _escape_string(value: str | None) -> str:
    """Escape double quotes in strings for safe code generation."""
    if not value:
        return ""
    return value.replace('"', '\\"')


def _get_attr_name(element) -> str | None:
    """Return the name attribute of a BUML element if present."""
    return getattr(element, "name", None) if element is not None else None


def safe_var_name(name: str) -> str:
    """
    Convert a name to a safe Python variable name.
    
    Args:
        name: Original name
        
    Returns:
        Safe variable name
    """
    if not name:
        return "unnamed"
    # Replace spaces and special characters with underscores
    safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
    # Remove leading digits
    if safe_name and safe_name[0].isdigit():
        safe_name = f"_{safe_name}"
    # Remove consecutive underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    return safe_name.strip('_').lower() or "unnamed"


def gui_model_to_code(model: GUIModel, file_path: str, domain_model=None):
    """
    Generates Python code for a BUML GUI model and writes it to a specified file.
    
    Args:
        model (GUIModel): The BUML GUI model containing modules, screens, and components
        file_path (str): The path where the generated code will be saved
        domain_model (DomainModel, optional): Structural model to emit before the GUI so that
            data bindings can reference the same `domain_model` variable.
    
    Outputs:
        A Python file containing the code representation of the BUML GUI model
    """
    output_path = file_path if file_path.endswith('.py') else f"{file_path}.py"
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if domain_model is not None:
        domain_model_to_code(domain_model, output_path)
        file_mode = 'a'
    else:
        file_mode = 'w'
    
    with open(output_path, file_mode, encoding='utf-8') as f:
        if domain_model is not None:
            f.write("\n\n")

        # Write header
        f.write("###############\n")
        f.write("#  GUI MODEL  #\n")
        f.write("###############\n\n")
        
        # Write imports
        f.write("from besser.BUML.metamodel.gui import (\n")
        f.write("    GUIModel, Module, Screen,\n")
        f.write("    ViewComponent, ViewContainer,\n")
        f.write("    Button, ButtonType, ButtonActionType,\n")
        f.write("    Text, Image, Link, InputField, InputFieldType,\n")
        f.write("    Form, Menu, MenuItem, DataList,\n")
        f.write("    DataSource, DataSourceElement, EmbeddedContent,\n")
        f.write("    Styling, Size, Position, Color, Layout, LayoutType,\n")
        f.write("    UnitSize, PositionType, Alignment\n")
        f.write(")\n")
        f.write("from besser.BUML.metamodel.gui.dashboard import (\n")
        f.write("    LineChart, BarChart, PieChart, RadarChart, RadialBarChart, TableChart\n")
        f.write(")\n")
        f.write("from besser.BUML.metamodel.gui.events_actions import (\n")
        f.write("    Event, EventType, Transition, Create, Read, Update, Delete, Parameter\n")
        f.write(")\n")
        f.write("from besser.BUML.metamodel.gui.binding import DataBinding\n")
        f.write("\n")
        
        # Track created variables to avoid duplicates
        created_vars = set()
        # Track pending button events to write after all screens are defined
        pending_button_events = []
        
        # Process each module
        for module_idx, module in enumerate(sorted(model.modules, key=lambda m: m.name)):
            f.write(f"# Module: {module.name}\n")
            
            # Process each screen in the module
            for screen_idx, screen in enumerate(sorted(module.screens, key=lambda s: s.name)):
                screen_var = safe_var_name(screen.name)
                if screen_var in created_vars:
                    screen_var = f"{screen_var}_{screen_idx}"
                created_vars.add(screen_var)
                
                f.write(f"\n# Screen: {screen.name}\n")
                
                # Create screen (view_elements is required, will be set after processing children)
                screen_params = [f'name="{screen.name}"']
                screen_params.append(f'description="{screen.description}"' if screen.description else 'description=""')
                screen_params.append('view_elements=set()')  # Required parameter, will be populated later
                if hasattr(screen, 'is_main_page') and screen.is_main_page:
                    screen_params.append('is_main_page=True')
                if hasattr(screen, 'route_path') and screen.route_path:
                    screen_params.append(f'route_path="{_escape_string(screen.route_path)}"')
                if hasattr(screen, 'x_dpi') and screen.x_dpi:
                    screen_params.append(f'x_dpi="{screen.x_dpi}"')
                if hasattr(screen, 'y_dpi') and screen.y_dpi:
                    screen_params.append(f'y_dpi="{screen.y_dpi}"')
                if hasattr(screen, 'screen_size') and screen.screen_size:
                    screen_params.append(f'screen_size="{screen.screen_size}"')
                
                f.write(f"{screen_var} = Screen({', '.join(screen_params)})\n")
                
                # Write screen styling if present
                if screen.styling:
                    _write_styling(f, screen_var, screen.styling, created_vars)
                
                # Write screen metadata (page_id, component_id for React fidelity)
                if hasattr(screen, 'page_id') and screen.page_id:
                    f.write(f"{screen_var}.page_id = \"{_escape_string(screen.page_id)}\"\n")
                if hasattr(screen, 'component_id') and screen.component_id:
                    f.write(f"{screen_var}.component_id = \"{_escape_string(screen.component_id)}\"\n")
                
                # Process screen elements - preserve original order
                element_vars = []
                if hasattr(screen, 'view_elements') and screen.view_elements:
                    # Sort by display_order first (JSON order), then by name
                    sorted_elements = sorted(screen.view_elements, key=lambda e: (getattr(e, 'display_order', 999999), e.name))
                    for elem in sorted_elements:
                        elem_var = _write_component(f, elem, created_vars, screen_var, pending_button_events)
                        if elem_var:
                            element_vars.append(elem_var)
                
                # Assign elements to screen - Screen requires view_elements in __init__ too,
                # but we set them after creation for code readability
                if element_vars:
                    f.write(f"{screen_var}.view_elements = {{{', '.join(element_vars)}}}\n")
                else:
                    f.write(f"{screen_var}.view_elements = set()\n")
                
                # Set screen layout if present
                if hasattr(screen, 'layout') and screen.layout:
                    layout_var = _write_layout(f, screen.layout, created_vars, f"{screen_var}_layout")
                    f.write(f"{screen_var}.layout = {layout_var}\n")
                
                f.write("\n")
            
            # Write deferred button events after all screens are defined
            if pending_button_events:
                f.write("# Button events and transitions (written after all screens defined to avoid forward references)\n")
                for button_var, button in pending_button_events:
                    if hasattr(button, 'events') and button.events:
                        for event_idx, event in enumerate(button.events):
                            event_var = f"{button_var}_event_{event_idx}"
                            _write_event(f, event_var, event, created_vars)
                            if event_idx == 0:
                                f.write(f"{button_var}.events = {{{event_var}}}\n")
                            else:
                                f.write(f"{button_var}.events.add({event_var})\n")
                f.write("\n")
                # Clear the pending events for next module
                pending_button_events.clear()
            
            # Create module with screens
            module_var = safe_var_name(module.name)
            if module_var in created_vars:
                module_var = f"{module_var}_{module_idx}"
            created_vars.add(module_var)
            
            screen_vars = [safe_var_name(s.name) for s in sorted(module.screens, key=lambda s: s.name)]
            f.write(f"{module_var} = Module(\n")
            f.write(f'    name="{module.name}",\n')
            f.write(f'    screens={{{", ".join(screen_vars)}}}\n')
            f.write(")\n\n")
        
        # Create GUI model
        f.write("# GUI Model\n")
        module_vars = [safe_var_name(m.name) for m in sorted(model.modules, key=lambda m: m.name)]
        f.write("gui_model = GUIModel(\n")
        f.write(f'    name="{model.name}",\n')
        f.write(f'    package="{model.package}",\n')
        f.write(f'    versionCode="{model.versionCode}",\n')
        f.write(f'    versionName="{model.versionName}",\n')
        f.write(f'    modules={{{", ".join(module_vars)}}},\n')
        f.write(f'    description="{model.description}"\n')
        f.write(")\n")
    
    print(f"GUI model code saved to {file_path}")


def _write_component(f, component, created_vars, parent_var="", pending_button_events=None):
    """
    Write code for a GUI component.
    
    Args:
        f: File handle
        component: Component to write
        created_vars: Set of created variable names
        parent_var: Parent variable name (optional)
        pending_button_events: List to collect buttons with events for deferred writing
    
    Returns:
        Variable name of the created component
    """
    if pending_button_events is None:
        pending_button_events = []
    comp_var = safe_var_name(component.name)
    base_var = comp_var
    counter = 1
    while comp_var in created_vars:
        comp_var = f"{base_var}_{counter}"
        counter += 1
    created_vars.add(comp_var)
    
    # Determine component type and write creation code
    if isinstance(component, Button):
        _write_button(f, comp_var, component, created_vars, pending_button_events)
    elif isinstance(component, Text):
        _write_text(f, comp_var, component)
    elif isinstance(component, Image):
        _write_image(f, comp_var, component)
    elif isinstance(component, Link):
        _write_link(f, comp_var, component)
    elif isinstance(component, InputField):
        _write_input_field(f, comp_var, component)
    elif isinstance(component, Form):
        _write_form(f, comp_var, component, created_vars, pending_button_events)
    elif isinstance(component, Menu):
        _write_menu(f, comp_var, component, created_vars)
    elif isinstance(component, DataList):
        _write_data_list(f, comp_var, component, created_vars)
    elif isinstance(component, EmbeddedContent):
        _write_embedded_content(f, comp_var, component)
    elif isinstance(component, LineChart):
        _write_line_chart(f, comp_var, component)
    elif isinstance(component, BarChart):
        _write_bar_chart(f, comp_var, component)
    elif isinstance(component, PieChart):
        _write_pie_chart(f, comp_var, component)
    elif isinstance(component, RadarChart):
        _write_radar_chart(f, comp_var, component)
    elif isinstance(component, RadialBarChart):
        _write_radial_bar_chart(f, comp_var, component)
    elif isinstance(component, TableChart):
        _write_table_chart(f, comp_var, component)
    elif isinstance(component, ViewContainer):
        _write_container(f, comp_var, component, created_vars, pending_button_events)
        # Metadata already written by _write_container
        return comp_var
    else:
        # Generic ViewComponent - check if it has children (acts as container)
        if hasattr(component, 'view_elements') and component.view_elements:
            # It's a container but not typed as ViewContainer
            child_vars = []
            # Sort by display_order first (JSON order), then by name
            sorted_children = sorted(component.view_elements, key=lambda e: (getattr(e, 'display_order', 999999), e.name))
            for child in sorted_children:
                child_var = _write_component(f, child, created_vars, comp_var, pending_button_events)
                if child_var:
                    child_vars.append(child_var)
            
            children_str = f'{{{", ".join(child_vars)}}}' if child_vars else 'set()'
            f.write(f'{comp_var} = ViewContainer(name="{component.name}", description="{component.description or ""}", view_elements={children_str})\n')
            
            if hasattr(component, 'layout') and component.layout:
                layout_var = _write_layout(f, component.layout, created_vars, f"{comp_var}_layout")
                f.write(f'{comp_var}.layout = {layout_var}\n')
        else:
            # Simple ViewComponent with no children
            f.write(f'{comp_var} = ViewComponent(name="{component.name}", description="{component.description or ""}")\n')
    
    # Write styling if present
    if hasattr(component, 'styling') and component.styling:
        _write_styling(f, comp_var, component.styling, created_vars)
    
    # Write metadata properties if present
    if hasattr(component, 'display_order'):
        f.write(f'{comp_var}.display_order = {component.display_order}\n')
    if hasattr(component, 'component_id') and component.component_id:
        f.write(f'{comp_var}.component_id = "{_escape_string(component.component_id)}"\n')
    if hasattr(component, 'component_type') and component.component_type:
        f.write(f'{comp_var}.component_type = "{_escape_string(component.component_type)}"\n')
    if hasattr(component, 'tag_name') and component.tag_name:
        f.write(f'{comp_var}.tag_name = "{_escape_string(component.tag_name)}"\n')
    if hasattr(component, 'css_classes') and component.css_classes:
        classes_str = '", "'.join(_escape_string(c) for c in component.css_classes if c)
        if classes_str:
            f.write(f'{comp_var}.css_classes = ["{classes_str}"]\n')
    if hasattr(component, 'custom_attributes') and component.custom_attributes:
        # Write custom attributes as a dictionary
        attrs_items = []
        for k, v in component.custom_attributes.items():
            if isinstance(v, str):
                attrs_items.append(f'"{_escape_string(k)}": "{_escape_string(v)}"')
            elif v is None:
                attrs_items.append(f'"{_escape_string(k)}": None')
            else:
                attrs_items.append(f'"{_escape_string(k)}": {repr(v)}')
        if attrs_items:
            f.write(f'{comp_var}.custom_attributes = {{{", ".join(attrs_items)}}}\n')

    return comp_var


def _write_button(f, var_name, button, created_vars, pending_button_events):
    """Write code for a Button component."""
    params = [f'name="{button.name}"']
    params.append(f'description="{button.description}"' if button.description else 'description=""')
    params.append(f'label="{_escape_string(button.label)}"' if hasattr(button, 'label') and button.label else 'label=""')
    
    if hasattr(button, 'buttonType') and button.buttonType:
        params.append(f'buttonType=ButtonType.{button.buttonType.name}')
    if hasattr(button, 'actionType') and button.actionType:
        params.append(f'actionType=ButtonActionType.{button.actionType.name}')
    
    # Don't write targetScreen here - will be handled via events
    # This avoids forward reference issues
    
    f.write(f'{var_name} = Button({", ".join(params)})\n')
    
    # Store button info for later event/action writing (after all screens defined)
    if hasattr(button, 'events') and button.events:
        pending_button_events.append((var_name, button))


def _write_event(f, var_name, event, created_vars):
    """Write code for an Event."""
    created_vars.add(var_name)
    
    # Write actions first
    action_vars = []
    if hasattr(event, 'actions') and event.actions:
        for action_idx, action in enumerate(event.actions):
            action_var = f"{var_name}_action_{action_idx}"
            _write_action(f, action_var, action, created_vars)
            action_vars.append(action_var)
    
    # Write event
    event_type = f'EventType.{event.event_type.name}' if hasattr(event, 'event_type') and event.event_type else 'EventType.OnClick'
    actions_str = f'{{{", ".join(action_vars)}}}' if action_vars else '{}'
    f.write(f'{var_name} = Event(name="{event.name}", event_type={event_type}, actions={actions_str})\n')


def _write_action(f, var_name, action, created_vars):
    """Write code for an Action."""
    created_vars.add(var_name)
    
    # Extract parameters if present
    params_code = ""
    if hasattr(action, 'parameters') and action.parameters:
        param_vars = []
        for param_idx, param in enumerate(action.parameters):
            param_var = f"{var_name}_param_{param_idx}"
            created_vars.add(param_var)
            f.write(f'{param_var} = Parameter(name="{param.name}", value="{param.value if hasattr(param, "value") else ""}")\n')
            param_vars.append(param_var)
        params_code = f', parameters={{{", ".join(param_vars)}}}'
    
    if isinstance(action, Transition):
        target_screen = 'None  # TODO: Set target_screen reference'
        if hasattr(action, 'target_screen') and action.target_screen:
            target_screen_var = safe_var_name(action.target_screen.name)
            target_screen = f'{target_screen_var}'
        f.write(f'{var_name} = Transition(name="{action.name}", description="{action.description or ""}", target_screen={target_screen}{params_code})\n')
    
    elif isinstance(action, Create):
        target_class = 'None  # TODO: Set target_class reference'
        if hasattr(action, 'target_class') and action.target_class:
            target_class = f'# {action.target_class.name}'
        f.write(f'{var_name} = Create(name="{action.name}", description="{action.description or ""}", target_class={target_class}{params_code})\n')
    
    elif isinstance(action, Read):
        target_class = 'None  # TODO: Set target_class reference'
        if hasattr(action, 'target_class') and action.target_class:
            target_class = f'# {action.target_class.name}'
        f.write(f'{var_name} = Read(name="{action.name}", description="{action.description or ""}", target_class={target_class}{params_code})\n')
    
    elif isinstance(action, Update):
        target_class = 'None  # TODO: Set target_class reference'
        if hasattr(action, 'target_class') and action.target_class:
            target_class = f'# {action.target_class.name}'
        f.write(f'{var_name} = Update(name="{action.name}", description="{action.description or ""}", target_class={target_class}{params_code})\n')
    
    elif isinstance(action, Delete):
        target_class = 'None  # TODO: Set target_class reference'
        if hasattr(action, 'target_class') and action.target_class:
            target_class = f'# {action.target_class.name}'
        f.write(f'{var_name} = Delete(name="{action.name}", description="{action.description or ""}", target_class={target_class}{params_code})\n')


def _write_text(f, var_name, text):
    """Write code for a Text component."""
    content = text.content.replace('"', '\\"').replace('\n', '\\n') if hasattr(text, 'content') and text.content else ""
    f.write(f'{var_name} = Text(name="{text.name}", content="{content}", description="{text.description or ""}")\n')


def _write_image(f, var_name, image):
    """Write code for an Image component."""
    params = [f'name="{image.name}"', f'description="{image.description or ""}"']
    image_source = getattr(image, "source", None)
    if image_source:
        params.append(f'source="{_escape_string(image_source)}"')
    f.write(f'{var_name} = Image({", ".join(params)})\n')


def _write_link(f, var_name, link):
    """Write code for a Link component."""
    params = [
        f'name="{link.name}"',
        f'description="{link.description or ""}"',
        f'label="{_escape_string(getattr(link, "label", ""))}"',
    ]
    if getattr(link, "url", None):
        params.append(f'url="{_escape_string(link.url)}"')
    if getattr(link, "target", None):
        params.append(f'target="{_escape_string(link.target)}"')
    if getattr(link, "rel", None):
        params.append(f'rel="{_escape_string(link.rel)}"')
    f.write(f'{var_name} = Link({", ".join(params)})\n')


def _write_embedded_content(f, var_name, embedded):
    """Write code for embedded content components."""
    params = [
        f'name="{embedded.name}"',
        f'description="{embedded.description or ""}"',
    ]
    if getattr(embedded, "source", None):
        params.append(f'source="{_escape_string(embedded.source)}"')
    if getattr(embedded, "content_type", None):
        params.append(f'content_type="{_escape_string(embedded.content_type)}"')
    f.write(f'{var_name} = EmbeddedContent({", ".join(params)})\n')


def _write_input_field(f, var_name, input_field):
    """Write code for an InputField component."""
    params = [f'name="{input_field.name}"']
    params.append(f'description="{input_field.description or ""}"')
    
    if hasattr(input_field, 'field_type') and input_field.field_type:
        params.append(f'field_type=InputFieldType.{input_field.field_type.name}')
    
    if hasattr(input_field, 'validationRules') and input_field.validationRules:
        params.append(f'validationRules="{input_field.validationRules}"')
    
    f.write(f'{var_name} = InputField({", ".join(params)})\n')


def _write_form(f, var_name, form, created_vars, pending_button_events):
    """Write code for a Form component."""
    # Write input fields first
    field_vars = []
    if hasattr(form, 'inputFields') and form.inputFields:
        for field in form.inputFields:
            field_var = _write_component(f, field, created_vars, var_name, pending_button_events)
            if field_var:
                field_vars.append(field_var)
    
    fields_str = f'{{{", ".join(field_vars)}}}' if field_vars else '{}'
    f.write(f'{var_name} = Form(name="{form.name}", description="{form.description or ""}", inputFields={fields_str})\n')


def _write_menu(f, var_name, menu, created_vars):
    """Write code for a Menu component."""
    # Write menu items first
    item_vars = []
    if hasattr(menu, 'menuItems') and menu.menuItems:
        for item_idx, item in enumerate(menu.menuItems):
            item_var = f"{var_name}_item_{item_idx}"
            created_vars.add(item_var)
            params = [f'label="{_escape_string(getattr(item, "label", ""))}"']
            if getattr(item, "url", None):
                params.append(f'url="{_escape_string(item.url)}"')
            if getattr(item, "target", None):
                params.append(f'target="{_escape_string(item.target)}"')
            if getattr(item, "rel", None):
                params.append(f'rel="{_escape_string(item.rel)}"')
            f.write(f'{item_var} = MenuItem({", ".join(params)})\n')
            item_vars.append(item_var)
    
    items_str = f'{{{", ".join(item_vars)}}}' if item_vars else '{}'
    f.write(f'{var_name} = Menu(name="{menu.name}", description="{menu.description or ""}", menuItems={items_str})\n')


def _write_data_list(f, var_name, data_list, created_vars):
    """Write code for a DataList component."""
    # Write data sources first
    source_vars = []
    if hasattr(data_list, 'list_sources') and data_list.list_sources:
        for source_idx, source in enumerate(data_list.list_sources):
            source_var = f"{var_name}_source_{source_idx}"
            created_vars.add(source_var)
            source_name = _escape_string(getattr(source, "name", ""))
            f.write(f'{source_var} = DataSourceElement(name="{source_name}")\n')
            _update_data_source_element(f, source_var, source)
            source_vars.append(source_var)
    
    sources_str = f'{{{", ".join(source_vars)}}}' if source_vars else '{}'
    f.write(f'{var_name} = DataList(name="{data_list.name}", description="{data_list.description or ""}", list_sources={sources_str})\n')


def _write_line_chart(f, var_name, chart):
    """Write code for a LineChart component."""
    params = [f'name="{chart.name}"']
    if hasattr(chart, 'title') and chart.title:
        params.append(f'title="{_escape_string(chart.title)}"')
    if hasattr(chart, 'primary_color') and chart.primary_color:
        params.append(f'primary_color="{_escape_string(chart.primary_color)}"')
    if hasattr(chart, 'line_width'):
        params.append(f'line_width={chart.line_width}')
    if hasattr(chart, 'show_grid'):
        params.append(f'show_grid={chart.show_grid}')
    if hasattr(chart, 'show_legend'):
        params.append(f'show_legend={chart.show_legend}')
    if hasattr(chart, 'show_tooltip'):
        params.append(f'show_tooltip={chart.show_tooltip}')
    if hasattr(chart, 'curve_type'):
        params.append(f'curve_type="{chart.curve_type}"')
    if hasattr(chart, 'animate'):
        params.append(f'animate={chart.animate}')
    if hasattr(chart, 'legend_position'):
        params.append(f'legend_position="{chart.legend_position}"')
    if hasattr(chart, 'grid_color'):
        params.append(f'grid_color="{chart.grid_color}"')
    if hasattr(chart, 'dot_size'):
        params.append(f'dot_size={chart.dot_size}')
    
    f.write(f'{var_name} = LineChart({", ".join(params)})\n')
    
    # Write data binding if present
    if hasattr(chart, 'data_binding') and chart.data_binding:
        _write_data_binding_assignment(f, var_name, chart.data_binding)


def _write_bar_chart(f, var_name, chart):
    """Write code for a BarChart component."""
    params = [f'name="{chart.name}"']
    if hasattr(chart, 'title') and chart.title:
        params.append(f'title="{_escape_string(chart.title)}"')
    if hasattr(chart, 'primary_color') and chart.primary_color:
        params.append(f'primary_color="{_escape_string(chart.primary_color)}"')
    if hasattr(chart, 'bar_width'):
        params.append(f'bar_width={chart.bar_width}')
    if hasattr(chart, 'orientation'):
        params.append(f'orientation="{chart.orientation}"')
    if hasattr(chart, 'show_grid'):
        params.append(f'show_grid={chart.show_grid}')
    if hasattr(chart, 'show_legend'):
        params.append(f'show_legend={chart.show_legend}')
    if hasattr(chart, 'show_tooltip'):
        params.append(f'show_tooltip={chart.show_tooltip}')
    if hasattr(chart, 'stacked'):
        params.append(f'stacked={chart.stacked}')
    if hasattr(chart, 'animate'):
        params.append(f'animate={chart.animate}')
    if hasattr(chart, 'legend_position'):
        params.append(f'legend_position="{chart.legend_position}"')
    if hasattr(chart, 'grid_color'):
        params.append(f'grid_color="{chart.grid_color}"')
    if hasattr(chart, 'bar_gap'):
        params.append(f'bar_gap={chart.bar_gap}')
    
    f.write(f'{var_name} = BarChart({", ".join(params)})\n')
    if hasattr(chart, 'data_binding') and chart.data_binding:
        _write_data_binding_assignment(f, var_name, chart.data_binding)


def _write_pie_chart(f, var_name, chart):
    """Write code for a PieChart component."""
    params = [f'name="{chart.name}"']
    if hasattr(chart, 'title') and chart.title:
        params.append(f'title="{_escape_string(chart.title)}"')
    if hasattr(chart, 'primary_color') and chart.primary_color:
        params.append(f'primary_color="{_escape_string(chart.primary_color)}"')
    if hasattr(chart, 'show_legend'):
        params.append(f'show_legend={chart.show_legend}')
    if hasattr(chart, 'legend_position'):
        params.append(f'legend_position=Alignment.{chart.legend_position.name}')
    if hasattr(chart, 'show_labels'):
        params.append(f'show_labels={chart.show_labels}')
    if hasattr(chart, 'label_position'):
        params.append(f'label_position=Alignment.{chart.label_position.name}')
    if hasattr(chart, 'padding_angle'):
        params.append(f'padding_angle={chart.padding_angle}')
    if hasattr(chart, 'inner_radius'):
        params.append(f'inner_radius={chart.inner_radius}')
    if hasattr(chart, 'outer_radius'):
        params.append(f'outer_radius={chart.outer_radius}')
    if hasattr(chart, 'start_angle'):
        params.append(f'start_angle={chart.start_angle}')
    if hasattr(chart, 'end_angle'):
        params.append(f'end_angle={chart.end_angle}')
    
    f.write(f'{var_name} = PieChart({", ".join(params)})\n')
    if hasattr(chart, 'data_binding') and chart.data_binding:
        _write_data_binding_assignment(f, var_name, chart.data_binding)


def _write_radar_chart(f, var_name, chart):
    """Write code for a RadarChart component."""
    params = [f'name="{chart.name}"']
    if hasattr(chart, 'title') and chart.title:
        params.append(f'title="{_escape_string(chart.title)}"')
    if hasattr(chart, 'primary_color') and chart.primary_color:
        params.append(f'primary_color="{_escape_string(chart.primary_color)}"')
    if hasattr(chart, 'show_grid'):
        params.append(f'show_grid={chart.show_grid}')
    if hasattr(chart, 'show_tooltip'):
        params.append(f'show_tooltip={chart.show_tooltip}')
    if hasattr(chart, 'show_radius_axis'):
        params.append(f'show_radius_axis={chart.show_radius_axis}')
    if hasattr(chart, 'show_legend'):
        params.append(f'show_legend={chart.show_legend}')
    if hasattr(chart, 'legend_position'):
        params.append(f'legend_position="{chart.legend_position}"')
    if hasattr(chart, 'dot_size'):
        params.append(f'dot_size={chart.dot_size}')
    if hasattr(chart, 'grid_type'):
        params.append(f'grid_type="{chart.grid_type}"')
    if hasattr(chart, 'stroke_width'):
        params.append(f'stroke_width={chart.stroke_width}')
    
    f.write(f'{var_name} = RadarChart({", ".join(params)})\n')
    if hasattr(chart, 'data_binding') and chart.data_binding:
        _write_data_binding_assignment(f, var_name, chart.data_binding)


def _write_radial_bar_chart(f, var_name, chart):
    """Write code for a RadialBarChart component."""
    params = [f'name="{chart.name}"']
    if hasattr(chart, 'title') and chart.title:
        params.append(f'title="{_escape_string(chart.title)}"')
    if hasattr(chart, 'primary_color') and chart.primary_color:
        params.append(f'primary_color="{_escape_string(chart.primary_color)}"')
    if hasattr(chart, 'start_angle'):
        params.append(f'start_angle={chart.start_angle}')
    if hasattr(chart, 'end_angle'):
        params.append(f'end_angle={chart.end_angle}')
    if hasattr(chart, 'inner_radius'):
        params.append(f'inner_radius={chart.inner_radius}')
    if hasattr(chart, 'outer_radius'):
        params.append(f'outer_radius={chart.outer_radius}')
    if hasattr(chart, 'show_legend'):
        params.append(f'show_legend={chart.show_legend}')
    if hasattr(chart, 'legend_position'):
        params.append(f'legend_position="{chart.legend_position}"')
    if hasattr(chart, 'show_tooltip'):
        params.append(f'show_tooltip={chart.show_tooltip}')
    
    f.write(f'{var_name} = RadialBarChart({", ".join(params)})\n')
    if hasattr(chart, 'data_binding') and chart.data_binding:
        _write_data_binding_assignment(f, var_name, chart.data_binding)


def _write_table_chart(f, var_name, chart):
    """Write code for a TableChart component."""
    params = [f'name="{chart.name}"']
    if hasattr(chart, 'title') and chart.title:
        params.append(f'title="{_escape_string(chart.title)}"')
    if hasattr(chart, 'primary_color') and chart.primary_color:
        params.append(f'primary_color="{_escape_string(chart.primary_color)}"')
    if hasattr(chart, 'show_header'):
        params.append(f'show_header={chart.show_header}')
    if hasattr(chart, 'striped_rows'):
        params.append(f'striped_rows={chart.striped_rows}')
    if hasattr(chart, 'show_pagination'):
        params.append(f'show_pagination={chart.show_pagination}')
    if hasattr(chart, 'rows_per_page'):
        params.append(f'rows_per_page={chart.rows_per_page}')
    if hasattr(chart, 'columns') and chart.columns:
        column_literals = ", ".join(f'"{_escape_string(col)}"' for col in chart.columns if col)
        params.append(f'columns=[{column_literals}]')

    f.write(f'{var_name} = TableChart({", ".join(params)})\n')
    if hasattr(chart, 'data_binding') and chart.data_binding:
        _write_data_binding_assignment(f, var_name, chart.data_binding)


def _write_container(f, var_name, container, created_vars, pending_button_events):
    """Write code for a ViewContainer component."""
    # Write child elements first - preserve original order
    child_vars = []
    if hasattr(container, 'view_elements') and container.view_elements:
        # Sort by display_order first (JSON order), then by name
        sorted_children = sorted(container.view_elements, key=lambda e: (getattr(e, 'display_order', 999999), e.name))
        for child in sorted_children:
            child_var = _write_component(f, child, created_vars, var_name, pending_button_events)
            if child_var:
                child_vars.append(child_var)
    
    # Create ViewContainer with view_elements as required parameter
    children_str = f'{{{", ".join(child_vars)}}}' if child_vars else 'set()'
    f.write(f'{var_name} = ViewContainer(name="{container.name}", description="{container.description or ""}", view_elements={children_str})\n')
    
    # Write styling if present
    if hasattr(container, 'styling') and container.styling:
        _write_styling(f, var_name, container.styling, created_vars)
    
    # Set layout if present
    if hasattr(container, 'layout') and container.layout:
        layout_var = _write_layout(f, container.layout, created_vars, f"{var_name}_layout")
        f.write(f'{var_name}.layout = {layout_var}\n')
    
    # Write metadata properties if present
    if hasattr(container, 'display_order'):
        f.write(f'{var_name}.display_order = {container.display_order}\n')
    if hasattr(container, 'component_id') and container.component_id:
        f.write(f'{var_name}.component_id = "{_escape_string(container.component_id)}"\n')
    if hasattr(container, 'component_type') and container.component_type:
        f.write(f'{var_name}.component_type = "{_escape_string(container.component_type)}"\n')
    if hasattr(container, 'tag_name') and container.tag_name:
        f.write(f'{var_name}.tag_name = "{_escape_string(container.tag_name)}"\n')
    if hasattr(container, 'css_classes') and container.css_classes:
        classes_str = '", "'.join(_escape_string(c) for c in container.css_classes if c)
        if classes_str:
            f.write(f'{var_name}.css_classes = ["{classes_str}"]\n')
    if hasattr(container, 'custom_attributes') and container.custom_attributes:
        # Write custom attributes as a dictionary
        attrs_items = []
        for k, v in container.custom_attributes.items():
            if isinstance(v, str):
                attrs_items.append(f'"{_escape_string(k)}": "{_escape_string(v)}"')
            elif v is None:
                attrs_items.append(f'"{_escape_string(k)}": None')
            else:
                attrs_items.append(f'"{_escape_string(k)}": {repr(v)}')
        if attrs_items:
            f.write(f'{var_name}.custom_attributes = {{{", ".join(attrs_items)}}}\n')


def _write_styling(f, component_var, styling, created_vars):
    """Write code for Styling object."""
    styling_var = f"{component_var}_styling"
    if styling_var in created_vars:
        return
    created_vars.add(styling_var)
    
    # Write Size, Position, Color objects
    size_var = f"{styling_var}_size"
    pos_var = f"{styling_var}_pos"
    color_var = f"{styling_var}_color"
    
    # Size
    if hasattr(styling, 'size') and styling.size:
        size_params = []
        if hasattr(styling.size, 'width') and styling.size.width:
            size_params.append(f'width="{styling.size.width}"')
        if hasattr(styling.size, 'height') and styling.size.height:
            size_params.append(f'height="{styling.size.height}"')
        if hasattr(styling.size, 'padding') and styling.size.padding:
            size_params.append(f'padding="{styling.size.padding}"')
        if hasattr(styling.size, 'margin') and styling.size.margin:
            size_params.append(f'margin="{styling.size.margin}"')
        if hasattr(styling.size, 'font_size') and styling.size.font_size:
            size_params.append(f'font_size="{styling.size.font_size}"')
        if hasattr(styling.size, 'line_height') and styling.size.line_height:
            size_params.append(f'line_height="{styling.size.line_height}"')
        if hasattr(styling.size, 'unit_size') and styling.size.unit_size:
            size_params.append(f'unit_size=UnitSize.{styling.size.unit_size.name}')
        
        if size_params:
            f.write(f'{size_var} = Size({", ".join(size_params)})\n')
        else:
            f.write(f'{size_var} = Size()\n')
    else:
        f.write(f'{size_var} = Size()\n')
    
    # Position
    if hasattr(styling, 'position') and styling.position:
        pos_params = []
        if hasattr(styling.position, 'alignment') and styling.position.alignment:
            # Check if alignment is an Alignment enum or a string
            alignment_val = styling.position.alignment
            if hasattr(alignment_val, 'name'):
                # It's an Alignment enum
                pos_params.append(f'alignment=Alignment.{alignment_val.name}')
            elif isinstance(alignment_val, str):
                # It's already a string - check if it contains "Alignment."
                if 'Alignment.' in alignment_val:
                    # Extract enum name and use it properly
                    enum_name = alignment_val.split('.')[-1]
                    pos_params.append(f'alignment=Alignment.{enum_name}')
                else:
                    # Try to map common alignment strings to enum values
                    alignment_map = {
                        'center': 'CENTER',
                        'left': 'LEFT',
                        'right': 'RIGHT',
                        'top': 'TOP',
                        'bottom': 'BOTTOM'
                    }
                    mapped = alignment_map.get(alignment_val.lower())
                    if mapped:
                        pos_params.append(f'alignment=Alignment.{mapped}')
                    else:
                        pos_params.append(f'alignment="{alignment_val}"')
        if hasattr(styling.position, 'top') and styling.position.top:
            pos_params.append(f'top="{styling.position.top}"')
        if hasattr(styling.position, 'left') and styling.position.left:
            pos_params.append(f'left="{styling.position.left}"')
        if hasattr(styling.position, 'right') and styling.position.right:
            pos_params.append(f'right="{styling.position.right}"')
        if hasattr(styling.position, 'bottom') and styling.position.bottom:
            pos_params.append(f'bottom="{styling.position.bottom}"')
        if hasattr(styling.position, 'z_index') and styling.position.z_index is not None:
            pos_params.append(f'z_index={styling.position.z_index}')
        if hasattr(styling.position, 'p_type') and styling.position.p_type:
            pos_params.append(f'p_type=PositionType.{styling.position.p_type.name}')
        
        if pos_params:
            f.write(f'{pos_var} = Position({", ".join(pos_params)})\n')
        else:
            f.write(f'{pos_var} = Position()\n')
    else:
        f.write(f'{pos_var} = Position()\n')
    
    # Color
    if hasattr(styling, 'color') and styling.color:
        color_params = []
        if hasattr(styling.color, 'background_color') and styling.color.background_color:
            color_params.append(f'background_color="{styling.color.background_color}"')
        if hasattr(styling.color, 'text_color') and styling.color.text_color:
            color_params.append(f'text_color="{styling.color.text_color}"')
        if hasattr(styling.color, 'border_color') and styling.color.border_color:
            color_params.append(f'border_color="{styling.color.border_color}"')
        if hasattr(styling.color, 'opacity') and styling.color.opacity:
            color_params.append(f'opacity="{styling.color.opacity}"')
        
        if color_params:
            f.write(f'{color_var} = Color({", ".join(color_params)})\n')
        else:
            f.write(f'{color_var} = Color()\n')
    else:
        f.write(f'{color_var} = Color()\n')
    
    # Styling
    f.write(f'{styling_var} = Styling(size={size_var}, position={pos_var}, color={color_var})\n')
    
    # Set layout if present
    if hasattr(styling, 'layout') and styling.layout:
        layout_var = _write_layout(f, styling.layout, created_vars, f"{styling_var}_layout")
        f.write(f'{styling_var}.layout = {layout_var}\n')
    
    # Assign styling to component
    f.write(f'{component_var}.styling = {styling_var}\n')


def _write_data_binding_assignment(f, var_name, binding):
    if not binding:
        return
    binding_var = f"{var_name}_binding"
    binding_name = _escape_string(getattr(binding, "name", binding_var))
    domain_name = _get_attr_name(getattr(binding, "domain_concept", None))
    label_name = _get_attr_name(getattr(binding, "label_field", None))
    data_name = _get_attr_name(getattr(binding, "data_field", None))

    # DataBinding requires domain_concept as first parameter, so we need to handle missing domain
    if domain_name:
        escaped_domain = _escape_string(domain_name)
        f.write("domain_model_ref = globals().get('domain_model')\n")
        f.write(f"{binding_var}_domain = None\n")
        f.write("if domain_model_ref is not None:\n")
        f.write(f"    {binding_var}_domain = domain_model_ref.get_class_by_name(\"{escaped_domain}\")\n")
        f.write(f"if {binding_var}_domain:\n")
        f.write(f"    {binding_var} = DataBinding(domain_concept={binding_var}_domain)\n")
        if label_name:
            escaped_label = _escape_string(label_name)
            f.write(f"    {binding_var}.label_field = next((attr for attr in {binding_var}_domain.attributes if attr.name == \"{escaped_label}\"), None)\n")
        if data_name:
            escaped_data = _escape_string(data_name)
            f.write(f"    {binding_var}.data_field = next((attr for attr in {binding_var}_domain.attributes if attr.name == \"{escaped_data}\"), None)\n")
        f.write("else:\n")
        f.write(f"    # Domain class '{escaped_domain}' not resolved; data binding skipped.\n")
        f.write(f"    {binding_var} = None\n")
        f.write(f"if {binding_var}:\n")
        f.write(f"    {var_name}.data_binding = {binding_var}\n")
    else:
        f.write(f"# DataBinding for {var_name} skipped: no domain concept specified.\n")


def _update_data_source_element(f, var_name, source):
    domain_name = _get_attr_name(getattr(source, "dataSourceClass", None))
    field_names = [
        name
        for name in (
            _get_attr_name(field) for field in getattr(source, "fields", set())
        )
        if name
    ]
    extra_field_names = getattr(source, "field_names", None) or []
    for name in extra_field_names:
        if name and name not in field_names:
            field_names.append(name)

    label_name = (
        _get_attr_name(getattr(source, "label_field", None))
        or getattr(source, "label_field_name", None)
    )
    value_name = (
        _get_attr_name(getattr(source, "value_field", None))
        or getattr(source, "value_field_name", None)
    )

    if not any([domain_name, field_names, label_name, value_name]):
        return

    f.write("domain_model_ref = globals().get('domain_model')\n")
    f.write(f"{var_name}_domain = None\n")
    if domain_name:
        escaped_domain = _escape_string(domain_name)
        f.write("if domain_model_ref is not None:\n")
        f.write(f"    {var_name}_domain = domain_model_ref.get_class_by_name(\"{escaped_domain}\")\n")
        f.write(f"if {var_name}_domain:\n")
        f.write(f"    {var_name}.dataSourceClass = {var_name}_domain\n")
        if field_names:
            fields_list = repr(field_names)
            f.write(f"    {var_name}.field_names = {fields_list}\n")
            f.write(f"    {var_name}.fields = set(attr for attr in {var_name}_domain.attributes if attr.name in {fields_list})\n")
        if label_name:
            escaped_label = _escape_string(label_name)
            f.write(f"    {var_name}.label_field = next((attr for attr in {var_name}_domain.attributes if attr.name == \"{escaped_label}\"), None)\n")
            f.write(f"    {var_name}.label_field_name = \"{escaped_label}\"\n")
        if value_name:
            escaped_value = _escape_string(value_name)
            f.write(f"    {var_name}.value_field = next((attr for attr in {var_name}_domain.attributes if attr.name == \"{escaped_value}\"), None)\n")
            f.write(f"    {var_name}.value_field_name = \"{escaped_value}\"\n")
        f.write("else:\n")
        f.write(f"    # Domain class '{escaped_domain}' not resolved for data source '{_escape_string(getattr(source, 'name', var_name))}'.\n")
        if field_names:
            f.write(f"    {var_name}.field_names = {repr(field_names)}\n")
        if label_name:
            f.write(f"    {var_name}.label_field_name = \"{_escape_string(label_name)}\"\n")
        if value_name:
            f.write(f"    {var_name}.value_field_name = \"{_escape_string(value_name)}\"\n")
    else:
        if field_names:
            f.write(f"{var_name}.field_names = {repr(field_names)}\n")
        if label_name:
            f.write(f"{var_name}.label_field_name = \"{_escape_string(label_name)}\"\n")
        if value_name:
            f.write(f"{var_name}.value_field_name = \"{_escape_string(value_name)}\"\n")

def _write_layout(f, layout, created_vars, layout_var):
    """Write code for Layout object."""
    if layout_var in created_vars:
        return layout_var
    created_vars.add(layout_var)
    
    params = []
    if hasattr(layout, 'layout_type') and layout.layout_type:
        params.append(f'layout_type=LayoutType.{layout.layout_type.name}')
    if hasattr(layout, 'flex_direction') and layout.flex_direction:
        params.append(f'flex_direction="{layout.flex_direction}"')
    if hasattr(layout, 'justify_content') and layout.justify_content:
        params.append(f'justify_content="{layout.justify_content}"')
    if hasattr(layout, 'align_items') and layout.align_items:
        params.append(f'align_items="{layout.align_items}"')
    if hasattr(layout, 'flex_wrap') and layout.flex_wrap:
        params.append(f'flex_wrap="{layout.flex_wrap}"')
    if hasattr(layout, 'grid_template_columns') and layout.grid_template_columns:
        params.append(f'grid_template_columns="{layout.grid_template_columns}"')
    if hasattr(layout, 'grid_template_rows') and layout.grid_template_rows:
        params.append(f'grid_template_rows="{layout.grid_template_rows}"')
    if hasattr(layout, 'grid_gap') and layout.grid_gap:
        params.append(f'grid_gap="{layout.grid_gap}"')
    if hasattr(layout, 'gap') and layout.gap:
        params.append(f'gap="{layout.gap}"')
    
    if params:
        f.write(f'{layout_var} = Layout({", ".join(params)})\n')
    else:
        f.write(f'{layout_var} = Layout()\n')
    
    return layout_var
