"""
Basic component parsers for GUI elements (Button, Text, Image, InputField, etc.).
"""

from typing import Any, Dict, List

from besser.BUML.metamodel.gui import (
    Button,
    ButtonActionType,
    ButtonType,
    DataList,
    DataSource,
    DataSourceElement,
    EmbeddedContent,
    Form,
    Image,
    InputField,
    InputFieldType,
    Link,
    Menu,
    MenuItem,
    Text,
    ViewComponent,
    ViewContainer,
)
from besser.BUML.metamodel.gui.events_actions import (
    Create,
    Delete,
    Event,
    EventType,
    Read,
    Transition,
    Update,
)

from .component_helpers import (
    collect_input_fields_recursive,
    extract_menu_items,
    extract_parameters_from_attributes,
    has_data_binding,
    has_menu_structure,
)
from .constants import INPUT_COMPONENT_TYPES, INPUT_TYPE_MAP, BUTTON_ACTION_BY_HTML_TYPE
from .utils import extract_text_content, clean_attribute_name


def _attach_component_metadata(element, component: Dict[str, Any], meta: Dict[str, Any] = None) -> None:
    """
    PRIMARY metadata attachment function for component parsers.
    
    This function extracts and attaches metadata directly from the GrapesJS JSON component dict.
    It should be called by ALL specialized component parsers (parse_button, parse_text, etc.)
    to ensure metadata is set during component creation.
    
    Design Pattern:
    1. This function sets metadata from JSON FIRST (during parsing)
    2. attach_meta() in processor.py is called AFTER as a fallback
    3. attach_meta() will NOT overwrite values set by this function
    
    Metadata attached:
    - component_id: Unique identifier from JSON attributes or component level
    - component_type: GrapesJS component type (e.g., "text", "button", "image")
    - tag_name: HTML tag name (e.g., "p", "h1", "div", "button")
    - css_classes: List of CSS class names
    - custom_attributes: Dictionary of HTML attributes from JSON
    
    Args:
        element: BUML ViewElement/ViewComponent/ViewContainer to attach metadata to
        component: GrapesJS JSON component dictionary
        meta: Optional metadata dict with additional context
    """
    if not element:
        return
    
    attributes = component.get("attributes", {})
    if isinstance(attributes, dict):
        element.component_id = attributes.get("id") or component.get("id")
    else:
        element.component_id = component.get("id")
    
    element.component_type = component.get("type")
    element.tag_name = component.get("tagName") or (meta.get("tagName") if meta else None)
    element.css_classes = [cls if isinstance(cls, str) else cls.get("name", "") for cls in (component.get("classes") or [])]
    element.custom_attributes = dict(attributes) if isinstance(attributes, dict) else {}


def parse_button(component: Dict[str, Any], styling, name: str, meta: Dict) -> Button:
    """
    Parse a button component with events and actions.
    
    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata
        
    Returns:
        Button instance with configured events and actions
    """
    attributes = component.get("attributes") if isinstance(component.get("attributes"), dict) else {}
    
    # Extract label from multiple sources
    label = None
    if isinstance(attributes, dict):
        label = attributes.get("button-label") or attributes.get("label")
    if not label:
        label = extract_text_content(component)
    if not label:
        label = component.get("content", "")
    
    # Extract action configuration
    target_screen_id = None
    action_button_type = None
    crud_entity = None
    
    # Check both component level and attributes (GrapesJS stores target-screen at component level)
    target_screen_id = component.get("target-screen") or component.get("data-target-screen")
    action_button_type = component.get("action-type") or component.get("data-action-type")
    crud_entity = component.get("crud-entity") or component.get("data-crud-entity")
    
    # Also check attributes object as fallback
    if isinstance(attributes, dict):
        target_screen_id = target_screen_id or attributes.get("target-screen") or attributes.get("data-target-screen")
        action_button_type = action_button_type or attributes.get("action-type") or attributes.get("data-action-type")
        crud_entity = crud_entity or attributes.get("crud-entity") or attributes.get("data-crud-entity")
        
        # Generate default label based on action
        if not label:
            if target_screen_id:
                label = f"Go to {target_screen_id}"
            elif action_button_type:
                label = action_button_type.capitalize()
            else:
                label = "Button"
    
    if not label:
        label = "Button"
    
    # Create Event with appropriate Action
    events = set()
    
    if action_button_type or target_screen_id:
        # Extract parameters from data-param-* attributes
        parameters = extract_parameters_from_attributes(attributes)
        
        # Determine action type - default to navigate if target-screen is set
        if not action_button_type and target_screen_id:
            action_button_type = "navigate"
        
        action_obj = None
        
        if action_button_type == "navigate" and target_screen_id:
            # Create Transition action for navigation
            action_obj = Transition(
                name=f"navigate_to_{target_screen_id}",
                description=f"Navigate to {target_screen_id}",
                target_screen=None,  # Will be resolved later
                parameters=parameters if parameters else None
            )
            setattr(action_obj, '_target_screen_id', target_screen_id)
        
        elif action_button_type == "read" and crud_entity:
            # Create Read action for fetching/displaying data
            action_obj = Read(
                name=f"read_{crud_entity}",
                description=f"Read/Load {crud_entity}",
                target_class=None,  # Will be resolved later
                parameters=parameters if parameters else None
            )
            setattr(action_obj, '_target_class_name', crud_entity)
            
        elif action_button_type == "create" and crud_entity:
            # Create Create action for adding new entity
            action_obj = Create(
                name=f"create_{crud_entity}",
                description=f"Create new {crud_entity}",
                target_class=None,  # Will be resolved later
                parameters=parameters if parameters else None
            )
            setattr(action_obj, '_target_class_name', crud_entity)
            
        elif action_button_type == "update" and crud_entity:
            action_obj = Update(
                name=f"update_{crud_entity}",
                description=f"Update {crud_entity}",
                target_class=None,  # Will be resolved later
                parameters=parameters if parameters else None
            )
            setattr(action_obj, '_target_class_name', crud_entity)
            
        elif action_button_type == "delete" and crud_entity:
            action_obj = Delete(
                name=f"delete_{crud_entity}",
                description=f"Delete {crud_entity}",
                target_class=None,  # Will be resolved later
                parameters=parameters if parameters else None
            )
            setattr(action_obj, '_target_class_name', crud_entity)
        
        # Create onClick Event if we have an action
        if action_obj:
            events.add(Event(
                name=f"onClick_{name}",
                event_type=EventType.OnClick,
                actions={action_obj}
            ))
    
    # Determine button action type from HTML type
    html_type = ""
    if isinstance(attributes, dict):
        html_type = str(attributes.get("type", "")).lower()
    action_type = BUTTON_ACTION_BY_HTML_TYPE.get(html_type, ButtonActionType.Send)
    
    button = Button(
        name=name,
        description="Button component",
        label=label,
        buttonType=ButtonType.CustomizableButton,
        actionType=action_type,
        styling=styling,
    )
    
    # Set events on button and triggered_by on actions
    if events:
        button.events = events
        # Set triggered_by bidirectional link
        for event in events:
            for action in event.actions:
                action.triggered_by = button
    
    if meta["tagName"] is None:
        meta["tagName"] = "button"
    
    # Attach GrapesJS metadata for code generation
    _attach_component_metadata(button, component, meta)
    
    return button


def parse_input_field(component: Dict[str, Any], styling, name: str, meta: Dict) -> InputField:
    """
    Parse an input field component with validation rules.
    
    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata
        
    Returns:
        InputField instance with validation rules
    """
    attributes = component.get("attributes") if isinstance(component.get("attributes"), dict) else {}
    comp_type = str(component.get("type", "")).lower()
    tag = str(component.get("tagName", "")).lower()
    
    field_type = InputFieldType.Text
    validation_rules = []
    
    if isinstance(attributes, dict):
        html_type = str(attributes.get("type", tag) or "").lower()
        field_type = INPUT_TYPE_MAP.get(html_type, InputFieldType.Text)
        
        # Enhanced validation rules extraction
        if attributes.get("required"):
            validation_rules.append("required")
        if attributes.get("pattern"):
            validation_rules.append(f"pattern:{attributes.get('pattern')}")
        if attributes.get("min"):
            validation_rules.append(f"min:{attributes.get('min')}")
        if attributes.get("max"):
            validation_rules.append(f"max:{attributes.get('max')}")
        if attributes.get("minlength"):
            validation_rules.append(f"minlength:{attributes.get('minlength')}")
        if attributes.get("maxlength"):
            validation_rules.append(f"maxlength:{attributes.get('maxlength')}")
        if attributes.get("step"):
            validation_rules.append(f"step:{attributes.get('step')}")
    
    description = f"{field_type.value} input"
    input_field = InputField(
        name=name,
        description=description,
        field_type=field_type,
        styling=styling,
        validationRules=",".join(validation_rules) if validation_rules else None,
    )
    
    if meta["tagName"] is None:
        meta["tagName"] = "input"
    
    _attach_component_metadata(input_field, component, meta)
    
    return input_field


def parse_form(component: Dict[str, Any], styling, name: str, meta: Dict, parse_component_list_func) -> Form:
    """
    Parse a form component with input fields and submit event.
    
    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata
        parse_component_list_func: Function to parse child components
        
    Returns:
        Form instance with input fields and events
    """
    # Extract all InputField children from form
    components = component.get("components", []) or []
    parsed_children = parse_component_list_func(components)
    input_fields = collect_input_fields_recursive(parsed_children)
    
    form = Form(
        name=name,
        description="Form component",
        inputFields=input_fields,
        styling=styling,
    )
    
    if meta["tagName"] is None:
        meta["tagName"] = "form"
    
    # Create OnSubmit event if form has submit button
    submit_button = None
    for child in parsed_children:
        if isinstance(child, Button) and child.actionType == ButtonActionType.SubmitForm:
            submit_button = child
            break
    
    if submit_button and hasattr(submit_button, 'events'):
        # Add OnSubmit event to form (forms can have events too as they extend ViewComponent)
        for event in submit_button.events:
            if event.type == EventType.OnClick:
                # Create OnSubmit event mirroring the button's actions
                form.events = form.events or set()
                form.events.add(Event(
                    name=f"onSubmit_{name}",
                    type=EventType.OnSubmit,
                    actions=event.actions
                ))
    
    _attach_component_metadata(form, component, meta)
    return form


def parse_link(component: Dict[str, Any], styling, name: str, meta: Dict) -> Link:
    """
    Parse a hyperlink component.
    """
    attributes = component.get("attributes") if isinstance(component.get("attributes"), dict) else {}
    label = extract_text_content(component) or ""
    if isinstance(attributes, dict) and not label:
        label = attributes.get("title") or attributes.get("aria-label") or attributes.get("data-label") or ""

    url = attributes.get("href") if isinstance(attributes, dict) else None
    if isinstance(attributes, dict):
        url = url or attributes.get("data-href")
        target = attributes.get("target") or attributes.get("data-target")
        rel = attributes.get("rel")
        description = attributes.get("title") or "Link element"
    else:
        target = None
        rel = None
        description = "Link element"

    link = Link(
        name=name,
        description=description,
        label=label,
        url=url,
        target=target,
        rel=rel,
        styling=styling,
    )

    if meta["tagName"] is None:
        meta["tagName"] = "a"

    _attach_component_metadata(link, component, meta)
    return link


def parse_embedded_content(component: Dict[str, Any], styling, name: str, meta: Dict) -> EmbeddedContent:
    """
    Parse generic embedded content (e.g., maps/iframes).
    """
    attributes = component.get("attributes") if isinstance(component.get("attributes"), dict) else {}
    source = None
    if isinstance(attributes, dict):
        source = attributes.get("src") or attributes.get("data-src")
    source = source or component.get("src")

    description = "Embedded content"
    if isinstance(attributes, dict):
        description = attributes.get("title") or attributes.get("aria-label") or description
    content_type = component.get("type") or component.get("tagName") or (attributes.get("data-content-type") if isinstance(attributes, dict) else None)

    embedded = EmbeddedContent(
        name=name,
        description=description,
        source=source,
        content_type=content_type,
        styling=styling,
    )

    if meta["tagName"] is None:
        meta["tagName"] = "iframe"

    _attach_component_metadata(embedded, component, meta)
    return embedded


def parse_text(component: Dict[str, Any], styling, name: str, meta: Dict) -> Text:
    """
    Parse a text component.
    
    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata
        
    Returns:
        Text instance
    """
    content = extract_text_content(component)
    text_element = Text(
        name=name,
        content=content or "",
        description="Text element",
        styling=styling,
    )
    _attach_component_metadata(text_element, component, meta)
    return text_element


def parse_image(component: Dict[str, Any], styling, name: str, meta: Dict) -> Image:
    """
    Parse an image component.
    
    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata
        
    Returns:
        Image instance
    """
    attributes = component.get("attributes") if isinstance(component.get("attributes"), dict) else {}
    description = "Image component"
    if isinstance(attributes, dict) and attributes.get("alt"):
        description = str(attributes.get("alt"))
    
    source = None
    if isinstance(attributes, dict):
        source = attributes.get("src") or attributes.get("data-src")
    if not source:
        source = component.get("src")

    image = Image(name=name, description=description, styling=styling, source=source)
    if meta["tagName"] is None:
        meta["tagName"] = "img"
    _attach_component_metadata(image, component, meta)
    return image


def parse_menu(component: Dict[str, Any], styling, name: str, meta: Dict) -> Menu:
    """
    Parse a menu/navigation component.
    
    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata
        
    Returns:
        Menu instance with menu items
    """
    menu_items = set()
    
    # Extract menu items from links or list items
    for menu_item_data in extract_menu_items(component):
        menu_items.add(
            MenuItem(
                label=menu_item_data.get('label', "Menu"),
                url=menu_item_data.get('url'),
                target=menu_item_data.get('target'),
                rel=menu_item_data.get('rel'),
            )
        )
    
    menu = Menu(
        name=name,
        description="Menu component",
        menuItems=menu_items,
        styling=styling,
    )
    
    if meta["tagName"] is None:
        meta["tagName"] = "nav"
    
    _attach_component_metadata(menu, component, meta)
    return menu


def parse_data_list(component: Dict[str, Any], styling, name: str, meta: Dict, domain_model) -> DataList:
    """
    Parse a data list component with data binding.
    
    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata
        domain_model: Domain model for class resolution
        
    Returns:
        DataList instance with data sources
    """
    attributes = component.get("attributes") if isinstance(component.get("attributes"), dict) else {}
    tag = str(component.get("tagName", "")).lower()
    
    # Extract data sources from attributes
    list_sources = set()
    raw_fields = []
    if isinstance(attributes, dict):
        data_source_name = attributes.get("data-source") or attributes.get("data-bind")
        label_field_name = attributes.get("label-field") or attributes.get("data-label-field")
        value_field_name = attributes.get("data-field") or attributes.get("value-field")
        raw_fields_value = attributes.get("fields") or attributes.get("data-fields")
        if isinstance(raw_fields_value, str):
            raw_fields = [clean_attribute_name(part.strip()) for part in raw_fields_value.split(",") if part.strip()]
        elif isinstance(raw_fields_value, list):
            raw_fields = [clean_attribute_name(str(part)) for part in raw_fields_value]
        else:
            raw_fields = []

        if data_source_name and domain_model:
            domain_class = domain_model.get_class_by_name(data_source_name)
            if domain_class:
                label_field = None
                value_field = None
                if label_field_name:
                    label_field = next(
                        (attr for attr in getattr(domain_class, "attributes", set()) if attr.name == clean_attribute_name(label_field_name)),
                        None,
                    )
                if value_field_name:
                    value_field = next(
                        (attr for attr in getattr(domain_class, "attributes", set()) if attr.name == clean_attribute_name(value_field_name)),
                        None,
                    )

                fields = set()
                if raw_fields:
                    fields = {
                        attr
                        for attr in getattr(domain_class, "attributes", set())
                        if attr.name in {clean_attribute_name(name) for name in raw_fields}
                    }

                source = DataSourceElement(
                    name=data_source_name,
                    dataSourceClass=domain_class,
                    fields=fields,
                    label_field=label_field,
                    value_field=value_field,
                    field_names=raw_fields,
                    label_field_name=clean_attribute_name(label_field_name) if label_field_name else None,
                    value_field_name=clean_attribute_name(value_field_name) if value_field_name else None,
                )
                list_sources.add(source)
        elif data_source_name:
            source = DataSourceElement(
                name=data_source_name,
                dataSourceClass=None,
                fields=set(),
                field_names=raw_fields,
                label_field_name=clean_attribute_name(label_field_name) if label_field_name else None,
                value_field_name=clean_attribute_name(value_field_name) if value_field_name else None,
            )
            list_sources.add(source)
    
    data_list = DataList(
        name=name,
        description="Data list component",
        list_sources=list_sources,
        styling=styling,
    )
    
    if meta["tagName"] is None:
        meta["tagName"] = tag or "ul"
    
    _attach_component_metadata(data_list, component, meta)
    return data_list


def parse_container(component: Dict[str, Any], styling, name: str, meta: Dict, parse_component_list_func) -> ViewContainer:
    """
    Parse a container component with children.
    
    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata
        parse_component_list_func: Function to parse child components
        
    Returns:
        ViewContainer instance with children
    """
    comp_type = str(component.get("type", "")).lower()
    tag = str(component.get("tagName", "")).lower()
    
    container = ViewContainer(
        name=name,
        description=f"{tag or comp_type} container",
        view_elements=set(),
        styling=styling,
    )
    
    children = parse_component_list_func(component.get("components"))
    container.view_elements = set(children)
    
    # Set layout if present in styling
    if styling and hasattr(styling, 'layout') and styling.layout:
        container.layout = styling.layout
    
    _attach_component_metadata(container, component, meta)
    return container


def parse_generic_component(component: Dict[str, Any], styling, name: str, meta: Dict, parse_component_list_func=None) -> ViewComponent:
    """
    Parse a generic/unknown component.
    If the component has children, creates a ViewContainer instead of ViewComponent.
    
    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata
        parse_component_list_func: Optional function to parse child components
        
    Returns:
        ViewComponent or ViewContainer instance
    """
    comp_type = str(component.get("type", "")).lower()
    tag = str(component.get("tagName", "")).lower()
    
    # Check if component has children
    components = component.get("components", [])
    has_children = isinstance(components, list) and len(components) > 0
    
    # If it has children, create a ViewContainer instead
    if has_children and parse_component_list_func:
        container = ViewContainer(
            name=name,
            description=f"{tag or comp_type} component",
            view_elements=set(),
            styling=styling,
        )
        
        children = parse_component_list_func(components)
        container.view_elements = set(children)
        
        # Set layout if present in styling
        if styling and hasattr(styling, 'layout') and styling.layout:
            container.layout = styling.layout
        
        _attach_component_metadata(container, component, meta)
        return container
    
    # Otherwise create a simple ViewComponent
    generic = ViewComponent(
        name=name,
        description=f"{tag or comp_type} component",
        styling=styling,
    )
    _attach_component_metadata(generic, component, meta)
    return generic
