"""
Basic component parsers for GUI elements (Button, Text, Image, InputField, etc.).
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

from besser.BUML.metamodel.gui import (
    Alert,
    AlertSeverity,
    Button,
    ButtonActionType,
    ButtonType,
    DataList,
    DataSourceElement,
    EmbeddedContent,
    Form,
    Image,
    InputField,
    InputFieldType,
    Link,
    Menu,
    MenuItem,
    SelectOption,
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
)
from .constants import ALERT_SEVERITY_MAP, INPUT_TYPE_MAP, BUTTON_ACTION_BY_HTML_TYPE
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
    entity_class = None  # For CRUD operations (Create/Update/Delete)
    method_class = None  # For Run Method action
    method_name = None
    instance_source = None  # Table component ID providing instance data
    confirmation_required = False
    confirmation_message = None
    is_instance_method = False

    # Check both component level and attributes (GrapesJS stores target-screen at component level)
    target_screen_id = component.get("target-screen") or component.get("data-target-screen")
    action_button_type = component.get("action-type") or component.get("data-action-type")

    # New naming convention (method-class, entity-class, instance-source)
    entity_class = component.get("entity-class") or component.get("data-entity-class")
    method_class = component.get("method-class") or component.get("data-method-class")
    instance_source = component.get("instance-source") or component.get("data-instance-source")

    # Legacy naming support (crud-entity, method-entity, method-entity-id)
    entity_class = entity_class or component.get("crud-entity") or component.get("data-crud-entity")
    method_class = method_class or component.get("method-entity") or component.get("data-method-entity")
    instance_source = instance_source or component.get("method-entity-id") or component.get("data-method-entity-id")

    # Other button attributes (note: 'method' now stores method ID, not method name)
    method_name = component.get("method") or component.get("data-method") or component.get("method-name") or component.get("data-method-name")
    confirmation_required_str = component.get("confirmation-required") or component.get("data-confirmation-required")
    confirmation_message = component.get("confirmation-message") or component.get("data-confirmation-message")
    is_instance_str = component.get("instance-method") or component.get("data-instance-method")

    logger.debug("is_instance_str from component level = %s", is_instance_str)

    # Parse confirmation_required flag
    if confirmation_required_str:
        confirmation_required = confirmation_required_str.lower() in ('true', '1', 'yes')

    # Parse instance_method flag
    if is_instance_str:
        is_instance_method = is_instance_str.lower() in ('true', '1', 'yes')

    # Infer instance method from run-method + instance-source even when the explicit flag is missing.
    # This is common for BAL methods where instance semantics are represented with `this` instead of `self`.
    is_run_method = str(action_button_type or "").lower() == "run-method"
    if not is_instance_method and is_run_method and instance_source:
        is_instance_method = True

    logger.debug("is_instance_method parsed = %s", is_instance_method)

    # Also check attributes object as fallback
    if isinstance(attributes, dict):
        target_screen_id = target_screen_id or attributes.get("target-screen") or attributes.get("data-target-screen")
        action_button_type = action_button_type or attributes.get("action-type") or attributes.get("data-action-type")

        # New naming convention
        entity_class = entity_class or attributes.get("entity-class") or attributes.get("data-entity-class")
        method_class = method_class or attributes.get("method-class") or attributes.get("data-method-class")
        instance_source = instance_source or attributes.get("instance-source") or attributes.get("data-instance-source")
        is_instance_str = is_instance_str or attributes.get("instance-method") or attributes.get("data-instance-method")

        logger.debug("is_instance_str from attributes = %s", is_instance_str)

        # Re-parse instance_method flag if found in attributes
        if is_instance_str and not is_instance_method:
            is_instance_method = is_instance_str.lower() in ('true', '1', 'yes')
            logger.debug("is_instance_method re-parsed from attributes = %s", is_instance_method)

        # Legacy naming support
        entity_class = entity_class or attributes.get("crud-entity") or attributes.get("data-crud-entity")
        method_class = method_class or attributes.get("method-entity") or attributes.get("data-method-entity")
        instance_source = instance_source or attributes.get("method-entity-id") or attributes.get("data-method-entity-id")

        method_name = method_name or attributes.get("method") or attributes.get("data-method") or attributes.get("method-name") or attributes.get("data-method-name")
        confirmation_message = confirmation_message or attributes.get("confirmation-message") or attributes.get("data-confirmation-message")

        if not confirmation_required_str:
            confirmation_required_str = attributes.get("confirmation-required") or attributes.get("data-confirmation-required")
            if confirmation_required_str:
                confirmation_required = confirmation_required_str.lower() in ('true', '1', 'yes')

        if not is_instance_str:
            is_instance_str = attributes.get("instance-method") or attributes.get("data-instance-method")
            if is_instance_str:
                is_instance_method = is_instance_str.lower() in ('true', '1', 'yes')

        # Re-apply inference after attribute-level fallbacks.
        is_run_method = str(action_button_type or "").lower() == "run-method"
        if not is_instance_method and is_run_method and instance_source:
            is_instance_method = True

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

        # Map action-type to ButtonActionType enum
        action_type_map = {
            "navigate": ButtonActionType.Navigate,
            "run-method": ButtonActionType.RunMethod,
            "create": ButtonActionType.Create,
            "update": ButtonActionType.Update,
            "delete": ButtonActionType.Delete,
        }

        if action_button_type == "navigate" and target_screen_id:
            # Create Transition action for navigation
            action_obj = Transition(
                name=f"navigate_to_{target_screen_id}",
                description=f"Navigate to {target_screen_id}",
                target_screen=None,  # Will be resolved later
                parameters=parameters if parameters else None
            )
            setattr(action_obj, '_target_screen_id', target_screen_id)

        elif action_button_type == "read" and entity_class:
            # Create Read action for fetching/displaying data
            action_obj = Read(
                name=f"read_{entity_class}",
                description=f"Read/Load {entity_class}",
                target_class=None,  # Will be resolved later
                parameters=parameters if parameters else None
            )
            setattr(action_obj, '_target_class_name', entity_class)

        elif action_button_type == "create" and entity_class:
            # Create Create action for adding new entity
            action_obj = Create(
                name=f"create_{entity_class}",
                description=f"Create new {entity_class}",
                target_class=None,  # Will be resolved later
                parameters=parameters if parameters else None
            )
            setattr(action_obj, '_target_class_name', entity_class)

        elif action_button_type == "update" and entity_class:
            action_obj = Update(
                name=f"update_{entity_class}",
                description=f"Update {entity_class}",
                target_class=None,  # Will be resolved later
                parameters=parameters if parameters else None
            )
            setattr(action_obj, '_target_class_name', entity_class)

        elif action_button_type == "delete" and entity_class:
            action_obj = Delete(
                name=f"delete_{entity_class}",
                description=f"Delete {entity_class}",
                target_class=None,  # Will be resolved later
                parameters=parameters if parameters else None
            )
            setattr(action_obj, '_target_class_name', entity_class)

        # Create onClick Event if we have an action
        if action_obj:
            events.add(Event(
                name=f"onClick_{name}",
                event_type=EventType.OnClick,
                actions={action_obj}
            ))

    # Determine button action type from action-type attribute or HTML type
    action_type = ButtonActionType.Navigate  # Default

    if action_button_type:
        # Map action-type string to ButtonActionType enum
        action_type_map = {
            "navigate": ButtonActionType.Navigate,
            "run-method": ButtonActionType.RunMethod,
            "create": ButtonActionType.Create,
            "update": ButtonActionType.Update,
            "delete": ButtonActionType.Delete,
        }
        action_type = action_type_map.get(action_button_type, ButtonActionType.Navigate)
    else:
        # Fallback to HTML type mapping
        html_type = ""
        if isinstance(attributes, dict):
            html_type = str(attributes.get("type", "")).lower()
        action_type = BUTTON_ACTION_BY_HTML_TYPE.get(html_type, ButtonActionType.Navigate)

    # Create Button with method execution configuration
    button = Button(
        name=name,
        description="Button component",
        label=label,
        buttonType=ButtonType.CustomizableButton,
        actionType=action_type,
        styling=styling,
        method_btn=None,  # Will be resolved later in processor
        entity_class=None,  # Will be resolved later in processor
        instance_source=instance_source,
        is_instance_method=is_instance_method,
        confirmation_required=confirmation_required,
        confirmation_message=confirmation_message,
    )

    # Store IDs for later resolution (class ID + method ID)
    if method_class:
        setattr(button, '_method_class_id', method_class)
    if method_name:
        setattr(button, '_method_id', method_name)
    if entity_class:
        setattr(button, '_entity_class_id', entity_class)

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

    Supports both plain HTML inputs (type="text", type="range", …) and the
    BESSER GUI editor's custom input blocks, which store the target
    InputFieldType in a ``data-gui-type`` attribute (e.g. "Slider", "Toggle").

    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata

    Returns:
        InputField instance with validation rules and optional SelectOptions
    """
    attributes = component.get("attributes") if isinstance(component.get("attributes"), dict) else {}
    tag = str(component.get("tagName", "")).lower()

    field_type = InputFieldType.Text
    validation_rules = []
    options: list[SelectOption] = []
    min_value = None
    max_value = None
    step_value = None
    label = None
    placeholder = None
    required = False
    default_value = None

    if isinstance(attributes, dict):
        # data-gui-type takes precedence – set by the BESSER GUI editor palette
        gui_type = attributes.get("data-gui-type") or component.get("data-gui-type")
        if gui_type and gui_type in INPUT_TYPE_MAP:
            field_type = INPUT_TYPE_MAP[gui_type]
        else:
            html_type = str(attributes.get("type", tag) or "").lower()
            field_type = INPUT_TYPE_MAP.get(html_type, InputFieldType.Text)

        # Label – data-label (from storeAttr) takes precedence, then top-level prop
        label = (
            attributes.get("data-label")
            or component.get("input-label")
        ) or None

        # Placeholder
        placeholder = (
            attributes.get("data-placeholder")
            or component.get("placeholder")
        ) or None

        # Required
        req_raw = (
            attributes.get("data-required")
            or component.get("required")
        )
        if req_raw:
            required = str(req_raw).lower() in ("true", "1", "yes")
        elif attributes.get("required"):
            required = True

        # Default value (Toggle's "Default On" / "default-checked")
        default_raw = (
            attributes.get("data-default-checked")
            or component.get("default-checked")
        )
        if default_raw is not None:
            default_value = str(default_raw).lower() in ("true", "1", "yes")

        # Validation rules
        if required:
            validation_rules.append("required")
        if attributes.get("pattern"):
            validation_rules.append(f"pattern:{attributes.get('pattern')}")

        # Numeric / range constraints – prefer data-* attrs, fallback to top-level props
        def _try_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _first_defined(*keys_and_sources):
            """Return first non-None value from (attr_key, dict) pairs."""
            for key, source in keys_and_sources:
                val = source.get(key)
                if val is not None:
                    return val
            return None

        # min
        min_raw = _first_defined(
            ("data-min", attributes), ("min", attributes), ("input-min", component)
        )
        if min_raw and min_value is None:
            min_value = _try_float(min_raw)
            if min_value is not None:
                validation_rules.append(f"min:{min_raw}")

        # max
        max_raw = _first_defined(
            ("data-max", attributes), ("max", attributes), ("input-max", component)
        )
        if max_raw and max_value is None:
            max_value = _try_float(max_raw)
            if max_value is not None:
                validation_rules.append(f"max:{max_raw}")

        # step
        step_raw = _first_defined(
            ("data-step", attributes), ("step", attributes), ("input-step", component)
        )
        if step_raw and step_value is None:
            step_value = _try_float(step_raw)

        # maxlength – for TextArea and text-like inputs
        maxlen_raw = (
            attributes.get("data-max-length")
            or attributes.get("maxlength")
            or component.get("max-length")
        )
        if maxlen_raw is not None:
            maxlen_v = _try_float(maxlen_raw)
            if maxlen_v is not None:
                validation_rules.append(f"maxlength:{int(maxlen_v)}")

        # minlength
        minlen_raw = attributes.get("minlength")
        if minlen_raw is not None:
            minlen_v = _try_float(minlen_raw)
            if minlen_v is not None:
                validation_rules.append(f"minlength:{int(minlen_v)}")

        # max_value for Rating (data-max-stars / max-stars prop)
        if field_type == InputFieldType.Rating and max_value is None:
            stars_raw = attributes.get("data-max-stars") or component.get("max-stars")
            if stars_raw is not None:
                max_value = _try_float(stars_raw)

        # multiple – File / MultiSelect
        multiple_raw = (
            attributes.get("data-multiple")
            or component.get("multiple-files")
        )
        multiple = str(multiple_raw).lower() in ("true", "1", "yes") if multiple_raw is not None else False

        # Select options for Dropdown / RadioGroup / CheckboxGroup / MultiSelect
        raw_options = (
            attributes.get("data-options")
            or component.get("select-options")
            or component.get("data-options")
            or ""
        )
        if raw_options and isinstance(raw_options, str):
            for opt_label in raw_options.split(","):
                opt_label = opt_label.strip()
                if opt_label:
                    options.append(SelectOption(value=opt_label, label=opt_label))

    else:
        multiple = False

    description = f"{field_type.value} input"
    input_field = InputField(
        name=name,
        description=description,
        field_type=field_type,
        styling=styling,
        label=label,
        placeholder=placeholder,
        required=required,
        default_value=str(default_value) if default_value is not None else None,
        validationRules=",".join(validation_rules) if validation_rules else None,
        options=options if options else None,
        min_value=min_value,
        max_value=max_value,
        step=step_value,
        multiple=multiple,
    )

    if meta["tagName"] is None:
        meta["tagName"] = "input"

    _attach_component_metadata(input_field, component, meta)

    return input_field


def parse_alert(component: Dict[str, Any], styling, name: str, meta: Dict) -> Alert:
    """
    Parse an Alert component (gui-alert block from the BESSER GUI editor).

    Reads severity, title, content and dismissible from the component's
    ``data-*`` attributes and creates an Alert BUML object.

    Args:
        component: GrapesJS component dict
        styling: Resolved Styling object
        name: Component name
        meta: Frontend metadata

    Returns:
        Alert instance
    """
    attributes = component.get("attributes") if isinstance(component.get("attributes"), dict) else {}

    severity_raw = None
    title = None
    content = ""
    dismissible = False

    if isinstance(attributes, dict):
        severity_raw = (
            attributes.get("data-severity")
            or attributes.get("alert-severity")
            or component.get("alert-severity")
        )
        title = (
            attributes.get("data-title")
            or attributes.get("alert-title")
            or component.get("alert-title")
        ) or None
        content = (
            attributes.get("data-content")
            or attributes.get("alert-content")
            or component.get("alert-content")
            or extract_text_content(component)
            or ""
        )
        dismissible_raw = (
            attributes.get("data-dismissible")
            or attributes.get("alert-dismissible")
            or component.get("alert-dismissible")
        )
        if dismissible_raw:
            dismissible = str(dismissible_raw).lower() in ("true", "1", "yes")

    severity = ALERT_SEVERITY_MAP.get(severity_raw or "Info", AlertSeverity.Info)

    alert = Alert(
        name=name,
        description=f"Alert component ({severity.value})",
        content=content,
        severity=severity,
        title=title,
        dismissible=dismissible,
        styling=styling,
    )

    if meta["tagName"] is None:
        meta["tagName"] = "div"

    _attach_component_metadata(alert, component, meta)
    return alert


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
                    event_type=EventType.OnSubmit,
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
