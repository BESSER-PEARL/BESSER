"""
Constants and mappings for GUI diagram processing.
"""

from besser.BUML.metamodel.gui import (
    ButtonActionType,
    InputFieldType,
)

# HTML tags that represent text content
TEXT_TAGS = {
    "p", "span", "label", "strong", "em", "small",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "strike", "u", "s", "del", "ins", "mark",
    "code", "kbd", "samp", "var", "sub", "sup",
    "abbr", "cite", "q", "dfn", "time", "b", "i",
}

# HTML tags that represent containers
CONTAINER_TAGS = {
    "div", "section", "main", "header", "footer",
    "nav", "article", "aside", "ul", "ol", "li",
    "form", "tbody", "thead", "tr",
}

# GrapesJS component types that represent containers
CONTAINER_TYPES = {
    "container", "default", "row", "column", "wrapper",
    "grid", "list", "flex", "tabs", "tab-container",
    "tab-contents", "tab-content", "tab", "form", "cell"
}

# Input component types
INPUT_COMPONENT_TYPES = {
    "input", "textarea", "select", "checkbox", "radio"
}

# Mapping HTML input types to ButtonActionType
BUTTON_ACTION_BY_HTML_TYPE = {
    "submit": ButtonActionType.SubmitForm,
    "reset": ButtonActionType.Cancel,
    "button": ButtonActionType.View,
}

# Mapping HTML input types to InputFieldType
INPUT_TYPE_MAP = {
    "text": InputFieldType.Text,
    "textarea": InputFieldType.Text,
    "email": InputFieldType.Email,
    "number": InputFieldType.Number,
    "password": InputFieldType.Password,
    "date": InputFieldType.Date,
    "time": InputFieldType.Time,
    "color": InputFieldType.Color,
    "range": InputFieldType.Range,
    "url": InputFieldType.URL,
    "tel": InputFieldType.Tel,
    "search": InputFieldType.Search,
    "checkbox": InputFieldType.Text,
    "radio": InputFieldType.Text,
    "select": InputFieldType.Text,
}
