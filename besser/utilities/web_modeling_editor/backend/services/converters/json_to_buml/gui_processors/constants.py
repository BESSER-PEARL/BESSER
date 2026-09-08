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
    # Standard HTML input types
    "text": InputFieldType.Text,
    "textarea": InputFieldType.TextArea,
    "email": InputFieldType.Email,
    "number": InputFieldType.Number,
    "password": InputFieldType.Password,
    "date": InputFieldType.Date,
    "time": InputFieldType.Time,
    "datetime-local": InputFieldType.DateTime,
    "color": InputFieldType.Color,
    "range": InputFieldType.Range,
    "url": InputFieldType.URL,
    "tel": InputFieldType.Tel,
    "search": InputFieldType.Search,
    "checkbox": InputFieldType.Checkbox,
    "radio": InputFieldType.RadioGroup,
    "select": InputFieldType.Dropdown,
    "file": InputFieldType.File,
    # BESSER data-gui-type values (set by the GUI editor input blocks)
    "Text": InputFieldType.Text,
    "TextArea": InputFieldType.TextArea,
    "RichText": InputFieldType.RichText,
    "Password": InputFieldType.Password,
    "Search": InputFieldType.Search,
    "Tags": InputFieldType.Tags,
    "OTP": InputFieldType.OTP,
    "Hidden": InputFieldType.Hidden,
    "Email": InputFieldType.Email,
    "URL": InputFieldType.URL,
    "Tel": InputFieldType.Tel,
    "Number": InputFieldType.Number,
    "Slider": InputFieldType.Slider,
    "Spinner": InputFieldType.Spinner,
    "Rating": InputFieldType.Rating,
    "Range": InputFieldType.Range,
    "Checkbox": InputFieldType.Checkbox,
    "Toggle": InputFieldType.Toggle,
    "Dropdown": InputFieldType.Dropdown,
    "RadioGroup": InputFieldType.RadioGroup,
    "CheckboxGroup": InputFieldType.CheckboxGroup,
    "MultiSelect": InputFieldType.MultiSelect,
    "Date": InputFieldType.Date,
    "Time": InputFieldType.Time,
    "DateTime": InputFieldType.DateTime,
    "DateRange": InputFieldType.DateRange,
    "File": InputFieldType.File,
    "ImageUpload": InputFieldType.ImageUpload,
    "Color": InputFieldType.Color,
}

# Mapping data-severity values to AlertSeverity
from besser.BUML.metamodel.gui import AlertSeverity  # noqa: E402 – local import to avoid circular

ALERT_SEVERITY_MAP = {
    "Info": AlertSeverity.Info,
    "Success": AlertSeverity.Success,
    "Warning": AlertSeverity.Warning,
    "Error": AlertSeverity.Error,
    # Lower-case variants for robustness
    "info": AlertSeverity.Info,
    "success": AlertSeverity.Success,
    "warning": AlertSeverity.Warning,
    "error": AlertSeverity.Error,
}
