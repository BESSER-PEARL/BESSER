"""
Common utilities for BUML code builders
"""

PRIMITIVE_TYPE_MAPPING = {
    'str': 'StringType',
    'string': 'StringType',
    'int': 'IntegerType',
    'integer': 'IntegerType',
    'float': 'FloatType',
    'bool': 'BooleanType',
    'boolean': 'BooleanType',
    'time': 'TimeType',
    'date': 'DateType',
    'datetime': 'DateTimeType',
    'timedelta': 'TimeDeltaType',
    'any': 'AnyType'
}

# Reserved names that need special handling
RESERVED_NAMES = ['Class', 'Property', 'Method', 'Parameter', 'Enumeration']


def safe_class_name(name):
    """
    Add a suffix to class names that match reserved keywords or BUML metaclass names.
    If the name already ends with an underscore and would conflict with a reserved name,
    add a numeric suffix instead.
    
    Parameters:
    name (str): The original class name
    
    Returns:
    str: A safe variable name for the class
    """
    if not name:
        return "unnamed_class"
        
    if name.endswith('_'):
        base_name = name[:-1]
        if base_name in RESERVED_NAMES:
            return f"{name}var"
        return name
    elif name in RESERVED_NAMES:
        return f"{name}_"
    else:
        return name


def safe_identifier(name: str, fallback: str) -> str:
    """Generate a lowercase, underscore-delimited identifier safe for Python variables."""

    sanitized = ''.join(ch.lower() if ch.isalnum() else '_' for ch in (name or ''))
    sanitized = sanitized.strip('_')
    if not sanitized:
        sanitized = fallback
    if sanitized[0].isdigit():
        sanitized = f"{fallback}_{sanitized}"
    return sanitized


def contains_user_class(model) -> bool:
    """Return True if the supplied model exposes a Class literally named 'User'."""
    if not model:
        return False
    get_classes = getattr(model, "get_classes", None)
    if not callable(get_classes):
        return False
    classes = get_classes() or []
    for cls in classes:
        class_name = (getattr(cls, "name", "") or "").strip().lower()
        if class_name == "user":
            return True
    return False


def is_user_object_model(obj_model) -> bool:
    """Detect whether an ObjectModel belongs to the user reference domain."""
    if not obj_model:
        return False
    domain_model = getattr(obj_model, "domain_model", None)
    if contains_user_class(domain_model):
        return True

    objects = getattr(obj_model, "objects", None) or []
    for obj in objects:
        classifier = getattr(obj, "classifier", None)
        classifier_name = (getattr(classifier, "name", "") or "").strip().lower()
        if classifier_name == "user":
            return True
    return False
