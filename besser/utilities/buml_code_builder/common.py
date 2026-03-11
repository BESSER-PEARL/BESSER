"""
Common utilities for BUML code builders
"""


def _escape_python_string(value: str) -> str:
    """Escape a string for safe interpolation into generated Python source code.

    Prevents code injection when user-controlled values (names, labels, etc.)
    are embedded inside string literals in generated Python files that may
    later be executed with ``exec()``.
    """
    return (value
            .replace('\\', '\\\\')
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace('\n', '\\n')
            .replace('\r', '\\r'))


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
