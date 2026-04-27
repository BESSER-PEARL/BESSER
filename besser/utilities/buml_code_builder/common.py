"""
Common utilities for BUML code builders
"""
import keyword


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


def safe_var_name(name: str, lowercase: bool = True) -> str:
    """
    Convert a name to a safe Python variable name.

    By default the result is lowercased — this is the historical behavior
    relied on by ``agent_model_builder`` and by ``WebAppGenerator``'s
    ``agent_slug`` (filesystem paths, container names, and hostnames are
    conventionally lowercase). Callers that need to preserve the user's
    original casing — e.g. emitting Python identifiers that must round-trip
    unchanged — can opt in via ``lowercase=False``.

    Args:
        name: Original name
        lowercase: If True (default), lowercase the result.

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
    safe_name = safe_name.strip('_') or "unnamed"
    if lowercase:
        safe_name = safe_name.lower()
    # Apply the keyword guard regardless of casing so e.g.
    # ``safe_var_name("Class", lowercase=False)`` is still escaped.
    if keyword.iskeyword(safe_name.lower()):
        safe_name = f"{safe_name}_"
    return safe_name


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

    if keyword.iskeyword(name):
        return f"{name}_"
    if name.endswith('_'):
        base_name = name[:-1]
        if base_name in RESERVED_NAMES:
            return f"{name}var"
        return name
    elif name in RESERVED_NAMES:
        return f"{name}_"
    else:
        return name
