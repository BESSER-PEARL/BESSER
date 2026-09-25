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


def _comment_safe(name: str) -> str:
    """Sanitise a name for safe embedding in a ``#`` comment line.

    Replaces newline / carriage-return characters with a space so that a
    user-supplied name cannot break out of the comment into executable code when
    the generated source is later processed with ``exec()``.
    """
    return (name or '').replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')


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
    # Only escape names that are *actually* reserved Python keywords. Python
    # keywords are all lowercase (``class``, ``from``, ``return``, ...), so a
    # PascalCase identifier like ``Class`` is a perfectly valid attribute or
    # variable name and should round-trip unchanged when ``lowercase=False``.
    if keyword.iskeyword(safe_name):
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


def bind_domain_field(domain_model, class_name: str, field_name: str):
    """Return the ``Property`` named *field_name* on *class_name*, or ``None``.

    Emitted GUI code calls this instead of inlining the lookup: the inline form
    used ``globals().get(...)``, an ``if``, a generator expression and
    ``_``-prefixed names, all refused by the safe BUML loader, so a domain-bound
    DataBinding could be exported and never re-imported.

    Returns ``None`` rather than raising when the model, class or field is
    absent - a GUI model is often exported without its domain model, so the
    binding degrades to "unbound" instead of breaking the import.
    """
    owner = _resolve_domain_class(domain_model, class_name)
    if owner is None:
        return None
    for attribute in getattr(owner, "attributes", ()) or ():
        if getattr(attribute, "name", None) == field_name:
            return attribute
    return None


def _resolve_domain_class(domain_model, class_name: str):
    """Return the class named *class_name* in *domain_model*, or ``None``."""
    if domain_model is None:
        return None
    try:
        return domain_model.get_class_by_name(class_name)
    except Exception:
        return None


def build_data_binding(domain_model, class_name: str, label_field=None, data_field=None, **kwargs):
    """Return a ``DataBinding`` over *class_name*, or ``None`` if the class is absent.

    Emitted GUI code calls this for the same reason as ``bind_domain_field``: the
    inline form needed ``globals()``, an ``if`` and generator expressions, which
    the safe BUML loader refuses. *label_field* / *data_field* are attribute
    names; the other keyword arguments go to the ``DataBinding`` constructor.
    """
    from besser.BUML.metamodel.gui.binding import DataBinding

    owner = _resolve_domain_class(domain_model, class_name)
    if not owner:
        return None
    binding = DataBinding(domain_concept=owner, **kwargs)
    if label_field:
        binding.label_field = bind_domain_field(domain_model, class_name, label_field)
    if data_field:
        binding.data_field = bind_domain_field(domain_model, class_name, data_field)
    return binding


def bind_association_end(domain_model, end_name: str):
    """Return the association end named *end_name*, or ``None``.

    A table's lookup column refers to an end that only exists inside its
    ``BinaryAssociation``; the inline ``next(...)`` lookup emitted before was
    refused by the safe BUML loader, so a project with such a table could be
    exported but never re-imported.
    """
    for association in getattr(domain_model, "associations", ()) or ():
        for end in association.ends:
            if end.name == end_name:
                return end
    return None


def bind_data_source(source, domain_model, class_name: str, field_names=None, label_field=None, value_field=None):
    """Bind a ``DataSourceElement`` to *class_name* when the domain model has it.

    The emitted code sets the unresolved names (``field_names``,
    ``label_field_name``, ``value_field_name``) before calling this, so a model
    loaded without its domain model keeps them; this only adds the resolved
    class and properties, in the order the former inline code used.
    """
    owner = _resolve_domain_class(domain_model, class_name)
    if not owner:
        return
    source.dataSourceClass = owner
    if field_names:
        source.field_names = list(field_names)
        source.fields = {attr for attr in owner.attributes if attr.name in field_names}
    if label_field:
        source.label_field = bind_domain_field(domain_model, class_name, label_field)
        source.label_field_name = label_field
    if value_field:
        source.value_field = bind_domain_field(domain_model, class_name, value_field)
        source.value_field_name = value_field
