"""Shared Java type mapping and naming helpers for the Spring Boot generator.

Every sub-generator of the Spring package (entity, repository, service,
controller) resolves B-UML types and B-UML names through this module, so that a
single attribute is never rendered with one Java type in the entity and another
one in the repository signature that reads it.

The naming helpers also act as the security boundary of the generator:
``NamedElement`` only rejects whitespace and hyphens, so a model may legally
contain a class called ``../../evil``. Every B-UML name that ends up as a file
name, a Java type, a field or a SQL identifier is therefore funnelled through
:func:`to_java_identifier` (or one of its wrappers) first.
"""

import re

from besser.BUML.metamodel.structural import (
    UNLIMITED_MAX_MULTIPLICITY,
    AnyType,
    BooleanType,
    Class,
    DateTimeType,
    DateType,
    Enumeration,
    FloatType,
    IntegerType,
    Multiplicity,
    StringType,
    TimeDeltaType,
    TimeType,
    Type,
)

#: Single source of truth for the B-UML primitive type -> Java type mapping.
#: Shared by the entity, repository, service and controller generators so that
#: a ``time`` attribute is a ``LocalTime`` everywhere it appears.
JAVA_TYPES: dict[str, str] = {
    StringType.name: "String",
    BooleanType.name: "Boolean",
    IntegerType.name: "Integer",
    FloatType.name: "Float",
    DateType.name: "LocalDate",
    DateTimeType.name: "LocalDateTime",
    TimeType.name: "LocalTime",
    TimeDeltaType.name: "Duration",
    AnyType.name: "Object",
}

#: Java type used for any B-UML type that has no explicit mapping.
DEFAULT_JAVA_TYPE: str = "Object"

#: ``java.*`` imports required by the Java types that are not in ``java.lang``.
JAVA_TYPE_IMPORTS: dict[str, str] = {
    "LocalDate": "java.time.LocalDate",
    "LocalDateTime": "java.time.LocalDateTime",
    "LocalTime": "java.time.LocalTime",
    "Duration": "java.time.Duration",
}

#: Reserved words and literals that cannot be used as Java identifiers.
JAVA_KEYWORDS: frozenset[str] = frozenset({
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while",
    "true", "false", "null", "_",
})

#: A Java package is a dot-separated list of lowercase identifiers.
JAVA_PACKAGE_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$")

_ILLEGAL_IDENTIFIER_CHARS = re.compile(r"[^0-9A-Za-z_$]+")
_SNAKE_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_SNAKE_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")

#: Fallback used when a name sanitizes down to nothing at all (e.g. ``"../.."``).
_FALLBACK_IDENTIFIER = "Unnamed"


def to_java_identifier(name: str, capitalize: bool = False) -> str:
    """Turn an arbitrary B-UML name into a legal Java identifier.

    Every character that Java does not accept in an identifier is replaced by an
    underscore, which is what stops path separators and ``..`` segments from
    escaping the output directory. Identifiers that would start with a digit are
    prefixed with an underscore, and Java keywords get a trailing underscore.

    Args:
        name: The B-UML name to sanitize.
        capitalize: Whether the first character should be upper-cased
            (used for type names).

    Returns:
        A legal, non-empty Java identifier.
    """
    cleaned = _ILLEGAL_IDENTIFIER_CHARS.sub("_", str(name))
    # Underscores introduced by the substitution at the edges carry no meaning
    # (``../../evil`` -> ``_evil``) and only make the generated code noisier.
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = _FALLBACK_IDENTIFIER
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    if capitalize:
        # Capitalizing first is what makes a class called "new" legal as "New";
        # only what is left after it can still collide with a keyword.
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned in JAVA_KEYWORDS:
        cleaned = f"{cleaned}_"
    return cleaned


def to_java_class_name(name: str) -> str:
    """str: The sanitized, capitalized Java type name for a B-UML name."""
    return to_java_identifier(name, capitalize=True)


def to_java_field_name(name: str) -> str:
    """str: The sanitized Java field/parameter name for a B-UML name."""
    return to_java_identifier(name)


def to_java_accessor_suffix(name: str) -> str:
    """str: The capitalized suffix of the ``getX``/``setX`` pair for a field."""
    field = to_java_field_name(name)
    return field[0].upper() + field[1:]


def to_snake_case(name: str) -> str:
    """str: The sanitized ``snake_case`` form of a name (column/table names)."""
    sanitized = to_java_identifier(name)
    partial = _SNAKE_BOUNDARY_1.sub(r"\1_\2", sanitized)
    return _SNAKE_BOUNDARY_2.sub(r"\1_\2", partial).lower()


def pluralize(name: str) -> str:
    """str: A naive English plural, used for ``@Table`` names."""
    if name.endswith("y"):
        return name[:-1] + "ies"
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    return name + "s"


#: B-UML visibility -> Java access modifier. ``package`` is the default access
#: in Java and has no keyword, so it maps to the empty modifier.
JAVA_VISIBILITY: dict[str, str] = {
    "public": "public",
    "private": "private",
    "protected": "protected",
    "package": "",
}


def java_visibility(visibility: str) -> str:
    """str: The Java access modifier for a B-UML visibility."""
    return JAVA_VISIBILITY.get(visibility, "public")


def validate_java_package(package_name: str) -> str:
    """Validate a Java package name and return it unchanged.

    The package name is split into directories, so an unvalidated value would be
    a second way out of the output directory.

    Raises:
        ValueError: If the package name is not a dot-separated list of lowercase
            Java identifiers, or if one of its segments is a Java keyword.
    """
    if not isinstance(package_name, str) or not JAVA_PACKAGE_PATTERN.match(package_name):
        raise ValueError(
            f"Invalid Java package name: {package_name!r}. A package name must be a "
            "dot-separated list of lowercase identifiers, e.g. 'com.example.app'."
        )
    for segment in package_name.split("."):
        if segment in JAVA_KEYWORDS:
            raise ValueError(
                f"Invalid Java package name: {package_name!r}. "
                f"'{segment}' is a reserved Java keyword."
            )
    return package_name


def java_type_for(buml_type: Type, model_type_names: set[str] | None = None) -> str:
    """Map a B-UML type to its Java counterpart.

    User-defined types (classes and enumerations) keep their own, sanitized
    name. Primitive types are looked up in :data:`JAVA_TYPES`, and anything the
    mapping does not know about degrades to ``Object`` rather than raising.

    Args:
        buml_type: The B-UML type to map.
        model_type_names: Names of the classes and enumerations of the model.
            Used so that a type referenced by name only is still recognised as
            user-defined.
    """
    if buml_type is None:
        return "void"
    name = buml_type.name
    if isinstance(buml_type, (Class, Enumeration)) or (model_type_names and name in model_type_names):
        return to_java_class_name(name)
    return JAVA_TYPES.get(name, DEFAULT_JAVA_TYPE)


def java_type_import(java_type_name: str) -> str | None:
    """str | None: The import required by a Java type name, if any."""
    return JAVA_TYPE_IMPORTS.get(java_type_name)


def is_many(multiplicity: Multiplicity | None) -> bool:
    """bool: Whether a multiplicity denotes "many" (an upper bound above one).

    ``*`` is stored as :data:`UNLIMITED_MAX_MULTIPLICITY` by the metamodel, so
    both the unbounded and the explicitly bounded cases are covered here.
    """
    if multiplicity is None:
        return False
    return multiplicity.max == UNLIMITED_MAX_MULTIPLICITY or multiplicity.max > 1


def get_id_attribute(cls: Class):
    """Property: The identifier attribute of a class, inherited ones included.

    Raises:
        ValueError: If the class has no attribute flagged with ``is_id``.
    """
    id_attributes = [attr for attr in cls.all_attributes() if attr.is_id]
    if not id_attributes:
        raise ValueError(
            f"Class '{cls.name}' has no identifier attribute. The Spring generator maps "
            "every concrete class to a JPA entity, which requires exactly one attribute "
            "marked with 'is_id=True' (directly or inherited)."
        )
    return sorted(id_attributes, key=lambda attr: attr.name)[0]
