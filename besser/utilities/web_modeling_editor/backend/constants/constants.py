VALID_PRIMITIVE_TYPES = {
    "str": "str",
    "string": "str",
    "int": "int",
    "integer": "int",
    "float": "float",
    "double": "float",
    "bool": "bool",
    "boolean": "bool",
    "date": "date",
    "datetime": "datetime",
    "time": "time",
    "timedelta": "timedelta",
    "any": "any"
}

VISIBILITY_MAP = {
    "+": "public",
    "-": "private",
    "#": "protected",
    "~": "package"
}

RELATIONSHIP_TYPES = {
    "bidirectional": "ClassBidirectional",
    "unidirectional": "ClassUnidirectional",
    "composition": "ClassComposition",
    "inheritance": "ClassInheritance"
} 