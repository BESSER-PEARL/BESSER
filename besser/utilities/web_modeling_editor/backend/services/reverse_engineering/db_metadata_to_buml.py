"""
Convert database metadata to B-UML DomainModel.
"""
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Multiplicity, BinaryAssociation,
    Enumeration, EnumerationLiteral,
    StringType, IntegerType, FloatType, BooleanType, TimeType, DateType,
    DateTimeType, TimeDeltaType, UNLIMITED_MAX_MULTIPLICITY
)

_TYPE_MAPPING = {
    "string": StringType, "varchar": StringType, "char": StringType, "text": StringType,
    "uuid": StringType, "json": StringType, "int": IntegerType, "integer": IntegerType,
    "smallint": IntegerType, "bigint": IntegerType, "serial": IntegerType,
    "bigserial": IntegerType, "float": FloatType, "real": FloatType, "double": FloatType,
    "double precision": FloatType, "numeric": FloatType, "decimal": FloatType,
    "bool": BooleanType, "boolean": BooleanType, "date": DateType,
    "timestamp": DateTimeType, "timestamp without time zone": DateTimeType,
    "timestamp with time zone": DateTimeType, "datetime": DateTimeType,
    "time": TimeType, "timedelta": TimeDeltaType, "year": IntegerType,
    "tinyint": IntegerType, "mediumint": IntegerType, "longtext": StringType,
    "mediumtext": StringType, "tinytext": StringType, "blob": StringType,
    "longblob": StringType, "mediumblob": StringType, "tinyblob": StringType,
    "integer primary key": IntegerType,
}


def map_type(type_str: str) -> type:
    normalized = type_str.strip().lower()
    base_type = normalized.split('(')[0].strip()
    return _TYPE_MAPPING.get(base_type, StringType)


def _fk_columns(fk: dict) -> list:
    """Return the constrained column(s) of a foreign key (supports composite keys)."""
    cols = fk.get("columns")
    if cols:
        return list(cols)
    # Backward compatibility with the older single-column format.
    if fk.get("column") is not None:
        return [fk["column"]]
    return []


def _all_fk_columns(table: dict) -> set:
    """All columns of the table that participate in any foreign key."""
    cols = set()
    for fk in table.get("foreign_keys", []):
        cols.update(_fk_columns(fk))
    return cols


def _is_bridge_table(table: dict) -> bool:
    """Check if a table is a bridge table for many-to-many relationships.

    A bridge table has exactly two foreign-key constraints and no columns
    other than the ones participating in those foreign keys.
    """
    fks = table.get("foreign_keys", [])
    if len(fks) != 2:
        return False

    fk_cols = _all_fk_columns(table)
    non_fk_cols = [col for col in table.get("columns", []) if col["name"] not in fk_cols]

    return len(non_fk_cols) == 0


def _get_bridge_table_targets(table: dict) -> tuple:
    """Get the two tables connected by a bridge table."""
    fks = table.get("foreign_keys", [])
    if len(fks) == 2:
        return (fks[0]["references_table"], fks[1]["references_table"])
    return None


def _role_from_columns(columns: list, fallback: str) -> str:
    """Derive a readable role name from a FK column, e.g. 'customer_id' -> 'customer'.

    Falls back to the referenced table name for composite keys or when the
    column name carries no usable stem.
    """
    if len(columns) == 1:
        stem = columns[0].lower()
        if stem.endswith("_id"):
            stem = stem[:-3]
        elif stem.endswith("id") and len(stem) > 2:
            stem = stem[:-2]
        stem = stem.strip("_")
        if stem:
            return stem
    return fallback.lower()


def _unique_name(base: str, used: set) -> str:
    """Return a name not present in `used`, suffixing _2, _3, ... on collision."""
    name = base
    counter = 2
    while name in used:
        name = f"{base}_{counter}"
        counter += 1
    used.add(name)
    return name


def _columns_nullable(table: dict, columns: list) -> bool:
    """True if every given column is nullable (an optional foreign key)."""
    by_name = {c["name"]: c for c in table.get("columns", [])}
    return all(by_name.get(c, {}).get("nullable", True) for c in columns)


def _clean_default(raw) -> str:
    """Normalise a reflected server default to a bare value.

    Handles PostgreSQL casts ('active'::student_status) and quoted literals
    ('active', "active"). Returns None when there is nothing usable.
    """
    if raw is None:
        return None
    value = str(raw).strip()
    # Drop a PostgreSQL type cast suffix: 'active'::student_status -> 'active'
    value = value.split("::", 1)[0].strip()
    # Strip a single pair of surrounding quotes
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value or None


def _build_enumeration(col: dict, table_name: str, registry: dict) -> Enumeration:
    """Return an Enumeration for an enum column, or None if the column isn't one.

    Named database enum types (e.g. PostgreSQL's) are shared across columns, so
    they are cached in `registry` by type name and reused. Anonymous enums
    (e.g. inline MySQL ENUM(...)) get a per-column name and are never shared.
    """
    values = col.get("enum_values")
    if not values:
        return None

    enum_name = col.get("enum_name")
    key = enum_name if enum_name else f"{table_name}.{col['name']}"
    if key in registry:
        return registry[key]

    display_name = enum_name or f"{table_name}_{col['name']}"
    literals = []
    seen = set()
    for value in values:
        literal_name = str(value)
        if literal_name in seen:
            continue
        seen.add(literal_name)
        literals.append(EnumerationLiteral(name=literal_name))

    enumeration = Enumeration(name=display_name, literals=set(literals))
    # Preserve the database definition order for deterministic rendering
    # (the converter reads this attribute when present).
    enumeration._ordered_literals = literals
    registry[key] = enumeration
    return enumeration


def _columns_unique(table: dict, columns: list) -> bool:
    """True if the given columns form a unique constraint or the full primary key.

    Used to distinguish a one-to-one relationship from a one-to-many one.
    """
    col_set = set(columns)
    for uc in table.get("unique_constraints", []):
        if set(uc) == col_set:
            return True
    pk = {c["name"] for c in table.get("columns", []) if c.get("primary_key", False)}
    return bool(pk) and pk == col_set


def parse_metadata_to_buml(metadata: dict, model_name: str = "DB_DomainModel") -> DomainModel:
    classes = {}
    associations = set()
    bridge_tables = set()
    tables_by_name = {t["name"]: t for t in metadata.get("tables", [])}

    used_assoc_names = set()
    # Per-class set of already-used attribute/association-end names, to keep
    # them unique when a table has several foreign keys to the same target,
    # self-references, or self many-to-many relationships.
    used_member_names = {}
    # Enumerations discovered across the schema, cached so a shared database
    # enum type (e.g. a PostgreSQL type) maps to a single Enumeration object.
    enum_registry = {}

    # Identify bridge tables
    for table in metadata.get("tables", []):
        if _is_bridge_table(table):
            bridge_tables.add(table["name"])

    # Create classes (excluding bridge tables)
    for table in metadata.get("tables", []):
        if table["name"] in bridge_tables:
            continue

        fk_cols = _all_fk_columns(table)
        pk_cols = [c["name"] for c in table.get("columns", []) if c.get("primary_key", False)]
        # The metamodel allows a single id attribute, so only a single-column
        # primary key is mapped to `is_id`; composite keys stay plain attributes.
        single_pk = pk_cols[0] if len(pk_cols) == 1 else None
        attrs = set()
        member_names = set()
        for col in table.get("columns", []):
            col_name = col["name"]
            is_pk = col.get("primary_key", False)
            # Foreign-key columns become associations. A primary-key foreign key
            # is also kept as an attribute so the class stays identifiable
            # (e.g. shared-primary-key one-to-one tables).
            if col_name in fk_cols and not is_pk:
                continue
            # Enum columns are typed by an Enumeration; everything else by a
            # primitive type. For enums, map a matching server default to its
            # literal so the diagram shows it (e.g. status: student_status = active).
            enumeration = _build_enumeration(col, table["name"], enum_registry)
            if enumeration is not None:
                prop_type = enumeration
                cleaned_default = _clean_default(col.get("default"))
                default_value = next(
                    (lit for lit in enumeration.literals if lit.name == cleaned_default), None
                )
            else:
                prop_type = map_type(col["type"])
                default_value = None
            attrs.add(Property(
                name=col_name, type=prop_type, is_id=(col_name == single_pk),
                default_value=default_value,
            ))
            member_names.add(col_name)
        classes[table["name"]] = Class(name=table["name"], attributes=attrs)
        used_member_names[table["name"]] = member_names

    # Create associations from foreign keys (excluding bridge table FKs)
    for table in metadata.get("tables", []):
        if table["name"] in bridge_tables:
            continue

        source_table = table["name"]
        for fk in table.get("foreign_keys", []):
            target_table = fk.get("references_table")

            # Skip if target is a bridge table or target table doesn't exist
            if target_table in bridge_tables or target_table not in classes:
                continue

            columns = _fk_columns(fk)
            source_class = classes[source_table]
            target_class = classes[target_table]

            # FK side: exactly one target, mandatory unless the FK column(s)
            # are nullable. Back side: 0..1 when the FK is unique (one-to-one),
            # otherwise 0..* (one-to-many).
            source_mult = (
                Multiplicity(0, 1) if _columns_nullable(table, columns) else Multiplicity(1, 1)
            )
            target_mult = (
                Multiplicity(0, 1) if _columns_unique(table, columns)
                else Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)
            )

            role = _role_from_columns(columns, target_table)
            # The end typed by the target lives on the source class; the "list"
            # end typed by the source lives on the target class.
            source_end = _unique_name(role, used_member_names[source_table])
            target_end = _unique_name(f"{source_table.lower()}_list", used_member_names[target_table])

            source_prop = Property(name=source_end, type=target_class, multiplicity=source_mult)
            target_prop = Property(name=target_end, type=source_class, multiplicity=target_mult)

            assoc_base = f"{source_table}_{'_'.join(columns) or target_table}_{target_table}_fk"
            assoc_name = _unique_name(assoc_base, used_assoc_names)
            associations.add(
                BinaryAssociation(name=assoc_name, ends={source_prop, target_prop})
            )

    # Create many-to-many associations from bridge tables
    for bridge_table_name in bridge_tables:
        bridge_table = tables_by_name[bridge_table_name]
        targets = _get_bridge_table_targets(bridge_table)
        if not targets:
            continue

        table_a, table_b = targets
        if table_a not in classes or table_b not in classes:
            continue

        class_a = classes[table_a]
        class_b = classes[table_b]
        fks = bridge_table.get("foreign_keys", [])
        # fks[0] references table_a, fks[1] references table_b.
        role_a = _role_from_columns(_fk_columns(fks[0]), table_a)
        role_b = _role_from_columns(_fk_columns(fks[1]), table_b)

        mult_many = Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)
        # The end living on class_a points to class_b, and vice versa.
        end_on_a = _unique_name(role_b, used_member_names[table_a])
        end_on_b = _unique_name(role_a, used_member_names[table_b])
        prop_a = Property(name=end_on_a, type=class_b, multiplicity=mult_many)
        prop_b = Property(name=end_on_b, type=class_a, multiplicity=mult_many)

        assoc_name = _unique_name(f"{bridge_table_name}_m2m", used_assoc_names)
        associations.add(
            BinaryAssociation(name=assoc_name, ends={prop_a, prop_b})
        )

    all_types = set(classes.values()) | set(enum_registry.values())
    return DomainModel(name=model_name, types=all_types, associations=associations)
