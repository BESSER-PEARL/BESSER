"""
Convert database metadata to B-UML DomainModel.
"""
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Multiplicity, BinaryAssociation,
    StringType, IntegerType, FloatType, BooleanType, TimeType, DateType,
    DateTimeType, TimeDeltaType
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
    "time": TimeType, "timedelta": TimeDeltaType, "year": DateType,
    "tinyint": IntegerType, "mediumint": IntegerType, "longtext": StringType,
    "mediumtext": StringType, "tinytext": StringType, "blob": StringType,
    "longblob": StringType, "mediumblob": StringType, "tinyblob": StringType,
    "integer primary key": IntegerType,
}


def map_type(type_str: str) -> type:
    normalized = type_str.strip().lower()
    base_type = normalized.split('(')[0].strip()
    return _TYPE_MAPPING.get(base_type, StringType)


def _is_bridge_table(table: dict, all_tables: dict) -> bool:
    """Check if a table is a bridge table for many-to-many relationships."""
    fks = table.get("foreign_keys", [])
    cols = table.get("columns", [])

    if len(fks) != 2:
        return False

    fk_cols = {fk["column"] for fk in fks}
    non_fk_cols = [col for col in cols if col["name"] not in fk_cols]

    return len(non_fk_cols) == 0


def _get_bridge_table_targets(table: dict) -> tuple:
    """Get the two tables connected by a bridge table."""
    fks = table.get("foreign_keys", [])
    if len(fks) == 2:
        return (fks[0]["references_table"], fks[1]["references_table"])
    return None


def parse_metadata_to_buml(metadata: dict, model_name: str = "DB_DomainModel") -> DomainModel:
    classes = {}
    associations = set()
    bridge_tables = set()
    tables_by_name = {t["name"]: t for t in metadata.get("tables", [])}

    # Identify bridge tables
    for table in metadata.get("tables", []):
        if _is_bridge_table(table, tables_by_name):
            bridge_tables.add(table["name"])

    # Create classes (excluding bridge tables)
    for table in metadata.get("tables", []):
        if table["name"] in bridge_tables:
            continue

        attrs = set()
        for col in table.get("columns", []):
            col_name = col["name"]
            if not any(fk["column"] == col_name for fk in table.get("foreign_keys", [])):
                attrs.add(Property(name=col_name, type=map_type(col["type"]), is_id=col.get("primary_key", False)))
        classes[table["name"]] = Class(name=table["name"], attributes=attrs)

    # Create associations from foreign keys (excluding bridge table FKs)
    for table in metadata.get("tables", []):
        if table["name"] in bridge_tables:
            continue

        for fk in table.get("foreign_keys", []):
            source_table = table["name"]
            target_table = fk["references_table"]

            # Skip if target is a bridge table or target table doesn't exist
            if target_table in bridge_tables or target_table not in classes:
                continue

            source_class = classes[source_table]
            target_class = classes[target_table]

            # source_table owns the FK → many of source per one target
            # target_table is referenced   → exactly 1 target per source
            source_prop = Property(
                name=target_table.lower(), type=target_class, multiplicity=Multiplicity(1, 1)
            )
            target_prop = Property(
                name=f"{source_table.lower()}_list", type=source_class, multiplicity=Multiplicity(0, "*")
            )
            associations.add(
                BinaryAssociation(
                    name=f"{source_table}_{target_table}_fk", ends={source_prop, target_prop}
                )
            )

    # Create many-to-many associations from bridge tables
    for bridge_table_name in bridge_tables:
        targets = _get_bridge_table_targets(tables_by_name[bridge_table_name])
        if targets and len(targets) == 2:
            table_a, table_b = targets

            if table_a not in classes or table_b not in classes:
                continue

            class_a = classes[table_a]
            class_b = classes[table_b]

            # Both sides: many-to-many
            mult_many = Multiplicity(0, "*")

            prop_a = Property(
                name=table_b.lower(), type=class_b, multiplicity=mult_many
            )
            prop_b = Property(
                name=table_a.lower(), type=class_a, multiplicity=mult_many
            )
            associations.add(
                BinaryAssociation(
                    name=f"{table_a}_{table_b}_m2m", ends={prop_a, prop_b}
                )
            )

    return DomainModel(name=model_name, types=set(classes.values()), associations=associations)
