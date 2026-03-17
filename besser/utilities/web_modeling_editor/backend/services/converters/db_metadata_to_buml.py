"""
Convert database metadata to B-UML DomainModel.
"""
from besser.BUML.metamodel.structural import DomainModel, Class, Property, Multiplicity, BinaryAssociation, StringType, IntegerType, FloatType, BooleanType, TimeType, DateType, DateTimeType, TimeDeltaType
from datetime import datetime

# Example metadata format:
# metadata = {
#     "tables": [
#         {
#             "name": "Book",
#             "columns": [
#                 {"name": "title", "type": "string"},
#                 {"name": "pages", "type": "int"},
#                 {"name": "release", "type": "date"}
#             ],
#             "relations": [
#                 {"target": "Author", "type": "many-to-one", "source_column": "author_id", "target_column": "id"}
#             ]
#         },
#         ...
#     ]
# }

def map_type(type_str):
    mapping = {
        # Generic types
        "string": StringType,
        "varchar": StringType,
        "char": StringType,
        "text": StringType,
        "uuid": StringType,
        "json": StringType,
        "int": IntegerType,
        "integer": IntegerType,
        "smallint": IntegerType,
        "bigint": IntegerType,
        "serial": IntegerType,
        "bigserial": IntegerType,
        "float": FloatType,
        "real": FloatType,
        "double": FloatType,
        "double precision": FloatType,
        "numeric": FloatType,
        "decimal": FloatType,
        "bool": BooleanType,
        "boolean": BooleanType,
        # Date/time types
        "date": DateType,
        "timestamp": DateTimeType,
        "timestamp without time zone": DateTimeType,
        "timestamp with time zone": DateTimeType,
        "datetime": DateTimeType,
        "time": TimeType,
        "timedelta": TimeDeltaType,
        "year": DateType,
        # MySQL specific
        "tinyint": IntegerType,
        "mediumint": IntegerType,
        "longtext": StringType,
        "mediumtext": StringType,
        "tinytext": StringType,
        "blob": StringType,
        "longblob": StringType,
        "mediumblob": StringType,
        "tinyblob": StringType,
        # SQLite
        "integer primary key": IntegerType,
        # Fallback
    }
    normalized = type_str.strip().lower()
    # Remove extra details (e.g., VARCHAR(255), DECIMAL(10,2))
    base_type = normalized.split('(')[0].strip()
    return mapping.get(base_type, StringType)


def parse_metadata_to_buml(metadata: dict, model_name: str = "DB_DomainModel") -> DomainModel:
    classes = {}
    associations = set()

    # Create classes
    for table in metadata.get("tables", []):
        attrs = set()
        for col in table.get("columns", []):
            attrs.add(Property(name=col["name"], type=map_type(col["type"])))
        classes[table["name"]] = Class(name=table["name"], attributes=attrs)

    # Create associations
    for table in metadata.get("tables", []):
        for rel in table.get("relations", []):
            source_class = classes[table["name"]]
            target_class = classes[rel["target"]]
            # Determine multiplicity
            if rel["type"] == "many-to-one":
                source_mult = Multiplicity(0, "*")
                target_mult = Multiplicity(1, 1)
            elif rel["type"] == "one-to-many":
                source_mult = Multiplicity(1, 1)
                target_mult = Multiplicity(0, "*")
            elif rel["type"] == "one-to-one":
                source_mult = Multiplicity(1, 1)
                target_mult = Multiplicity(1, 1)
            else:
                source_mult = Multiplicity(0, "*")
                target_mult = Multiplicity(0, "*")
            # Create properties for association ends
            source_prop = Property(name=f"{table['name']}_to_{rel['target']}", type=target_class, multiplicity=source_mult)
            target_prop = Property(name=f"{rel['target']}_to_{table['name']}", type=source_class, multiplicity=target_mult)
            associations.add(BinaryAssociation(name=f"{table['name']}_{rel['target']}_assoc", ends={source_prop, target_prop}))

    return DomainModel(name=model_name, types=set(classes.values()), associations=associations)
