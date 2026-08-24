import logging
from sqlalchemy import create_engine, MetaData, inspect, types as sqltypes

logger = logging.getLogger(__name__)

def get_database_metadata(connection_url: str) -> dict:
    """
    Connects to an external database and retrieves its schema metadata.

    Args:
        connection_url (str): The SQLAlchemy connection string.

    Returns:
        dict: A dictionary containing tables, columns, primary keys,
            foreign keys (one entry per constraint, so composite foreign keys
            stay grouped) and unique constraints.
    """
    try:
        engine = create_engine(connection_url)
        metadata = MetaData()

        # Reflect the tables from the database
        metadata.reflect(bind=engine)
        inspector = inspect(engine)

        db_metadata = {
            "tables": []
        }

        for table_name, table in metadata.tables.items():
            table_info = {
                "name": table_name,
                "columns": [],
                "foreign_keys": [],
                "unique_constraints": [],
            }

            # Extract column information
            for column in table.columns:
                col_info = {
                    "name": column.name,
                    "type": str(column.type),
                    "primary_key": column.primary_key,
                    "nullable": column.nullable
                }

                # Detect enumerated types. SQLAlchemy reflects PostgreSQL native
                # enums (CREATE TYPE ... AS ENUM), MySQL ENUM(...) and others as
                # an Enum type exposing the ordered value list in `.enums` and,
                # for named database types, the type name in `.name`.
                enum_values = getattr(column.type, "enums", None)
                if enum_values and (
                    isinstance(column.type, sqltypes.Enum) or hasattr(column.type, "enums")
                ):
                    col_info["enum_values"] = list(enum_values)
                    col_info["enum_name"] = getattr(column.type, "name", None)

                # Best-effort server-side default (e.g. DEFAULT 'active'). Kept as
                # raw SQL text; the parser normalises it per dialect.
                server_default = getattr(column, "server_default", None)
                if server_default is not None and getattr(server_default, "arg", None) is not None:
                    col_info["default"] = str(server_default.arg)

                table_info["columns"].append(col_info)

            # Extract foreign keys as one entry per constraint. Keeping the
            # constrained columns grouped lets the parser tell a single
            # composite foreign key apart from two independent ones.
            for fk in inspector.get_foreign_keys(table_name):
                fk_info = {
                    "name": fk.get("name"),
                    "columns": list(fk.get("constrained_columns", [])),
                    "references_table": fk.get("referred_table"),
                    "references_columns": list(fk.get("referred_columns", [])),
                }
                table_info["foreign_keys"].append(fk_info)

            # Extract unique constraints (used to detect one-to-one relationships).
            # Some dialects do not implement this, so fail soft.
            try:
                for uc in inspector.get_unique_constraints(table_name):
                    cols = list(uc.get("column_names", []))
                    if cols:
                        table_info["unique_constraints"].append(cols)
            except NotImplementedError:
                pass

            db_metadata["tables"].append(table_info)

        return db_metadata

    except Exception as e:
        logger.error(f"Error retrieving database metadata: {e}")
        raise ValueError(f"Failed to retrieve database metadata: {str(e)}")
