import logging
from sqlalchemy import create_engine, MetaData, inspect

logger = logging.getLogger(__name__)

def get_database_metadata(connection_url: str) -> dict:
    """
    Connects to an external database and retrieves its schema metadata.
    
    Args:
        connection_url (str): The SQLAlchemy connection string.
        
    Returns:
        dict: A dictionary containing tables, columns, primary keys, and foreign keys.
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
                "foreign_keys": []
            }
            
            # Extract column information
            for column in table.columns:
                col_info = {
                    "name": column.name,
                    "type": str(column.type),
                    "primary_key": column.primary_key,
                    "nullable": column.nullable
                }
                table_info["columns"].append(col_info)
            
            # Extract foreign key information using inspector for more detailed relationships
            fks = inspector.get_foreign_keys(table_name)
            for fk in fks:
                # SQLAlchemy inspector returns foreign keys, we map them to our format
                # A single foreign key constraint might involve multiple columns, 
                # but we'll try to map it per column for simplicity as requested, OR just keep the array
                for constrained_col, referred_col in zip(fk['constrained_columns'], fk['referred_columns']):
                    fk_info = {
                        "column": constrained_col,
                        "references_table": fk['referred_table'],
                        "references_column": referred_col
                    }
                    table_info["foreign_keys"].append(fk_info)
            
            db_metadata["tables"].append(table_info)
            
        return db_metadata
        
    except Exception as e:
        logger.error(f"Error retrieving database metadata: {e}")
        raise ValueError(f"Failed to retrieve database metadata: {str(e)}")
