import subprocess
import os
import sys
import tempfile
import shutil
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.generators.sql_alchemy import SQLAlchemyGenerator

class SQLGenerator(GeneratorInterface):
    def __init__(self, model: DomainModel, output_dir: str = None, sql_dialect: str = "sqlite"):
        super().__init__(model, output_dir)
        self.sql_dialect = sql_dialect

    def generate(self):
        """
        Generates SQL code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory is not specified, the code will be stored in the <current directory>/output folder.
        """
        # Determine output path
        final_output_dir = self.output_dir or os.path.join(os.getcwd(), "output")
        os.makedirs(final_output_dir, exist_ok=True)

        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_py_path = os.path.join(temp_dir, "sql_alchemy.py")

            # Generate SQLAlchemy code into the temporary folder
            alchemy_gen = SQLAlchemyGenerator(model=self.model, output_dir=temp_dir)
            alchemy_gen.generate(dbms=self.sql_dialect)

            # Read and strip final line (optional)
            with open(temp_py_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            lines = lines[:-1]  # remove last line if needed
            content = "\n".join(lines)
            dialect = self.sql_dialect

            # Append DDL dump logic
            content += f"""

from sqlalchemy.schema import CreateTable
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
sql_file_path = os.path.join(current_directory, f"tables_{dialect}.sql")

ddl_statements = []


# --- Emit ENUM types ---
if '{dialect}'.lower() != 'oracle':
    for table in Base.metadata.tables.values():
        for col in table.columns:
            if isinstance(col.type, Enum):
                enum_name = getattr(col.type, 'name', None) or (getattr(col, 'name', 'unknown_enum') + '_enum')
                enum_class = getattr(col.type, 'enum_class', None)
                enum_values = [repr(e.value) for e in enum_class] if enum_class else []
                ddl_statements.append(f"CREATE TYPE {{enum_name}} AS ENUM ({{', '.join(enum_values)}});")

for table in Base.metadata.sorted_tables:
    ddl = str(CreateTable(table).compile(engine))
    # For Oracle, add CHECK constraint for enum columns
    if '{dialect}'.lower() == 'oracle':
        for col in table.columns:
            enum_class = getattr(col.type, 'enum_class', None)
            if enum_class:
                enum_values = [repr(e.value) for e in enum_class]
                col_name = col.name
                # Try both quoted and unquoted column names
                quoted_col = f'"{{col_name}}"'
                unquoted_col = col_name
                
                # Search for column definition (quoted or unquoted)
                search_patterns = [
                    (f'{{quoted_col}} VARCHAR', quoted_col),  # Column is quoted in DDL
                    (f'{{unquoted_col}} VARCHAR', unquoted_col)  # Column is unquoted in DDL
                ]
                
                for search_str, check_col_ref in search_patterns:
                    if search_str in ddl:
                        # Use the same column reference style (quoted/unquoted) in CHECK
                        check_constraint = f' CHECK ({{check_col_ref}} IN ({{", ".join(enum_values)}}))'
                        # Find the end of this column's NOT NULL clause
                        idx = ddl.find(search_str)
                        not_null_idx = ddl.find('NOT NULL', idx)
                        if not_null_idx > 0:
                            end_idx = not_null_idx + len('NOT NULL')
                            # Insert the CHECK constraint
                            ddl = ddl[:end_idx] + check_constraint + ddl[end_idx:]
                            break
    ddl_statements.append(ddl)

with open(sql_file_path, "w") as f:
    for stmt in ddl_statements:
        f.write(f"{{stmt}};\\n\\n")
"""

            # Write back modified file
            with open(temp_py_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Run the temporary Python file
            try:
                subprocess.run([sys.executable, temp_py_path], check=True, text=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running generated SQLAlchemy file:\n{e.stderr}")

            # Copy generated .sql file to final output location
            src_sql = os.path.join(temp_dir, f"tables_{self.sql_dialect}.sql")
            dst_sql = os.path.join(final_output_dir, f"tables_{self.sql_dialect}.sql")
            shutil.copyfile(src_sql, dst_sql)
