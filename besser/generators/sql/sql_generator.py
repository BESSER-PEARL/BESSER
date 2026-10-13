import subprocess
import os
import sys
import tempfile
import shutil
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.generators.sql_alchemy import SQLAlchemyGenerator

# Marker of the trailing "database connection" section of the SQLAlchemy module.
# Everything from here on (``create_engine(DATABASE_URL)``, the ``__main__``
# block) needs a DB driver and is irrelevant for DDL generation, so it is cut.
_DB_CONNECTION_MARKER = "\n# Database connection"

# SQLAlchemy URL whose dialect the DDL is compiled against. The DDL is compiled
# against a mock engine (``create_mock_engine``), which builds the dialect object
# without importing the DBAPI driver, so no database driver (pyodbc, psycopg2,
# oracledb, ...) has to be installed. The drivers match the default
# ``DATABASE_URL`` of the generated SQLAlchemy module so the DDL is unchanged.
DDL_DIALECT_URLS = {
    "sqlite": "sqlite://",
    "postgresql": "postgresql+psycopg2://",
    "mysql": "mysql+pymysql://",
    "mariadb": "mysql+pymysql://",
    "mssql": "mssql+pyodbc://",
    "oracle": "oracle+oracledb://",
}


class SQLGenerator(GeneratorInterface):
    """Generates SQL DDL (``CREATE TABLE`` statements) for a B-UML domain model.

    Args:
        model (DomainModel): The B-UML domain model.
        output_dir (str, optional): Output directory; defaults to ``<cwd>/output``.
        sql_dialect (str, optional): One of ``sqlite``, ``postgresql``, ``mysql``,
            ``mssql``, ``mariadb`` or ``oracle``. Defaults to ``sqlite``.
    """

    def __init__(self, model: DomainModel, output_dir: str = None, sql_dialect: str = "sqlite"):
        super().__init__(model, output_dir)
        self.sql_dialect = sql_dialect

    def generate(self):
        """
        Generates SQL code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory is not specified, the code will be stored in the <current directory>/output folder.

        Raises:
            ValueError: If the dialect is not supported.
            RuntimeError: If the DDL could not be produced.
        """
        if self.sql_dialect not in DDL_DIALECT_URLS:
            raise ValueError(
                f"Invalid SQL dialect '{self.sql_dialect}'. "
                f"Valid options are {', '.join(sorted(DDL_DIALECT_URLS))}."
            )

        # Determine output path
        final_output_dir = self.output_dir or os.path.join(os.getcwd(), "output")
        os.makedirs(final_output_dir, exist_ok=True)

        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_py_path = os.path.join(temp_dir, "sql_alchemy.py")

            # Generate SQLAlchemy code into the temporary folder
            alchemy_gen = SQLAlchemyGenerator(model=self.model, output_dir=temp_dir)
            alchemy_gen.generate(dbms=self.sql_dialect)

            # Keep only the model definitions: drop the database connection
            # section (engine creation needs a driver, __main__ creates tables).
            with open(temp_py_path, "r", encoding="utf-8") as f:
                content = f.read()
            marker_idx = content.find(_DB_CONNECTION_MARKER)
            if marker_idx == -1:
                raise RuntimeError(
                    "SQL generation failed: could not locate the database connection "
                    "section in the generated SQLAlchemy module."
                )
            content = content[:marker_idx]
            dialect = self.sql_dialect

            # Append DDL dump logic
            ddl_dump_code = '''

from sqlalchemy import create_mock_engine as _create_mock_engine
from sqlalchemy.schema import CreateTable
import os

# Dialect only: building it does not import the DBAPI driver.
_ddl_dialect = _create_mock_engine("DIALECT_URL_PLACEHOLDER", lambda *args, **kwargs: None).dialect
if 'DIALECT_PLACEHOLDER' == 'mssql':
    # Without a connection the server version is unknown and SQL Server is assumed to
    # predate 2008 (DATE/TIME rendered as DATETIME). Target a current SQL Server.
    _ddl_dialect.server_version_info = (16,)
    _ddl_dialect.deprecate_large_types = True

current_directory = os.path.dirname(os.path.realpath(__file__))
sql_file_path = os.path.join(current_directory, "tables_DIALECT_PLACEHOLDER.sql")

ddl_statements = []

# --- Emit ENUM types: PostgreSQL only; other dialects render enums inline (Oracle via CHECK) ---
if 'DIALECT_PLACEHOLDER'.lower() == 'postgresql':
    for table in Base.metadata.tables.values():
        for col in table.columns:
            if isinstance(col.type, Enum):
                enum_name = getattr(col.type, 'name', None) or (getattr(col, 'name', 'unknown_enum') + '_enum')
                enum_class = getattr(col.type, 'enum_class', None)
                if enum_class:
                    enum_values = [repr(e.value) for e in enum_class]
                    ddl_statements.append(f"CREATE TYPE {enum_name} AS ENUM ({', '.join(enum_values)})")

for table in Base.metadata.sorted_tables:
    ddl = str(CreateTable(table).compile(dialect=_ddl_dialect))
    # For Oracle, add CHECK constraint for enum columns
    if 'DIALECT_PLACEHOLDER'.lower() == 'oracle':
        for col in table.columns:
            enum_class = getattr(col.type, 'enum_class', None)
            if enum_class:
                enum_values = [repr(e.value) for e in enum_class]
                col_name = col.name
                # Try both quoted and unquoted column names
                quoted_col = f'"{col_name}"'
                unquoted_col = col_name

                # Search for column definition (quoted or unquoted)
                search_patterns = [
                    (f'{quoted_col} VARCHAR', quoted_col),  # Column is quoted in DDL
                    (f'{unquoted_col} VARCHAR', unquoted_col)  # Column is unquoted in DDL
                ]

                constraint_applied = False
                for search_str, check_col_ref in search_patterns:
                    if search_str in ddl:
                        # Work line-by-line so we only modify the correct column definition
                        ddl_lines = ddl.splitlines()
                        for i, line in enumerate(ddl_lines):
                            if search_str in line:
                                # Use the same column reference style (quoted/unquoted) in CHECK
                                check_constraint = f' CHECK ({check_col_ref} IN ({", ".join(enum_values)}))'
                                if 'NOT NULL' in line:
                                    # Insert the CHECK constraint immediately after NOT NULL on this line
                                    nn_idx = line.index('NOT NULL') + len('NOT NULL')
                                    new_line = line[:nn_idx] + check_constraint + line[nn_idx:]
                                else:
                                    # Nullable column: insert before trailing comma if present, otherwise at end
                                    comma_idx = line.rfind(',')
                                    if comma_idx != -1:
                                        new_line = line[:comma_idx] + check_constraint + line[comma_idx:]
                                    else:
                                        new_line = line + check_constraint
                                ddl_lines[i] = new_line
                                ddl = "\\n".join(ddl_lines)
                                constraint_applied = True
                                break
                        # Once we've applied the constraint using this pattern, stop trying others
                        break
                if not constraint_applied:
                    import sys as _sys
                    print(f"WARNING: Could not inject CHECK constraint for enum column '{col_name}' in table '{table.name}'", file=_sys.stderr)
    ddl_statements.append(ddl)

with open(sql_file_path, "w") as f:
    for stmt in ddl_statements:
        f.write(f"{stmt};\\n\\n")
'''
            content += ddl_dump_code.replace(
                "DIALECT_URL_PLACEHOLDER", DDL_DIALECT_URLS[dialect]
            ).replace("DIALECT_PLACEHOLDER", dialect)

            # Write back modified file
            with open(temp_py_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Run the temporary Python file (in the temp dir, so it leaves nothing behind)
            result = subprocess.run(
                [sys.executable, temp_py_path], text=True, capture_output=True, cwd=temp_dir, check=False
            )
            src_sql = os.path.join(temp_dir, f"tables_{self.sql_dialect}.sql")
            if result.returncode != 0 or not os.path.isfile(src_sql):
                details = (result.stderr or result.stdout or "").strip() or "no output"
                raise RuntimeError(
                    f"SQL generation failed for dialect '{self.sql_dialect}': the DDL script "
                    f"exited with code {result.returncode}.\n{details}"
                )

            # Copy generated .sql file to final output location
            dst_sql = os.path.join(final_output_dir, f"tables_{self.sql_dialect}.sql")
            shutil.copyfile(src_sql, dst_sql)
