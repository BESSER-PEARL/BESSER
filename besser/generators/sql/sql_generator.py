import subprocess
import os
import sys
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.generators.sql_alchemy import SQLAlchemyGenerator

class SQLGenerator(GeneratorInterface):
    """
    SQLGenerator is a class that implements the GeneratorInterface and produces the code or set of SQL statements 
    used to define and modify the structure of the tables in a database.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
        sql_dialect (str, optional): The SQL dialect. Values allowed: "sqlite", "postgresql", "mysql",
        "mssql", or "mariadb". Defaults to "sqlite".
    """

    def __init__(self, model: DomainModel, output_dir: str = None, sql_dialect: str = "sqlite"):
        super().__init__(model, output_dir)
        self.sql_dialect = sql_dialect

    def generate(self):
        """
        Generates SQL code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named tables.sql.
        """
        file_path = self.build_generation_path(file_name="sql_alchemy.py")
        alchemy_gen = SQLAlchemyGenerator(model=self.model, output_dir=self.output_dir)
        alchemy_gen.generate(dbms=self.sql_dialect)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        # remove the last line
        lines = content.splitlines()
        lines = lines[:-1]
        content = "\n".join(lines)
        dialect = self.sql_dialect
        content += f"""

from sqlalchemy.schema import CreateTable
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
sql_file_path = os.path.join(current_directory, f"tables_{dialect}.sql")

ddl_statements = []

for table in Base.metadata.sorted_tables:
    ddl_statements.append(str(CreateTable(table).compile(engine)))

with open(sql_file_path, "w") as f:
    for stmt in ddl_statements:
        f.write(f"{{stmt}};\\n\\n")
"""
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)

        # Execute the generated file
        try:
            #subprocess.run(['python', file_path], check=True, text=True, capture_output=True)
            subprocess.run([sys.executable, file_path], check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the sql_alchemy file: {e.stderr}")

        # Remove sql_alchemy.py
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing the sql_alchemy file: {e}")
