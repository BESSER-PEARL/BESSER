import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface


class SQLGenerator(GeneratorInterface):
    """
    SQLGenerator is a class that implements the GeneratorInterface and produces the code or set of SQL statements 
    used to define and modify the structure of the tables in a database..

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
        sql_dialect (str, optional): The SQL dialect. Values allowed: None, "postgres", or "mysql".
    """

    TYPES = {
        "int": "int",
        "str": "varchar(100)",
        "float": "float8",
        "bool": "bool",
        "time": "time",
        "date": "date",
        "datetime": "timestamp",
        "timedelta": "interval",
    }

    def __init__(self, model: DomainModel, output_dir: str = None, sql_dialect: str = None):
        super().__init__(model, output_dir)
        self.sql_dialect = sql_dialect

    def generate(self):
        """
        Generates SQL code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named sql_alchemy.py 
        """
        file_path = self.build_generation_path(file_name="tables.sql")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path), trim_blocks=True, lstrip_blocks=True)
        template = env.get_template('sql_template.sql.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(model=self.model, types=self.TYPES, sql_dialect=self.sql_dialect)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)
