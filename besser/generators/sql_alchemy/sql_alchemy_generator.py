import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface

class SQLAlchemyGenerator(GeneratorInterface):
    """
    SQLAlchemyGenerator is a class that implements the GeneratorInterface and is responsible for generating
    SQLAlchemy code based on B-UML models.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        dbms (str, optional): The database management system to be used. Defaults to "sqlite".
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    TYPES = {
        "int": "Integer",
        "str": "String(100)",
        "float": "Float",
        "bool": "Boolean",
        "time": "Time",
        "date": "Date",
        "datetime": "DateTime",
    }

    VALID_DBMS = {"sqlite", "postgresql", "mysql"}

    def __init__(self, model: DomainModel, dbms: str = "sqlite", output_dir: str = None):
        super().__init__(model, output_dir)
        self.dbms: str = dbms
        # Add enums to TYPES dictionary
        for enum in model.get_enumerations():
            self.TYPES[enum.name] = f"Enum('{enum.name}')"

    @property
    def dbms(self):
        """Getter for the dbms attribute."""
        return self._dbms

    @dbms.setter
    def dbms(self, value: str):
        """Setter for the dbms attribute, validates the value."""
        if value not in self.VALID_DBMS:
            raise ValueError(f"Invalid DBMS. Valid options are {', '.join(self.VALID_DBMS)}.")
        self._dbms = value

    def generate(self):
        """
        Generates SQLAlchemy code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named sql_alchemy.py 
        """
        file_path = self.build_generation_path(file_name="sql_alchemy.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('sql_alchemy_template.py.j2')
        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(
                classes=self.model.classes_sorted_by_inheritance(),
                types=self.TYPES,
                associations=self.model.associations,
                enumerations=self.model.get_enumerations(),
                model_name=self.model.name,
                dbms=self.dbms
            )
            f.write(generated_code)
            print("Code generated in the location: " + file_path)
