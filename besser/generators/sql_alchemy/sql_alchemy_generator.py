import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel, AssociationClass
from besser.generators import GeneratorInterface
from besser.utilities import sort_by_timestamp

class SQLAlchemyGenerator(GeneratorInterface):
    """
    SQLAlchemyGenerator is a class that implements the GeneratorInterface and is responsible for generating
    SQLAlchemy code based on B-UML models.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
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

    VALID_DBMS = {"sqlite", "postgresql", "mysql", "mssql", "mariadb", "oracle"}

    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)
        # Add enums to TYPES dictionary
        for enum in model.get_enumerations():
            self.TYPES[enum.name] = f"Enum('{enum.name}')"

    def get_ids(self):
        """
        Returns a dictionary with the class names as keys and the id attributes as values.
        """
        ids_dict = {}
        for cls in self.model.get_classes():
            # First, look for the first attribute marked with is_id
            id_attr = next((attr.name for attr in cls.attributes if attr.is_id), None)

            # If no attribute with is_id is found, check for an attribute with name "id"
            if not id_attr:
                id_attr = next((attr.name for attr in cls.attributes if attr.name == "id"), None)

            # Only add to the dictionary if an is_id or "id" attribute is found
            if id_attr:
                ids_dict[cls.name] = id_attr

        return ids_dict

    def get_foreign_keys(self):
        """
        Returns a dictionary with the association names as keys and the class names that should hold the foreign
        keys as values.
        """
        fkeys = dict()

        for association in self.model.associations:
            ends = list(association.ends)  # Convert set to list

            # One-to-one
            if ends[0].multiplicity.max == 1 and ends[1].multiplicity.max == 1:
                fkeys[association.name] = [ends[0].type.name, ends[1].name]
                if ends[1].multiplicity.min == 0:
                    fkeys[association.name] = [ends[1].type.name, ends[0].name]

            # Many to one
            elif ends[0].multiplicity.max > 1 and ends[1].multiplicity.max <= 1:
                fkeys[association.name] = [ends[0].type.name, ends[1].name]

            elif ends[0].multiplicity.max <= 1 and ends[1].multiplicity.max > 1:
                fkeys[association.name] = [ends[1].type.name, ends[0].name]

        return fkeys

    def separate_classes(self):
        """
        Separates regular classes from association classes in the model.
        
        Returns:
            tuple: A tuple containing two lists (regular_classes, association_classes)
        """
        classes_list = self.model.classes_sorted_by_inheritance()
        classes = []
        asso_classes = []

        # Separate regular classes and association classes
        for class_item in classes_list:
            if isinstance(class_item, AssociationClass):
                asso_classes.append(class_item)
            else:
                classes.append(class_item)

        return classes, asso_classes

    def get_concrete_table_inheritance(self):
        """
        Determines if the model uses concrete table inheritance.

        Returns:
            list: A list of class parents that use concrete table inheritance.
        """
        concrete_parents = []
        for class_ in self.model.get_classes():
            if class_.is_abstract and not class_.parents() and not class_.association_ends():
                concrete_parents.append(class_.name)
        return concrete_parents

    def generate(self, dbms: str = "sqlite"):
        """
        Generates SQLAlchemy code based on the provided B-UML model and saves it to the specified 
        output directory.
        If the output directory was not specified, the code generated will be stored in the 
        <current directory>/output
        folder.

        Args:
            dbms (str, optional): The database management system to be used. Values allowed: 
            "sqlite", "postgresql", "mysql", "mssql", "mariadb", or "oracle". Defaults to "sqlite".

        Returns:
            None, but stores the generated code as a file named sql_alchemy.py 
        """
        if dbms not in self.VALID_DBMS:
            raise ValueError(f"Invalid DBMS. Valid options are {', '.join(self.VALID_DBMS)}.")

        classes, asso_classes = self.separate_classes()
        concrete_parents = self.get_concrete_table_inheritance()

        file_path = self.build_generation_path(file_name="sql_alchemy.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('sql_alchemy_template.py.j2')
        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(
                classes=classes,
                asso_classes=asso_classes,
                types=self.TYPES,
                associations=self.model.associations,
                enumerations=self.model.get_enumerations(),
                model_name=self.model.name,
                dbms=dbms,
                ids=self.get_ids(),
                fkeys=self.get_foreign_keys(),
                sort=sort_by_timestamp,
                concrete_parents=concrete_parents
            )
            f.write(generated_code)
            print("Code generated successfully!")
