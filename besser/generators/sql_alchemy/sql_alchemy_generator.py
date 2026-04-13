import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel, AssociationClass
from besser.generators import GeneratorInterface
from besser.utilities.utils import sort_by_timestamp
from besser.generators.structural_utils import get_foreign_keys

class SQLAlchemyGenerator(GeneratorInterface):
    """
    SQLAlchemyGenerator is a class that implements the GeneratorInterface and is responsible for generating
    SQLAlchemy code based on B-UML models.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    TYPES = {
        "int": "Integer_",
        "str": "String_(100)",
        "float": "Float_",
        "bool": "Boolean_",
        "time": "Time_",
        "date": "Date_",
        "datetime": "DateTime_",
    }

    VALID_DBMS = {"sqlite", "postgresql", "mysql", "mssql", "mariadb", "oracle"}
    
    # Reserved names that conflict with template imports and definitions.
    # These names cannot be used for classes, enumerations, attributes, or associations.
    RESERVED_NAMES = {
        # Core template definitions
        "Base",              # Defined in template as DeclarativeBase subclass
        "Enum",              # SQLAlchemy Enum class (required by SQL generator's isinstance check)
        "enum",              # Python enum module imported at top of generated code
        
        # SQLAlchemy import aliases (underscore suffix)
        "Table_",           # sqlalchemy.Table
        "Column_",          # sqlalchemy.Column
        "ForeignKey_",      # sqlalchemy.ForeignKey
        "Mapped_",          # sqlalchemy.orm.Mapped
        
        # SQLAlchemy type aliases (underscore suffix)
        "Boolean_",         # sqlalchemy.Boolean
        "String_",          # sqlalchemy.String
        "Integer_",         # sqlalchemy.Integer
        "Float_",           # sqlalchemy.Float
        "Date_",            # sqlalchemy.Date
        "Time_",            # sqlalchemy.Time
        "DateTime_",        # sqlalchemy.DateTime
        "Text_",            # sqlalchemy.Text
        
        # Typing module aliases (underscore suffix)
        "List_",            # typing.List
        "Optional_",        # typing.Optional
    }

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

    def validate_model(self):
        """
        Validates that the model doesn't use reserved names for classes, enumerations, or attributes.
        
        Raises:
            ValueError: If any reserved names are found in the model.
        """
        conflicts = []
        
        # Check class names
        for cls in self.model.get_classes():
            if cls.name in self.RESERVED_NAMES:
                conflicts.append(f"Class name '{cls.name}' is reserved and cannot be used.")
            
            # Check attribute names within classes
            for attr in cls.attributes:
                if attr.name in self.RESERVED_NAMES:
                    conflicts.append(
                        f"Attribute name '{attr.name}' in class '{cls.name}' is reserved and cannot be used."
                    )
        
        # Check enumeration names
        for enum in self.model.get_enumerations():
            if enum.name in self.RESERVED_NAMES:
                conflicts.append(f"Enumeration name '{enum.name}' is reserved and cannot be used.")
        
        # Check association names
        for association in self.model.associations:
            if association.name in self.RESERVED_NAMES:
                conflicts.append(f"Association name '{association.name}' is reserved and cannot be used.")
        
        if conflicts:
            error_message = "SQLAlchemy code generation failed due to reserved name conflicts:\n" + "\n".join(
                f"  - {conflict}" for conflict in conflicts
            )
            error_message += (
                f"\n\nReserved names that cannot be used: {', '.join(sorted(self.RESERVED_NAMES))}."
            )
            raise ValueError(error_message)

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
        
        # Validate the model for reserved name conflicts
        self.validate_model()

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
                fkeys=get_foreign_keys(self.model),
                sort=sort_by_timestamp,
                concrete_parents=concrete_parents
            )
            f.write(generated_code)
            print("Code generated successfully!")
