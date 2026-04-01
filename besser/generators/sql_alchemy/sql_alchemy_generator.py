import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.utilities.utils import sort_by_timestamp
from besser.generators.structural_utils import (
    get_foreign_keys, get_sqlalchemy_types, get_ids, separate_classes,
    get_concrete_table_inheritance, used_enums_for_class,
)

class SQLAlchemyGenerator(GeneratorInterface):
    """
    SQLAlchemyGenerator is a class that implements the GeneratorInterface and is responsible for generating
    SQLAlchemy code based on B-UML models.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    VALID_DBMS = {"sqlite", "postgresql", "mysql", "mssql", "mariadb", "oracle"}

    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)
        self.TYPES = get_sqlalchemy_types(model)

    def get_ids(self):
        return get_ids(self.model)

    def separate_classes(self):
        return separate_classes(self.model)

    def get_concrete_table_inheritance(self):
        return get_concrete_table_inheritance(self.model)

    def _create_env(self):
        """Create Jinja2 environment pointing to this generator's templates.

        Uses the same bare env as generate() so that shared templates
        (helpers.py.j2) and their {%- whitespace control work consistently.
        """
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        return Environment(loader=FileSystemLoader(templates_path))

    def _build_layered_context(self):
        """Build the template context for layered output."""
        classes, asso_classes = self.separate_classes()
        return {
            "classes": classes,
            "asso_classes": asso_classes,
            "types": self.TYPES,
            "ids": self.get_ids(),
            "fkeys": get_foreign_keys(self.model),
            "concrete_parents": self.get_concrete_table_inheritance(),
            "enumerations": list(self.model.get_enumerations()),
            "associations": list(self.model.associations),
            "sort": sort_by_timestamp,
        }

    def generate_layered(self, app_dir: str):
        """Generate per-entity SQLAlchemy model files into app_dir/models/.

        Args:
            app_dir: Path to the app/ directory where models/ will be created.
        """
        models_dir = os.path.join(app_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        env = self._create_env()
        ctx = self._build_layered_context()

        def _write(path, content):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)

        # _base.py — Base class
        _write(
            os.path.join(models_dir, "_base.py"),
            env.get_template("models_base.py.j2").render(),
        )

        # _enums.py — All enum definitions
        _write(
            os.path.join(models_dir, "_enums.py"),
            env.get_template("models_enums.py.j2").render(**ctx),
        )

        # Per-entity model files
        for class_obj in ctx["classes"]:
            used = used_enums_for_class(class_obj, ctx["enumerations"])
            _write(
                os.path.join(models_dir, f"{class_obj.name.lower()}.py"),
                env.get_template("model_entity.py.j2").render(
                    class_obj=class_obj, used_enums=used, **ctx
                ),
            )

        # Per-association-class model files
        for asso_class in ctx["asso_classes"]:
            used = used_enums_for_class(asso_class, ctx["enumerations"])
            _write(
                os.path.join(models_dir, f"{asso_class.name.lower()}.py"),
                env.get_template("model_asso_entity.py.j2").render(
                    asso_class=asso_class, used_enums=used, **ctx
                ),
            )

        # __init__.py — Association tables, imports, relationships
        _write(
            os.path.join(models_dir, "__init__.py"),
            env.get_template("models_init.py.j2").render(**ctx),
        )

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
                fkeys=get_foreign_keys(self.model),
                sort=sort_by_timestamp,
                concrete_parents=concrete_parents
            )
            f.write(generated_code)
            print("Code generated successfully!")
