"""Round-trip tests for GUI diagrams that bind to inherited domain attributes.

A GUI table column references a domain ``Property`` by the
``ClassName_attributeName`` convention the domain model builder uses. That
variable is named after the class which *declares* the attribute, so a table
bound to a subclass that inherits the attribute used to emit a name nothing
defined -- a ``Guest`` table showing ``Person.id`` wrote ``Guest_id`` -- and
re-importing the exported project failed with ``NameError``.
"""

import os
import tempfile

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import FieldColumn, Table
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    Generalization,
    IntegerType,
    Metadata,
    Method,
    MethodImplementationType,
    Parameter,
    Property,
    StringType,
)
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    project_to_json,
)


@pytest.fixture
def project_with_inherited_table():
    """A Guest table showing attributes Guest inherits from Person."""
    person = Class(name="Person")
    person_id = Property(name="id", type=IntegerType, is_id=True)
    person_name = Property(name="name", type=StringType)
    person.attributes = {person_id, person_name}

    guest = Class(name="Guest")  # declares nothing of its own

    domain_model = DomainModel(name="hotel")
    domain_model.types = {person, guest}
    domain_model.generalizations = {Generalization(general=person, specific=guest)}

    table = Table(
        name="guests",
        label="Guests",
        columns=[
            FieldColumn(label="Id", field=person_id),
            FieldColumn(label="Name", field=person_name),
        ],
        data_binding=DataBinding(domain_concept=guest),
    )
    screen = Screen(
        name="guest_screen",
        description="Guests",
        view_elements={table},
        is_main_page=True,
        route_path="/guests",
        screen_size="Medium",
    )
    gui_model = GUIModel(
        name="ui",
        package="com.example",
        versionCode="1",
        versionName="1.0",
        description="demo",
        modules={Module(name="MainModule", screens={screen})},
    )
    return Project(
        name="hotel_project",
        models=[domain_model, gui_model],
        metadata=Metadata(description="demo"),
    )


def _export(project) -> str:
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "project.py")
    project_to_code(project, path)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_inherited_field_is_named_after_the_declaring_class(project_with_inherited_table):
    """The column must reference Person_id, the variable that actually exists."""
    code = _export(project_with_inherited_table)

    assert "field=Person_id" in code
    assert "field=Person_name" in code
    # The binding class must not be used for an attribute it does not declare:
    # no such variable is ever emitted by the domain model builder.
    assert "Guest_id" not in code
    assert "Guest_name" not in code


def test_exported_project_with_inherited_field_reimports(project_with_inherited_table):
    """Exporting then importing must round-trip, not raise NameError."""
    code = _export(project_with_inherited_table)

    result = project_to_json(code)

    gui_entry = result["diagrams"]["GUINoCodeDiagram"]
    if isinstance(gui_entry, list):
        gui_entry = gui_entry[0]
    pages = gui_entry["model"]["pages"]
    assert pages, "GUI diagram came back empty"

    components = pages[0]["frames"][0]["component"]["components"]
    assert any(component.get("type") == "table" for component in components)


def test_gui_section_resolves_inherited_attribute_aliases():
    """Older exports spell an inherited field after the binding class.

    Those files are still readable: the GUI section is executed with the
    domain model in scope, and ``ClassName_attributeName`` is bound for every
    attribute a class exposes, inherited ones included. ``Guest_id`` and
    ``Person_id`` therefore denote the same Property rather than one of them
    being undefined.
    """
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
        _bind_domain_aliases,
    )

    person = Class(name="Person")
    person_id = Property(name="id", type=IntegerType, is_id=True)
    person.attributes = {person_id}
    guest = Class(name="Guest")
    domain_model = DomainModel(name="hotel")
    domain_model.types = {person, guest}
    domain_model.generalizations = {Generalization(general=person, specific=guest)}

    namespace = {"domain_model": domain_model}
    _bind_domain_aliases(namespace)

    assert namespace["Person_id"] is person_id
    assert namespace["Guest_id"] is person_id


def test_domain_method_parameters_do_not_collide_with_the_gui_parameter():
    """``Parameter`` names two different classes across the two metamodels.

    A domain method signature uses ``structural.Parameter(name=..., type=...)``
    while a GUI section uses ``gui.events_actions.Parameter``. Executing the
    domain context against the GUI vocabulary bound the wrong one and failed
    with ``Parameter.__init__() got an unexpected keyword argument 'type'``,
    so a project whose class diagram has any method with a typed parameter
    could not be imported alongside a GUI model.
    """
    book = Class(name="Book")
    stock = Property(name="stock", type=IntegerType)
    book.attributes = {stock}
    book.methods = {
        Method(
            name="decrease_stock",
            parameters={Parameter(name="qty", type=IntegerType)},
            implementation_type=MethodImplementationType.CODE,
        )
    }
    domain_model = DomainModel(name="library")
    domain_model.types = {book}

    table = Table(
        name="books",
        label="Books",
        columns=[FieldColumn(label="Stock", field=stock)],
        data_binding=DataBinding(domain_concept=book),
    )
    screen = Screen(
        name="book_screen",
        description="Books",
        view_elements={table},
        is_main_page=True,
        route_path="/books",
        screen_size="Medium",
    )
    gui_model = GUIModel(
        name="ui",
        package="com.example",
        versionCode="1",
        versionName="1.0",
        description="demo",
        modules={Module(name="MainModule", screens={screen})},
    )
    project = Project(
        name="library_project",
        models=[domain_model, gui_model],
        metadata=Metadata(description="demo"),
    )

    result = project_to_json(_export(project))

    gui_entry = result["diagrams"]["GUINoCodeDiagram"]
    if isinstance(gui_entry, list):
        gui_entry = gui_entry[0]
    assert gui_entry["model"]["pages"], "GUI diagram came back empty"
