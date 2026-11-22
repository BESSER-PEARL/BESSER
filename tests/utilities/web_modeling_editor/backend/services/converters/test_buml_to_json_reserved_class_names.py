"""Objects of classes whose names the code builder renames, such as ``Class`` (``Class_`` in code)."""
import pytest

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Multiplicity, Property, StringType,
)
from besser.utilities.buml_code_builder.domain_model_builder import object_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.object_diagram_converter import (
    object_buml_to_json,
)
from tests.utilities.web_modeling_editor.backend.services.converters.utils_buml_to_json_tests import (
    object_attributes, object_links, object_names,
)


@pytest.fixture()
def model() -> DomainModel:
    """``Class`` is the editor's default class name; ``Property`` is another name the generated
    script imports."""
    a_class = Class("Class", attributes={Property("title", StringType)})
    a_property = Class("Property", attributes={Property("key", StringType)})
    owns = BinaryAssociation("Owns", ends={
        Property("properties", a_property, multiplicity=Multiplicity(0, "*")),
        Property("owner", a_class, multiplicity=Multiplicity(1, 1)),
    })
    return DomainModel("Reserved", types={a_class, a_property}, associations={owns})


@pytest.fixture()
def objects(model) -> ObjectModel:
    first = model.get_class_by_name("Class")("first").attributes(title="a").build()
    colour = model.get_class_by_name("Property")("colour").attributes(key="c").build()
    colour.owner = first
    return ObjectModel("Reserved", objects={first, colour})


def test_objects_of_renamed_classes_are_kept(objects, model, tmp_path):
    code_file = tmp_path / "objects.py"
    object_model_to_code(objects, str(code_file))
    assert 'Class_("first")' in code_file.read_text()

    result = object_buml_to_json(code_file.read_text(), class_buml_to_json(model))

    # Every object of these classes used to be dropped: the code names the class Class_
    assert set(object_names(result).values()) == {"first", "colour"}
    assert "first.title = a" in object_attributes(result)
    assert "colour.key = c" in object_attributes(result)
    assert "colour -owner-> first" in object_links(result)
