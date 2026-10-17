import datetime

import pytest

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, BinaryAssociation, Multiplicity, StringType, DateType,
    Enumeration, EnumerationLiteral,
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
    """``Person`` with optional ``name``, ``gender`` and ``born``, and the ``is_child_of``
    self-association (``parents`` 0..*)."""
    gender = Enumeration("Gender", literals={EnumerationLiteral("FEMALE"), EnumerationLiteral("MALE")})
    person = Class("Person", attributes={
        Property("name", StringType, is_optional=True),
        Property("gender", gender, is_optional=True),
        Property("born", DateType, is_optional=True),
    })
    is_child_of = BinaryAssociation("is_child_of", ends={
        Property("person_1", person, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property("parents", person, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel("Genealogy", types={person, gender}, associations={is_child_of})


@pytest.fixture()
def object(model) -> ObjectModel:
    """``child`` with parents ``mom`` and ``dad``; ``mom`` has a gender and a birth date, and
    ``adam`` has no attribute values on purpose."""
    person = model.get_class_by_name("Person")
    gender = model.get_type_by_name("Gender")
    child = person("child").attributes(name="Child").build()
    mom = person("mom").attributes(name="Mom", gender=gender.FEMALE).build()
    dad = person("dad").attributes(name="Dad", born=datetime.date(1970, 5, 17)).build()
    adam = person("adam").build()
    child.parents = {mom, dad}
    return ObjectModel("Genealogy", objects={child, mom, dad, adam})


def test_to_json_keeps_information(object, model, tmp_path):
    code_file = tmp_path / "genealogy_objects.py"
    object_model_to_code(object, str(code_file))
    result = object_buml_to_json(code_file.read_text(), class_buml_to_json(model))

    names = object_names(result).values()
    attributes = object_attributes(result)
    links = object_links(result)
    # Information lost before this PR is below
    assert "adam" in names
    assert "mom.gender = FEMALE" in attributes
    assert "dad.born = 1970-05-17" in attributes
    assert "child -parents-> mom" in links
    assert "child -parents-> dad" in links
    # Information that was kept before this PR is below
    assert "child" in names
    assert "mom" in names
    assert "dad" in names
    assert "child.name = Child" in attributes
    assert "mom.name = Mom" in attributes
    assert "dad.name = Dad" in attributes
