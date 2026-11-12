"""A date, time or datetime slot is written back as a date, time or datetime.

``domain_model_to_code`` wrote every such value as ``datetime.datetime.fromisoformat``:
a date came back as a datetime (and the editor JSON then showed ``1965-08-01T00:00:00``),
and a time gave ``datetime.datetime.fromisoformat("10:30:00")``, which raises.
"""
import datetime
import runpy

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural import (
    Class, DateTimeType, DateType, DomainModel, Property, TimeType,
)
from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.models import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters import (
    json_to_buml_project,
    project_to_json,
)

VALUES = {
    "release": datetime.date(1965, 8, 1),
    "opens": datetime.time(10, 30),
    "updated": datetime.datetime(2024, 1, 2, 3, 4, 5),
}


def _models():
    event = Class(name="Event", attributes={
        Property(name="release", type=DateType),
        Property(name="opens", type=TimeType),
        Property(name="updated", type=DateTimeType),
    })
    domain = DomainModel(name="Events", types={event})
    launch = event("launch").attributes(**VALUES).build()
    return domain, ObjectModel(name="Launches", objects={launch})


def _slot_values(object_model):
    [obj] = object_model.objects
    return {slot.attribute.name: slot.value.value for slot in obj.slots}


def test_the_code_rebuilds_each_value_with_its_own_type(tmp_path):
    domain, objects = _models()
    path = tmp_path / "events.py"
    domain_model_to_code(domain, str(path), objectmodel=objects)

    rebuilt = runpy.run_path(str(path))["object_model"]

    values = _slot_values(rebuilt)
    assert values == VALUES
    assert {name: type(value) for name, value in values.items()} == {name: type(value)
                                                                     for name, value in VALUES.items()}


def test_a_date_and_a_time_keep_their_form_through_the_editor_json(tmp_path):
    path = tmp_path / "project.py"
    project_to_code(Project(name="P", models=list(_models())), str(path))

    project = project_to_json(path.read_text(encoding="utf-8"))
    [diagram] = project["diagrams"]["ObjectDiagram"]
    shown = sorted(element["name"] for element in diagram["model"]["elements"].values()
                   if element["type"] == "ObjectAttribute")
    [objects] = [model for model in json_to_buml_project(ProjectInput.model_validate(project)).models
                 if isinstance(model, ObjectModel)]

    assert shown == ["opens = 10:30:00", "release = 1965-08-01", "updated = 2024-01-02T03:04:05"]
    assert _slot_values(objects) == VALUES
