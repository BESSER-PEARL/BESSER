"""The ids and timestamps project_to_json makes up: fresh for the editor, fixed when asked.

The editor's import (``/get-project-json-model``) calls ``project_to_json(content)``
and gets new ids and the current time on every call, as before. ``besser convert``
and ``besser new`` pass ``id_seed`` and ``timestamp`` and get the same JSON for the
same model.
"""
import re
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from besser.utilities.web_modeling_editor.backend.models import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters import (
    json_to_buml_project,
    project_to_json,
    reproducible_project_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.minted_values import new_id

PROJECT = '''
####################
# STRUCTURAL MODEL #
####################
from besser.BUML.metamodel.structural import (
    Class, Property, DomainModel, BinaryAssociation, Multiplicity, StringType,
)
Library = Class(name="Library")
Book = Class(name="Book")
Book_title: Property = Property(name="title", type=StringType)
Book.attributes = {Book_title}
holds: BinaryAssociation = BinaryAssociation(name="holds", ends={
    Property(name="library", type=Library, multiplicity=Multiplicity(1, 1)),
    Property(name="books", type=Book, multiplicity=Multiplicity(0, 9999)),
})
domain_model = DomainModel(name="Shop", types={Library, Book}, associations={holds})

######################
# STATE MACHINE MODEL #
######################
from besser.BUML.metamodel.state_machine.state_machine import StateMachine
sm = StateMachine(name='Orders')
draft = sm.new_state(name='draft', initial=True)
done = sm.new_state(name='done')

######################
# PROJECT DEFINITION #
######################
from besser.BUML.metamodel.project import Project
project = Project(name="Shop", models=[domain_model, sm], owner="me")
'''
FIXED_TIME = "1970-01-01T00:00:00+00:00"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _ids(project):
    """The project id, the diagram ids and the element and relationship ids, in order."""
    ids = [project["id"]]
    for diagrams in project["diagrams"].values():
        for diagram in diagrams:
            ids.append(diagram["id"])
            model = diagram["model"]
            ids.extend(model.get("elements", {}) if isinstance(model.get("elements"), dict) else [])
            ids.extend(model.get("relationships", {}) if isinstance(model.get("relationships"), dict) else [])
    return ids


def _times(project):
    return [project["createdAt"]] + [diagram["lastUpdate"] for diagrams in project["diagrams"].values()
                                     for diagram in diagrams]


def test_by_default_every_conversion_mints_fresh_ids_and_the_current_time():
    before = datetime.now(timezone.utc)
    first, second = project_to_json(PROJECT), project_to_json(PROJECT)

    assert set(_ids(first)).isdisjoint(_ids(second))
    for minted in _ids(first):
        assert uuid.UUID(minted).version == 4
    for time in _times(first):
        assert before - timedelta(seconds=1) <= datetime.fromisoformat(time) <= datetime.now(timezone.utc)


def test_with_a_seed_and_a_timestamp_the_same_content_gives_the_same_json():
    first = project_to_json(PROJECT, id_seed="abc", timestamp=FIXED_TIME)
    second = project_to_json(PROJECT, id_seed="abc", timestamp=FIXED_TIME)

    assert first == second
    assert set(_times(first)) == {FIXED_TIME}
    ids = _ids(first)
    assert len(ids) == len(set(ids)) and all(UUID.fullmatch(minted) for minted in ids)
    assert set(ids).isdisjoint(_ids(project_to_json(PROJECT, id_seed="xyz", timestamp=FIXED_TIME)))


def test_fixed_values_end_with_the_call():
    project_to_json(PROJECT, id_seed="abc", timestamp=FIXED_TIME)

    assert uuid.UUID(new_id()).version == 4
    assert project_to_json(PROJECT)["createdAt"] != FIXED_TIME


def test_the_fixed_json_is_one_the_editor_backend_reads():
    project = project_to_json(PROJECT, id_seed="abc", timestamp=FIXED_TIME)

    loaded = json_to_buml_project(ProjectInput.model_validate(project))

    assert sorted(type(model).__name__ for model in loaded.models) == ["DomainModel", "StateMachine"]


    # The same models written another way: a comment, an annotation left out.
    # The same models in another form: other variable names, a comment, other spacing.
    reworded = PROJECT.replace("holds: BinaryAssociation", "# shelves\nholds").replace(
        "Book_title: Property = Property", "Book_title = Property")
    assert reworded != PROJECT

    first = reproducible_project_to_json(PROJECT, timestamp=FIXED_TIME)

    assert reproducible_project_to_json(reworded, timestamp=FIXED_TIME) == first
    other = reproducible_project_to_json(PROJECT.replace('name="Book"', 'name="Novel"'), timestamp=FIXED_TIME)
    assert set(_ids(other)).isdisjoint(_ids(first))


def test_an_id_a_model_carries_shifts_no_derived_id_outside_its_diagram():
    # A class with its own id takes no derived id: the other diagrams and the project keep theirs.
    with_id = PROJECT.replace('Library = Class(name="Library")', 'Library = Class(name="Library", id="lib-1")')

    plain = project_to_json(PROJECT, id_seed="abc", timestamp=FIXED_TIME)
    carried = project_to_json(with_id, id_seed="abc", timestamp=FIXED_TIME)

    assert carried["id"] == plain["id"]
    assert [d["id"] for d in carried["diagrams"]["StateMachineDiagram"]] == \
        [d["id"] for d in plain["diagrams"]["StateMachineDiagram"]]
    assert "lib-1" in carried["diagrams"]["ClassDiagram"][0]["model"]["elements"]


@pytest.mark.parametrize("arguments", [{"id_seed": "abc"}, {"timestamp": FIXED_TIME}])
def test_ids_and_timestamps_are_fixed_together(arguments):
    with pytest.raises(ValueError, match="both or neither"):
        project_to_json(PROJECT, **arguments)


def test_the_editors_import_endpoint_still_mints_fresh_ids_and_times(client):
    def import_project():
        response = client.post(
            "/besser_api/get-project-json-model",
            files={"buml_file": ("shop.py", PROJECT.encode("utf-8"), "text/plain")},
        )
        assert response.status_code == 200
        return response.json()

    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    first, second = import_project(), import_project()

    assert set(_ids(first["project"])).isdisjoint(_ids(second["project"]))
    for time in _times(first["project"]) + [first["exportedAt"]]:
        assert datetime.fromisoformat(time) >= before
