"""A model keeps its name through the editor's JSON: the diagram title carries it.

``project_to_json`` titles each diagram after the model its section builds, and
``json_to_buml_project`` names each model after its diagram's title, so
B-UML -> JSON -> B-UML keeps the names and JSON -> B-UML -> JSON keeps the titles.
"""
import os
import runpy
import tempfile

import pytest

from besser.BUML.metamodel.project import Project
from besser.cli.starters import starter_source
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.models import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters import (
    json_to_buml_project,
    project_to_json,
)

# Every model kind the editor has a diagram for (``besser new --list``), as its starter builds it.
KINDS = ["domain", "object", "state_machine", "agent", "gui", "nn", "quantum", "bpmn", "uml_deployment",
         "component", "kg", "project"]


def _starter_models(kind, folder):
    # Run from a file, kept while the test runs: the code builder reads the source of a model's functions.
    path = folder / f"{kind}_starter.py"
    if not path.exists():
        path.write_text(starter_source(kind, "model.py"), encoding="utf-8")
    namespace = runpy.run_path(str(path))
    if isinstance(namespace.get("project"), Project):
        return list(namespace["project"].models)
    models, seen = [], set()
    for value in namespace.values():
        if type(value).__module__.startswith("besser.BUML.metamodel") and hasattr(value, "name") \
                and id(value) not in seen and type(value).__name__ in _MODEL_TYPES:
            seen.add(id(value))
            models.append(value)
    return models


_MODEL_TYPES = {"DomainModel", "ObjectModel", "StateMachine", "Agent", "GUIModel", "NN", "QuantumCircuit",
                "BPMNModel", "ComponentModel", "DeploymentModel", "KnowledgeGraph"}


def _to_code(models, name="Shop") -> str:
    with tempfile.TemporaryDirectory(prefix="besser_") as folder:
        path = os.path.join(folder, "project.py")
        project_to_code(Project(name=name, models=models), path)
        with open(path, encoding="utf-8") as handle:
            return handle.read()


def _to_models(project_json):
    return json_to_buml_project(ProjectInput.model_validate(project_json)).models


def _titles(project_json):
    return sorted(
        (diagram_type, diagram["title"])
        for diagram_type, diagrams in project_json["diagrams"].items()
        for diagram in diagrams
        if diagram["model"].get("elements") or diagram["model"].get("pages")
        or diagram["model"].get("nodes") or diagram["model"].get("cols") is not None
    )


def _names(models):
    return sorted((type(model).__name__, model.name) for model in models)


@pytest.mark.parametrize("kind", KINDS)
def test_buml_to_json_to_buml_keeps_every_model_name(kind, tmp_path):
    models = _starter_models(kind, tmp_path)
    assert models

    back = _to_models(project_to_json(_to_code(models)))

    assert _names(back) == _names(models)


@pytest.mark.parametrize("kind", KINDS)
def test_json_to_buml_to_json_keeps_every_diagram_title(kind, tmp_path):
    first = project_to_json(_to_code(_starter_models(kind, tmp_path)))

    second = project_to_json(_to_code(_to_models(first)))

    assert _titles(second) == _titles(first)
    assert {title for _, title in _titles(first)} == {model.name for model in _starter_models(kind, tmp_path)}


def test_a_model_named_as_the_default_title_gets_the_default_title():
    from besser.BUML.metamodel.structural import Class, DomainModel

    model = DomainModel(name="Class_Diagram", types={Class(name="Book")})

    project = project_to_json(_to_code([model]))

    assert [diagram["title"] for diagram in project["diagrams"]["ClassDiagram"]] == ["Class Diagram"]
    assert _names(_to_models(project)) == [("DomainModel", "Class_Diagram")]


def test_a_section_that_names_no_model_keeps_the_default_title():
    code = '''
####################
# STRUCTURAL MODEL #
####################
from besser.BUML.metamodel.structural import Class, DomainModel
Book = Class(name="Book")
NAME = "Hidden"
domain_model = DomainModel(NAME, types={Book})

######################
# PROJECT DEFINITION #
######################
from besser.BUML.metamodel.project import Project
project = Project(name="P", models=[domain_model], owner="")
'''
    project = project_to_json(code)

    assert [diagram["title"] for diagram in project["diagrams"]["ClassDiagram"]] == ["Class Diagram"]


def test_a_gui_diagram_title_names_the_gui_model(tmp_path):
    first = project_to_json(_to_code(_starter_models("gui", tmp_path)))
    first["diagrams"]["GUINoCodeDiagram"][0]["title"] = "Shop App"

    [gui] = [model for model in _to_models(first) if type(model).__name__ == "GUIModel"]

    assert gui.name == "Shop_App"
