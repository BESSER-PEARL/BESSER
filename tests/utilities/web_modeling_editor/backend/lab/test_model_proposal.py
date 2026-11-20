"""A plan's class diagram to the editor's JSON, and the project's class diagram back to a ``model_spec``."""

import copy

from besser.BUML.metamodel.structural import UNLIMITED_MAX_MULTIPLICITY
from besser.spec_driven_agent.lab.contracts import ChangeKind, Plan
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.lab.model_proposal import (
    domain_model_to_spec,
    model_spec_from_project,
    model_spec_to_domain_model,
    proposal_from_plan,
)

# A note element for the class diagram: a composition (Diagram is the whole), a plain association,
# an enumeration, an abstract class and a generalization.
SPEC = {
    "classes": [
        {"name": "Diagram", "attributes": [], "is_abstract": True},
        {"name": "Element", "attributes": [], "is_abstract": False},
        {"name": "Note", "attributes": [{"name": "color", "type": "Color"}, {"name": "text", "type": "str"}],
         "is_abstract": False},
    ],
    "enumerations": [{"name": "Color", "literals": ["BLUE", "RED"]}],
    "associations": [
        {"name": "annotates", "source": "Note", "target": "Element", "source_multiplicity": "0..*",
         "target_multiplicity": "0..1", "composition": False},
        {"name": "notes", "source": "Note", "target": "Diagram", "source_multiplicity": "0..*",
         "target_multiplicity": "1", "composition": True},
    ],
    "generalizations": [{"child": "Note", "parent": "Element"}],
}


def _plan(spec=None) -> Plan:
    return Plan(kind=ChangeKind.METAMODEL, title="Add a note element", summary="Notes on class diagrams.",
                steps=["Add the metaclass"], repos=["besser"], model_spec=copy.deepcopy(spec))


CHANGE_ID = "abcdef012345"


def _project(model: dict, *others: dict, active: int = 0) -> ProjectInput:
    """A project whose first class diagram is the proposal of CHANGE_ID, followed by ``others``."""
    proposal = {"id": "d1", "title": "Notes", "model": model, "config": {"labProposalChangeId": CHANGE_ID}}
    return ProjectInput.model_validate({
        "id": "p1", "type": "Project", "name": "Lab", "createdAt": "2026-11-19T00:00:00Z",
        "currentDiagramIndices": {"ClassDiagram": active},
        "diagrams": {"ClassDiagram": [proposal, *others]},
    })


def _library(diagram_id: str = "d2", class_name: str = "Library", **config) -> dict:
    """A class diagram with one class: the user's own model, or a kept proposal with ``config``."""
    spec = {"classes": [{"name": class_name, "attributes": [], "is_abstract": False}]}
    model = proposal_from_plan(_plan(spec))["model"]
    return {"id": diagram_id, "title": class_name, "model": model, **({"config": config} if config else {})}


def _elements_by_name(model: dict) -> dict:
    return {e["name"]: e for e in model["elements"].values() if e.get("name")}


def test_a_plan_model_becomes_an_editor_class_diagram():
    proposal = proposal_from_plan(_plan(SPEC))

    assert proposal["title"] == "Add a note element"
    assert proposal["diagramType"] == "ClassDiagram"
    model = proposal["model"]
    assert model["type"] == "ClassDiagram"
    elements = _elements_by_name(model)
    assert {"Diagram", "Element", "Note", "Color"} <= set(elements)
    assert elements["Diagram"]["type"] == "AbstractClass"
    assert elements["Color"]["type"] == "Enumeration"


def test_the_whole_of_a_composition_is_the_target_in_the_editor():
    model = proposal_from_plan(_plan(SPEC))["model"]
    ids = {element_id: e.get("name") for element_id, e in model["elements"].items()}

    compositions = [r for r in model["relationships"].values() if r["type"] == "ClassComposition"]

    assert len(compositions) == 1
    assert ids[compositions[0]["target"]["element"]] == "Diagram"
    assert ids[compositions[0]["source"]["element"]] == "Note"


def test_the_reviewed_diagram_comes_back_as_the_same_spec():
    """Plan -> editor JSON -> the project the editor sends back -> model_spec: nothing is lost or turned."""
    proposal = proposal_from_plan(_plan(SPEC))

    assert model_spec_from_project(_project(proposal["model"]), CHANGE_ID) == SPEC


def test_the_users_own_class_diagram_is_never_the_changes_model():
    """The active diagram is the user's application model: the change's proposal is read, wherever it is."""
    proposal = proposal_from_plan(_plan(SPEC))["model"]

    assert model_spec_from_project(_project(proposal, _library(), active=1), CHANGE_ID) == SPEC
    only_users = ProjectInput.model_validate({
        "id": "p1", "type": "Project", "name": "Lab", "createdAt": "2026-11-19T00:00:00Z",
        "diagrams": {"ClassDiagram": [_library("d1"), _library("d2", labProposalChangeId="0123456789ab")]}})
    assert model_spec_from_project(only_users, CHANGE_ID) is None


def test_a_kept_proposal_is_still_the_changes_model_and_the_active_one_wins():
    """Kept (``labChangeId``) and then proposed again: the user's active tab of the change is the model."""
    proposal = proposal_from_plan(_plan(SPEC))["model"]
    kept = _library("d2", "Memo", labChangeId=CHANGE_ID)

    assert model_spec_from_project(_project(proposal, kept, active=1), CHANGE_ID)["classes"][0]["name"] == "Memo"
    # Neither is active (the user's own model is): the last one of the change.
    assert model_spec_from_project(_project(proposal, kept, _library("d3"), active=2),
                                   CHANGE_ID)["classes"][0]["name"] == "Memo"


def test_a_star_multiplicity_reads_back_as_zero_to_many():
    spec = copy.deepcopy(SPEC)
    spec["associations"][0]["source_multiplicity"] = "*"

    domain_model = model_spec_to_domain_model(spec, "x")

    ends = {end.type.name: end for a in domain_model.associations if a.name == "annotates" for end in a.ends}
    assert (ends["Note"].multiplicity.min, ends["Note"].multiplicity.max) == (0, UNLIMITED_MAX_MULTIPLICITY)
    assert domain_model_to_spec(domain_model)["associations"][0]["source_multiplicity"] == "0..*"


def test_two_associations_between_the_same_classes_get_distinct_end_names():
    spec = copy.deepcopy(SPEC)
    spec["associations"].append({"name": "pinned", "source": "Note", "target": "Element",
                                 "source_multiplicity": "1", "target_multiplicity": "1"})

    domain_model = model_spec_to_domain_model(spec, "x")

    note_ends = sorted(end.name for a in domain_model.associations for end in a.ends if end.type.name == "Element")
    assert note_ends == ["element", "element_1"]


def test_an_end_name_is_not_taken_twice_along_a_generalization():
    """C inherits P's ends: an association to C gets an end name distinct from the one to P."""
    spec = {"classes": [{"name": n, "attributes": []} for n in ("P", "C", "X")],
            "associations": [
                {"name": "toP", "source": "X", "target": "P", "source_multiplicity": "1", "target_multiplicity": "1"},
                {"name": "toC", "source": "X", "target": "C", "source_multiplicity": "1", "target_multiplicity": "1"}],
            "generalizations": [{"child": "C", "parent": "P"}]}

    domain_model = model_spec_to_domain_model(spec, "x")

    ends = {a.name: {end.type.name: end.name for end in a.ends} for a in domain_model.associations}
    assert ends["toP"]["X"] == "x" and ends["toC"]["X"] == "x_1"
    assert proposal_from_plan(_plan(spec)) is not None


def test_a_generalization_cycle_becomes_an_open_question():
    spec = {"classes": [{"name": n, "attributes": []} for n in ("A", "B", "C")],
            "generalizations": [{"child": "A", "parent": "B"}, {"child": "B", "parent": "C"},
                                {"child": "C", "parent": "A"}]}
    plan = _plan(spec)

    assert proposal_from_plan(plan) is None
    assert "form a cycle" in plan.questions[0]


def test_a_model_that_cannot_be_converted_becomes_an_open_question():
    spec = copy.deepcopy(SPEC)
    spec["classes"][2]["attributes"].append({"name": "size", "type": "Size"})
    plan = _plan(spec)

    assert proposal_from_plan(plan) is None
    assert len(plan.questions) == 1
    assert "could not be shown" in plan.questions[0] and "Size" in plan.questions[0]


def test_a_plan_without_a_model_proposes_none():
    plan = _plan(None)

    assert proposal_from_plan(plan) is None
    assert plan.questions == []


def test_a_project_without_classes_has_no_model_spec():
    empty = {"version": "3.0.0", "type": "ClassDiagram", "elements": {}, "relationships": {}}

    assert model_spec_from_project(_project(empty), CHANGE_ID) is None
    no_class_diagram = ProjectInput.model_validate({
        "id": "p1", "type": "Project", "name": "Lab", "createdAt": "2026-11-19T00:00:00Z", "diagrams": {}})
    assert model_spec_from_project(no_class_diagram, CHANGE_ID) is None
