"""How a modify run's model relates to the one its base app was built from (#659).

Live second-run tests: with the same class names but changed attributes, the
modify run kept old and new attributes side by side and reported clean; told
nothing, it applied none of 5 attribute changes in 3/3 runs (hotel case m).
Given the full change list in the inventory and as a Phase 2 issue, it applied
all of them in 6/6 runs. Nothing compared the two models: the recipe stored
class names only.
"""
import json

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral, Multiplicity,
    PrimitiveDataType, Property, UNLIMITED_MAX_MULTIPLICITY,
)
from besser.spec_driven_agent.planning.model_change import (
    LARGE, SMALL, UNRELATED, compare_models, compare_with_base, describe_model_change,
    model_summary,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator

# The hotel base model of the live test, reduced: {class: {attribute: type}}.
HOTEL = {
    "Booking": {"arrivalDate": "date", "bookingNumber": "str", "departureDate": "date"},
    "Employee": {"position": "str"},
    "Person": {"email": "str", "familyName": "str", "firstName": "str", "phone": "str"},
    "Room": {"description": "str", "maxOccupancy": "int", "roomNumber": "str"},
}


def _domain(classes: dict, optional=(), handled_by_min=1, enums=None) -> DomainModel:
    """``classes``: {class: {attribute: type}}; ``optional``: {"Class.attr"}."""
    built = {}
    for name, attrs in classes.items():
        built[name] = Class(name=name, attributes={
            Property(name=a, type=PrimitiveDataType(t), is_optional=f"{name}.{a}" in optional)
            for a, t in attrs.items()
        })
    types = set(built.values())
    for enum_name, literals in (enums or {}).items():
        types.add(Enumeration(enum_name, literals={EnumerationLiteral(lit) for lit in literals}))
    associations = set()
    if "Booking" in built and "Employee" in built:
        associations.add(BinaryAssociation(name="handledBy", ends={
            Property(name="handledBy", type=built["Employee"],
                     multiplicity=Multiplicity(handled_by_min, 1)),
            Property(name="bookingsHandled", type=built["Booking"],
                     multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
        }))
    return DomainModel(name="Hotel", types=types, associations=associations)


def _changed(**edits) -> dict:
    classes = {c: dict(a) for c, a in HOTEL.items()}
    for cls, attrs in edits.items():
        classes[cls] = attrs
    return classes


def _summary(classes, **kw) -> dict:
    return model_summary(_domain(classes, **kw))


# The live case m, minus the attributes not needed here.
CASE_M = _changed(
    Booking={"arrivalDate": "date", "bookingNumber": "int", "departureDate": "date"},
    Person={"dateOfBirth": "date", "email": "str", "familyName": "str", "firstName": "str"},
    Room={"amenities": "str", "floor": "int", "maxOccupancy": "int", "roomNumber": "str"},
)


# ----------------------------------------------------------------------
# The change list
# ----------------------------------------------------------------------


def test_case_m_lists_every_change_with_types_and_association_ends():
    lines = describe_model_change(_summary(HOTEL), _summary(CASE_M, handled_by_min=0))
    assert lines == [
        "Booking.bookingNumber type str -> int",
        "Person.dateOfBirth: date added",
        "Person.phone removed",
        "Room.amenities: str added",
        "Room.floor: int added",
        "Room.description removed",
        # Ends named by their owning class: handledBy points at Employee and
        # belongs to Booking.
        "association handledBy: Booking.handledBy[1..1] - Employee.bookingsHandled[0..*]"
        " -> Booking.handledBy[0..1] - Employee.bookingsHandled[0..*]",
    ]


def test_optionality_changes_are_listed():
    old = _summary(HOTEL, optional={"Person.phone"})
    new = _summary(HOTEL, optional={"Person.email"})
    assert describe_model_change(old, new) == [
        "Person.email required -> optional",
        "Person.phone optional -> required",
    ]
    added = _changed(Room={**HOTEL["Room"], "floor": "int"})
    assert describe_model_change(old, _summary(added, optional={"Person.phone", "Room.floor"})) == [
        "Room.floor: int (optional) added",
    ]


def test_enumerations_classes_and_associations_added_or_removed():
    old = _summary(HOTEL, enums={"Status": ["OPEN", "CLOSED"], "Old": ["A"]})
    new_classes = {c: a for c, a in HOTEL.items() if c != "Employee"}
    new_classes["Invoice"] = {"total": "float"}
    new = _summary(new_classes, enums={"Status": ["OPEN", "PAID"], "Kind": ["X"]})
    assert describe_model_change(old, new) == [
        "class Invoice added",
        "class Employee removed",
        "enumeration Kind added: X",
        "enumeration Old removed",
        "enumeration Status literals CLOSED, OPEN -> OPEN, PAID",
        "association handledBy: Booking.handledBy[1..1] - Employee.bookingsHandled[0..*] -> removed",
    ]


def test_unchanged_model_has_no_change_lines():
    assert describe_model_change(_summary(HOTEL), _summary(HOTEL)) == []


# ----------------------------------------------------------------------
# Element ids: a rename reads as a rename
# ----------------------------------------------------------------------


def _with_ids(domain: DomainModel, ids: dict) -> DomainModel:
    """Set ``id`` on elements by name ("Class" / "Class.attr"), as the editor does."""
    for cls in domain.get_classes():
        if cls.name in ids:
            cls.id = ids[cls.name]
        for attr in cls.attributes:
            if f"{cls.name}.{attr.name}" in ids:
                attr.id = ids[f"{cls.name}.{attr.name}"]
    return domain


def test_renames_matched_by_id():
    base_ids = {"Room": "c-room", "Room.description": "a-desc", "Person": "c-person",
                "Booking": "c-booking", "Employee": "c-emp"}
    old = model_summary(_with_ids(_domain(HOTEL), base_ids))
    renamed = {c: dict(a) for c, a in HOTEL.items()}
    renamed["Guest"] = renamed.pop("Person")
    renamed["Room"]["amenities"] = renamed["Room"].pop("description")
    new_ids = {**base_ids, "Guest": "c-person", "Room.amenities": "a-desc"}
    new = model_summary(_with_ids(_domain(renamed), new_ids))

    assert old["classes"]["Room"]["attributes"]["description"]["id"] == "a-desc"
    assert describe_model_change(old, new) == [
        "class Person renamed to Guest",
        "Room.description renamed to Room.amenities",
    ]
    change = compare_models(old, new)
    assert change.verdict == SMALL and change.gone == 0
    assert change.removed == () and change.added == ()
    assert change.to_dict()["renamed"] == [
        {"from": "Person", "to": "Guest"},
        {"from": "Room.description", "to": "Room.amenities"},
    ]


def test_without_ids_the_same_rename_reads_as_remove_and_add():
    renamed = {c: dict(a) for c, a in HOTEL.items()}
    renamed["Room"]["amenities"] = renamed["Room"].pop("description")
    old, new = _summary(HOTEL), _summary(renamed)
    assert describe_model_change(old, new) == ["Room.amenities: str added", "Room.description removed"]
    change = compare_models(old, new)
    assert change.gone > 0 and change.removed == ("Room.description",)
    assert change.to_dict()["renamed"] == []


def test_an_id_on_one_side_only_falls_back_to_names():
    old = model_summary(_with_ids(_domain(HOTEL), {"Room": "c-room"}))
    assert describe_model_change(old, _summary(HOTEL)) == []


# ----------------------------------------------------------------------
# Verdict (thresholds from the large-change experiment, by name)
# ----------------------------------------------------------------------


def test_added_attribute_is_small():
    change = compare_models(_summary(HOTEL), _summary(_changed(Room={**HOTEL["Room"], "floor": "int"})))
    assert change.verdict == SMALL and change.gone == 0
    assert change.added == ("Room.floor",)


def test_name_only_class_changes_are_small():
    renamed = {c: dict(a) for c, a in HOTEL.items()}
    renamed["Customer"] = renamed.pop("Person")
    assert compare_models(_summary(HOTEL), _summary(renamed)).verdict == SMALL
    lowered = {c.lower(): a for c, a in HOTEL.items()}
    assert compare_models(_summary(HOTEL), _summary(lowered)).verdict == SMALL


def test_same_classes_everything_else_changed_is_large():
    new = {
        "Booking": {"checkIn": "date", "checkOut": "date", "reference": "str"},
        "Employee": {"role": "str"},
        "Person": {"email": "str", "givenName": "str", "lastName": "str"},
        "Room": {"capacity": "int", "label": "str", "number": "str"},
    }
    change = compare_models(_summary(HOTEL), _summary(new))
    assert change.verdict == LARGE
    assert "Room.description" in change.removed and "Room.label" in change.added


def test_unrelated_model():
    new = {"Book": {"isbn": "str", "title": "str"}, "Loan": {"dueDate": "date"},
           "Fine": {"amount": "float", "paid": "bool"}}
    assert compare_models(_summary(HOTEL), _summary(new)).verdict == UNRELATED


def test_deleting_classes_is_large_not_another_app():
    kept = {c: HOTEL[c] for c in ("Person", "Room")}
    assert compare_models(_summary(HOTEL), _summary(kept)).verdict == LARGE


def test_older_recipes():
    new = _summary(HOTEL)
    # Class names only (model-sync classes included): a verdict, no change list.
    assert compare_models({"classes": sorted(HOTEL)}, new).verdict == SMALL
    change = compare_models({"classes": sorted(HOTEL) + ["User"]}, new)
    assert change.basis == "classes" and change.removed == ("User",) and change.changes == ()
    # Malformed or empty: unknown.
    assert compare_models({"classes": "Room"}, new) is None
    assert compare_models({}, new) is None


def test_unknown_base_is_none(tmp_path):
    assert compare_with_base(str(tmp_path), None) is None
    assert compare_with_base(str(tmp_path), _domain(HOTEL)) is None  # no recipe


# ----------------------------------------------------------------------
# Wiring: the recipe records the user's model, a modify run lists the change
# ----------------------------------------------------------------------


class _Usage:
    estimated_cost = 0.0

    def summary(self):
        return {"api_calls": 0, "cost_usd": 0.0}


class _Client:
    model = "test-model"
    usage = _Usage()
    max_tokens = 4096

    def chat(self, system=None, messages=None, tools=None, **kwargs):
        from types import SimpleNamespace
        return {"stop_reason": "end_turn", "content": [SimpleNamespace(type="text", text="done")]}


def _orch(workspace, domain) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=domain, output_dir=str(workspace),
        enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False,
    )


def test_recipe_records_the_users_full_model(tmp_path):
    _orch(tmp_path, _domain(HOTEL, optional={"Person.phone"}))._save_recipe("build it", elapsed=1.0)
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    user_model = recipe["model"]["user_model"]
    assert user_model["classes"]["Person"]["attributes"]["phone"] == {"type": "str", "optional": True}
    assert user_model["associations"]["handledBy"]["ends"] == (
        "Booking.handledBy[1..1] - Employee.bookingsHandled[0..*]")
    assert compare_with_base(str(tmp_path), _domain(HOTEL, optional={"Person.phone"})).changes == ()


def _modify(tmp_path, monkeypatch, recorded: dict, new_domain) -> tuple[LLMOrchestrator, dict]:
    (tmp_path / ".besser_recipe.json").write_text(json.dumps({"model": recorded}), encoding="utf-8")
    orch = _orch(tmp_path, new_domain)
    seen = {}
    for name in ("_validate_phase1_output", "_run_phase3_validation", "_create_snapshot",
                 "_remove_snapshot"):
        monkeypatch.setattr(orch, name, lambda *a, **k: [])
    monkeypatch.setattr(orch, "_run_phase2",
                        lambda instr, extra_issues=None: seen.update(instr=instr, issues=extra_issues))
    monkeypatch.setattr(orch, "_save_recipe", lambda instr, elapsed: None)
    monkeypatch.setattr(orch._trace, "write", lambda event, **kw: seen.setdefault(event, kw))
    orch.modify("Update the app to the updated class diagram")
    return orch, seen


def test_modify_gets_the_change_list_in_the_inventory_and_as_a_phase2_issue(tmp_path, monkeypatch):
    orch, seen = _modify(tmp_path, monkeypatch, {"user_model": _summary(HOTEL)},
                         _domain(CASE_M, handled_by_min=0))
    note = seen["issues"][0]
    assert note.startswith("The class model changed since the last run. Model change: "
                           "Booking.bookingNumber type str -> int; Person.dateOfBirth: date added; "
                           "Person.phone removed;")
    assert "association handledBy: Booking.handledBy[1..1]" in note
    assert note.endswith("Update the code (generated and hand-written files) to this model change. "
                         "The new model wins: do not bring back removed or renamed elements.")
    assert f"## Model change since the last run\n{note}" in orch._inventory
    # Context, not the user's words.
    assert seen["instr"] == "Update the app to the updated class diagram"
    assert seen["run_start"]["model_change"]["changes"][0] == "Booking.bookingNumber type str -> int"


def test_modify_without_a_model_change_adds_nothing(tmp_path, monkeypatch):
    orch, seen = _modify(tmp_path, monkeypatch, {"user_model": _summary(HOTEL)}, _domain(HOTEL))
    assert "Model change" not in orch._inventory
    assert not any("Model change" in issue for issue in seen["issues"])
    assert seen["run_start"]["model_change"]["verdict"] == SMALL


def test_no_change_list_from_an_older_class_only_recipe(tmp_path, monkeypatch):
    # Such a recipe lists model-sync classes too; a missing one proves nothing.
    orch, seen = _modify(tmp_path, monkeypatch, {"classes": sorted(HOTEL) + ["User"]},
                         _domain(CASE_M))
    assert orch._model_change.removed == ("User",)
    assert "Model change" not in orch._inventory and orch._render_model_change() == ""


def test_model_sync_classes_are_not_reported_removed_next_run(tmp_path):
    # Run 2 ("add login") model-synced a User class into the run's model; the
    # editor never received it. Run 3 with the unchanged editor model must not
    # read User as removed and tell the agent to delete it.
    orch = _orch(tmp_path, _domain(HOTEL))
    orch._modify_mode = True
    assert orch._apply_new_classes([{"name": "User", "attributes": [{"name": "username", "type": "str"}]}]) == 1
    orch._save_recipe("add login", elapsed=1.0)

    change = compare_with_base(str(tmp_path), _domain(HOTEL))
    assert change.verdict == SMALL and change.removed == () and change.changes == ()
