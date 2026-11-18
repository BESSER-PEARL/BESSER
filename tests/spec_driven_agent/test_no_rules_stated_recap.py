"""A request that states no business rule says so, in one line the user sees.

Live evidence, r11-p11 (hotel bookings, "I want to create an application to
manage a hotel. To manage bookings."): the request states no rule, the
requirements ledger is empty, and the delivered app lets a room be booked twice
for the same night. The recipe said ``requirements: []`` and nothing else, so the
run read as fully checked.
"""

import json
import os

from besser.BUML.metamodel.structural import Class, Constraint, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.run_recap import NO_RULES_STATED, build_recap, next_run_notes, records_index

HOTEL = "I want to create an application to manage a hotel. To manage bookings."


class _Client:
    model = "mock"
    usage = UsageTracker("mock")

    def chat(self, **kwargs):
        return {"stop_reason": "end_turn", "content": []}


def _recipe(tmp_path, requirements, constrained=False):
    booking = Class(name="Booking")
    model = DomainModel(name="Hotel", types={booking, Class(name="Room")})
    if constrained:
        model.constraints = {Constraint(name="positiveGuests", context=booking, language="OCL",
                                        expression="context Booking inv positiveGuests: self.guests > 0")}
    tmp_path.mkdir(exist_ok=True)
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=model, output_dir=str(tmp_path))
    orch._requirements = requirements
    orch._save_recipe(HOTEL, 1.0)
    with open(os.path.join(str(tmp_path), ".besser_recipe.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_an_empty_ledger_tells_the_user_no_rules_were_checked(tmp_path):
    recipe = _recipe(tmp_path, [])
    assert recipe["business_rules_stated"] is False

    recap = build_recap(recipe)
    assert {"kind": "not_checked", "text": NO_RULES_STATED} in recap["open_items"]
    assert recap["open_item_count"] == 1
    assert NO_RULES_STATED in records_index(recap)
    # The next run's prompt does not get it: it must not read as work to do.
    assert NO_RULES_STATED not in next_run_notes(recipe)


def test_any_extracted_requirement_or_a_model_constraint_adds_nothing(tmp_path):
    """The extractor's kind labels differ between runs of one request (p8: all 'action'
    in r9 and r10, 'rule' in r11 and r12), so only an empty ledger says it."""
    ruled = _recipe(tmp_path / "a", [{"id": 1, "kind": "rule", "text": "A room cannot be booked twice"}])
    action = _recipe(tmp_path / "b", [{"id": 1, "kind": "action", "text": "The system flags aggressive accounts"}])
    screen = _recipe(tmp_path / "c", [{"id": 1, "kind": "ui", "text": "Show the bookings on a calendar"}])
    constrained = _recipe(tmp_path / "d", [], constrained=True)

    for recipe in (ruled, action, screen, constrained):
        assert recipe["business_rules_stated"] is True
        assert NO_RULES_STATED not in json.dumps(build_recap(recipe))


def test_a_ledger_that_did_not_run_claims_nothing(tmp_path):
    recipe = _recipe(tmp_path, None)
    assert recipe["business_rules_stated"] is None
    assert NO_RULES_STATED not in json.dumps(build_recap(recipe))


def test_the_records_note_stays_out_of_the_next_runs_prompt(tmp_path):
    """Review: the line sat under "Still open" in BESSER's records, and a modify run quotes
    them to the model through records_notes. It is a plain note there, and not quoted."""
    from besser.spec_driven_agent.run_recap import RECORDS_INDEX, records_notes

    recipe = _recipe(tmp_path / "app", [])
    index = records_index(build_recap(recipe))
    assert f"Note: {NO_RULES_STATED}." in index
    assert "Still open" not in index  # it was the only open item
    (tmp_path / "repo" / ".besser").mkdir(parents=True)
    (tmp_path / "repo" / RECORDS_INDEX).write_text(index, encoding="utf-8")
    notes = records_notes(str(tmp_path / "repo"))
    assert notes and "business rules" not in notes
