"""A modeled rule the generators cannot enforce becomes a Phase 2 task.

The pydantic generator declines OCL constraints that span relationships and
leaves ``# NOTE: OCL constraint '...' is not enforced by this Create model``
in the schema (pydantic_classes_template.py.j2). Nothing turned that marker
into work: on a recorded run the guest-capacity rule was in the model,
the planner happened to list it, and the app still shipped without it. The
rule is the user's own words in the model; the harness seeds it as a checklist
item so the end_turn gate holds until it is implemented or honestly dropped.
"""

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, Constraint, DomainModel, Multiplicity, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator


class _MockClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):
        raise AssertionError("never called")


def _hotel(constraint_expression: str, context_name: str = "Booking"):
    booking = Class(name="Booking")
    booking.attributes = {
        Property(name="id", type=PrimitiveDataType("int"), is_id=True),
        Property(name="guestCount", type=PrimitiveDataType("int")),
    }
    room = Class(name="Room")
    room.attributes = {
        Property(name="id", type=PrimitiveDataType("int"), is_id=True),
        Property(name="capacity", type=PrimitiveDataType("int")),
    }
    model = DomainModel(name="Hotel", types={booking, room})
    context = booking if context_name == "Booking" else room
    model.constraints = {Constraint(
        name="guestsWithinCapacity", context=context,
        expression=constraint_expression, language="OCL",
    )}
    return model


def _seeds(model, tmp_path, generator_used="generate_web_app"):
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=model,
                           output_dir=str(tmp_path))
    orch._generator_used = generator_used
    orch._instructions = "Build the hotel API."
    return [t["text"] if isinstance(t, dict) else t
            for t in orch._deterministic_gap_tasks()]


def test_a_relationship_spanning_rule_is_seeded_with_its_placement(tmp_path):
    model = _hotel(
        "context Booking inv guestsWithinCapacity: "
        "self.guestCount <= self.rooms->collect(capacity)->sum()"
    )
    seeds = _seeds(model, tmp_path)
    assert len(seeds) == 1, seeds
    text = seeds[0]
    assert "guestsWithinCapacity" in text
    assert "self.rooms->collect(capacity)->sum()" in text
    assert "routers/booking.py" in text
    assert "HTTP 400" in text


def test_a_rule_the_schema_already_enforces_is_not_seeded(tmp_path):
    """``self.guestCount >= 1`` becomes a Pydantic validator; seeding it too
    would send the model to re-implement what is already there."""
    model = _hotel("context Booking inv guestsWithinCapacity: self.guestCount >= 1")
    assert _seeds(model, tmp_path) == []


def test_no_seed_without_a_fastapi_scaffold(tmp_path):
    """The placement names a router file; with no scaffold there is none."""
    model = _hotel(
        "context Booking inv guestsWithinCapacity: "
        "self.guestCount <= self.rooms->collect(capacity)->sum()"
    )
    assert _seeds(model, tmp_path, generator_used=None) == []


def _library(expression: str, book_needs_library: bool = True, books_in_one_library: bool = True):
    """Library 1 -- 0..* Book; the Book's end is mandatory unless stated otherwise."""
    library = Class(name="Library")
    library.attributes = {Property(name="name", type=PrimitiveDataType("str"))}
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=PrimitiveDataType("str"))}
    holds = BinaryAssociation(name="Library_Book", ends={
        Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
        Property(name="library", type=library, multiplicity=Multiplicity(
            1 if book_needs_library else 0, 1 if books_in_one_library else "*")),
    })
    model = DomainModel(name="Catalogue", types={library, book}, associations={holds})
    model.constraints = {Constraint(name="books_at_least_1", context=library,
                                    expression=expression, language="OCL")}
    return model


@pytest.mark.parametrize("body", [
    "self.books->size() >= 1", "self.books->size() > 0", "self.books->notEmpty()",
    "self.books->size() >= 2", "1 <= self.books->size()", "0 < self.books->size()",
    "not self.books->isEmpty()", "not (self.books->isEmpty())", "self.books->isEmpty() = false",
    "(self.books->size() >= 1)", "self.books -> size ( ) >= 1",
])
def test_a_minimum_on_records_that_need_their_owner_first_is_not_a_refusal(tmp_path, body):
    """A Book needs its Library, so a new Library cannot have one yet: refusing
    the create, or the removal of a Book, leaves records nobody can create or delete."""
    seeds = _seeds(_library(f"context Library inv books_at_least_1: {body}"), tmp_path)
    assert len(seeds) == 1, seeds
    text = seeds[0]
    assert text.startswith("Report the modeled rule 'books_at_least_1'") and body in text
    assert ("never refuse creating or editing a Library, or deleting or moving one of its books, "
            "because of this minimum") in text
    assert "create and update endpoints" not in text


def test_a_later_step_that_needs_a_complete_record_may_still_refuse(tmp_path):
    """Only the steps that cannot meet the minimum are spared: a step that needs a
    complete record, such as submitting it, keeps its refusal."""
    text = _seeds(_library("context Library inv books_at_least_1: self.books->notEmpty()"), tmp_path)[0]
    assert "may refuse an incomplete one with HTTP 400 naming the rule" in text


def test_the_minimum_half_of_an_and_rule_is_reported_and_the_rest_enforced(tmp_path):
    seeds = _seeds(_library("context Library inv books_at_least_1: self.books->size() >= 1 and "
                            "self.books->forAll(b | b.title <> '')"), tmp_path)
    enforce, report = seeds
    assert enforce.startswith("Enforce the modeled rule 'books_at_least_1'")
    assert "OCL: self.books->forAll(b | b.title <> '') (the rest of the rule" in enforce
    assert "create and update endpoints of routers/library.py" in enforce and "HTTP 400" in enforce
    assert report.startswith("Report the modeled rule 'books_at_least_1'")


def test_an_exact_size_keeps_its_upper_bound_enforced(tmp_path):
    enforce, report = _seeds(_library("context Library inv books_at_least_1: self.books->size() = 3"), tmp_path)
    assert "OCL: self.books->size() <= 3 (the rest of the rule" in enforce and "HTTP 400" in enforce
    assert report.startswith("Report the modeled rule 'books_at_least_1'")


@pytest.mark.parametrize("expression, book_needs_library, books_in_one_library", [
    # The Books can exist first: the create can carry them, so the rule is enforceable.
    ("context Library inv books_at_least_1: self.books->size() >= 1", False, True),
    # A Book in several Libraries already has one when a Library is created: the create links it.
    ("context Library inv books_at_least_1: self.books->size() >= 1", True, False),
    # An upper bound, and no minimum at all, keep the refusal.
    ("context Library inv books_at_least_1: self.books->size() <= 5", True, True),
    ("context Library inv books_at_least_1: self.books->size() >= 0", True, True),
    # An alternative is no minimum on its own.
    ("context Library inv books_at_least_1: self.books->size() >= 1 or self.name = 'empty'", True, True),
])
def test_every_other_collection_rule_is_still_enforced(tmp_path, expression, book_needs_library,
                                                      books_in_one_library):
    seeds = _seeds(_library(expression, book_needs_library, books_in_one_library), tmp_path)
    assert len(seeds) == 1, seeds
    assert "create and update endpoints of routers/library.py" in seeds[0]
    assert "HTTP 400" in seeds[0]
