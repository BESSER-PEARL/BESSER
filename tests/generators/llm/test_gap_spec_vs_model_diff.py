"""The gap analyser must diff the SPEC against the MODEL, not only the code.

Observed live on run ``ca48a6dd`` (2026-09-17). The user's hotel spec named
five booking actions, two independent status dimensions, four validity rules
and a price that belongs to the booking-room *link*. The modelling step
captured two methods, one merged enum, zero constraints and no link class.
The gap analyser then produced four tasks, all cosmetic:

    "Generated enhanced employee page with improved styling"
    "Added HotelNavigation component"
    "Created personalized Home page"
    "Improved Person page with clean card-based layout and Tailwind"

Nothing about the missing behaviour. The analyser is the only stage that
holds the request and the model side by side, so whatever it does not
notice there is lost for the rest of the run — Phase 2 works from the
checklist, and Phase 3 validates the code against the model.

The prompt used to say only "List the missing or incorrect work", which
invites a code-vs-model reading. These tests pin the spec-vs-model pass and
the framing that makes it load-bearing.
"""

from __future__ import annotations

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Method,
    Multiplicity,
    Property,
    StringType,
)
from besser.generators.llm import gap_analyzer
from besser.generators.llm.gap_analyzer import _SYSTEM_PROMPT, analyze_gaps_via_llm
from besser.generators.llm.llm_client import UsageTracker

# An abridged form of the live request, keeping every construct the model
# lost: five named actions, two status dimensions, prose rules, and a fact
# that belongs to the relationship.
SPEC = """
Build a web application to manage a small hotel.

A booking offers five actions: produce the bill, check the guest in, check
the guest out, cancel the booking, and compute the amount due.

A booking has a commercial status: awaiting payment, confirmed, or
cancelled. Separately it has a physical status: not arrived, checked in, or
checked out.

The total number of guests must not exceed the combined capacity of the
rooms booked. A room cannot be double-booked for overlapping dates. Email
and phone must be valid. The arrival date must not be after the departure
date.

For each room in a booking we record the price actually agreed for that
room in that booking, which may differ from the standard price.
"""


def _under_captured_model() -> DomainModel:
    """What the modelling step actually produced: 2 of 5 methods, the two
    status dimensions merged into one enum with invented members, no
    constraints, and no link class carrying the agreed price."""
    status = Enumeration(
        name="BookingStatus",
        literals={
            EnumerationLiteral(name="BOOKED"),
            EnumerationLiteral(name="CHECKED_IN"),
            EnumerationLiteral(name="CHECKED_OUT"),
            EnumerationLiteral(name="CANCELLED"),
            EnumerationLiteral(name="NO_SHOW"),
        },
    )

    booking = Class(name="Booking")
    booking.attributes = {Property(name="reference", type=StringType)}
    booking.methods = {Method(name="checkIn"), Method(name="calculateTotal")}

    room = Class(name="Room")
    room.attributes = {Property(name="number", type=StringType)}

    booked = Property(name="rooms", type=room, multiplicity=Multiplicity(1, "*"))
    booked_by = Property(name="bookings", type=booking, multiplicity=Multiplicity(0, "*"))
    assoc = BinaryAssociation(name="booking_room", ends={booked, booked_by})

    return DomainModel(
        name="Hotel",
        types={booking, room, status},
        associations={assoc},
    )


class _CapturingClient:
    """Records the planner prompt, then returns an empty task list."""

    model = "capture-model"

    def __init__(self):
        self.usage = UsageTracker("capture-model")
        self._client = object()  # makes the analyser treat it as real
        self.system = None
        self.prompt = None

    def chat(self, system, messages, tools):
        self.system = system
        self.prompt = messages[-1]["content"]
        return {
            "stop_reason": "end_turn",
            "content": [type("B", (), {"type": "text", "text": "[]"})()],
        }


@pytest.fixture
def planner_prompt() -> str:
    client = _CapturingClient()
    result = analyze_gaps_via_llm(
        instructions=SPEC,
        generator_used="generate_fastapi_backend",
        domain_model=_under_captured_model(),
        inventory="backend/main.py 1.2K\nfrontend/src/App.tsx 3.4K",
        llm_client=client,
    )
    assert result == [], "fixture expects the stubbed empty reply"
    assert client.prompt, "the planner was never called"
    return client.prompt


def test_prompt_carries_the_spec_and_the_model_side_by_side(planner_prompt):
    """The diff is only possible if both halves actually reach the model."""
    assert "produce the bill" in planner_prompt
    assert "awaiting payment" in planner_prompt
    assert "BookingStatus" in planner_prompt  # the model JSON
    assert "checkIn" in planner_prompt


def test_prompt_asks_for_a_spec_vs_model_pass(planner_prompt):
    """The regression this file exists for: the instruction used to be a
    bare 'list the missing or incorrect work', which reads as code-vs-model
    and produced four styling tasks against a spec with 12 real gaps."""
    low = planner_prompt.lower()
    assert "user request vs domain model" in low
    assert "does not appear is a gap" in low
    # The four categories the live run dropped, each named explicitly.
    assert "operations an entity must support" in low
    assert "status or state vocabularies" in low
    assert "rules, limits and validity conditions" in low
    assert "belong to a relationship" in low


def test_prompt_orders_spec_gaps_first(planner_prompt):
    """With a 16-task cap, ordering decides what survives truncation."""
    assert "Pass-1 gaps FIRST" in planner_prompt
    assert planner_prompt.index("PASS 1") < planner_prompt.index("PASS 2")


def test_two_status_dimensions_must_not_merge(planner_prompt):
    low = planner_prompt.lower()
    assert "two enumerations, not one" in low
    assert "never invented" in low


def test_system_prompt_denies_the_model_authority_over_the_spec():
    """The analyser has to be told the model is lossy; otherwise a model
    that omits a requirement reads as a requirement that does not exist."""
    assert "THE DOMAIN MODEL IS NOT THE SPEC" in _SYSTEM_PROMPT
    low = _SYSTEM_PROMPT.lower()
    assert "the user request is the authority" in low
    assert "you are the only step that sees" in low


def test_spec_fits_the_instruction_budget():
    """A clipped spec silently removes the requirements being diffed."""
    assert len(SPEC) < gap_analyzer._MAX_INSTRUCTIONS_CHARS
