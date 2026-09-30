"""Two entities that each require the other cannot both be created.

Observed on a hotel model. Two symptoms with two different causes:

  1. ``BookingCreate`` required ``guest`` and ``GuestCreate`` required
     ``booking``, so neither could be created first. This needs only ONE
     association: the Pydantic template's FK-owning and non-owning branches
     were byte-identical, so the side that does not hold the foreign key
     still demanded one. The router then ignored that field entirely — it
     validates and assigns only the FK-owning end — so the client was forced
     to send an id that was silently discarded.

  2. ``sqlalchemy.exc.CircularDependencyError: Can't sort tables; there are
     unresolvable cycles between tables booking, guest`` at
     ``create_all``. This needs TWO associations with opposite FK
     ownership, since one association yields only one FK column. The app
     died before serving a request.

The second is fixed by ``get_deferred_fk_associations``: one edge of each
cycle becomes a nullable FK with ``use_alter`` and ``post_update``.
"""

from __future__ import annotations

import os
import sys

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Multiplicity,
    Property,
    StringType,
)
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.structural_utils import (
    get_deferred_fk_associations,
    get_foreign_keys,
)


def _booking_guest(second_association: bool) -> DomainModel:
    """Booking and Guest, each required to the other.

    With ``second_association`` the pair is linked twice, which is what the
    motivating model did: a booking's *contact* guest, plus the guests staying on
    the booking. That is what makes the FK ownership opposite and closes the
    table cycle.
    """
    booking = Class(name="Booking")
    booking.attributes = {Property(name="reference", type=StringType)}
    guest = Class(name="Guest")
    guest.attributes = {Property(name="fullName", type=StringType)}

    # Booking 1 -- 1 Guest, required on both ends.
    contact = Property(name="guest", type=guest, multiplicity=Multiplicity(1, 1))
    held_by = Property(name="booking", type=booking, multiplicity=Multiplicity(1, 1))
    associations = {BinaryAssociation(name="booking_guest", ends={contact, held_by})}

    if second_association:
        # Booking 1 -- 1..* Guest: the guests staying on the booking. This
        # puts the FK on Guest (the many end owns it), pointing back at
        # Booking, and required. Two symmetric 1..1s would NOT cycle: the
        # tiebreak in get_foreign_keys lands both on the alphabetically
        # first class, so ownership has to differ for a cycle to close.
        stays_on = Property(name="stayBooking", type=booking, multiplicity=Multiplicity(1, 1))
        guests = Property(name="guests", type=guest, multiplicity=Multiplicity(1, 9999))
        associations.add(
            BinaryAssociation(name="booking_stay", ends={stays_on, guests})
        )

    return DomainModel(
        name="Hotel", types={booking, guest}, associations=associations
    )


# ----------------------------------------------------------------------
# 1. The create-schema deadlock (one association is enough)
# ----------------------------------------------------------------------


def test_only_the_fk_owner_gets_a_create_field(tmp_path):
    model = _booking_guest(second_association=False)
    PydanticGenerator(model=model, output_dir=str(tmp_path), backend=True).generate()
    code = (tmp_path / "pydantic_classes.py").read_text(encoding="utf-8")

    owner = get_foreign_keys(model)["booking_guest"][0]
    other = "Guest" if owner == "Booking" else "Booking"

    owner_block = _class_block(code, f"{owner}Create")
    other_block = _class_block(code, f"{other}Create")

    # Exactly one side carries the relationship field.
    assert "1:1 Relationship" in owner_block
    assert "1:1 Relationship" not in other_block, (
        f"{other}Create still demands a link it does not own:\n{other_block}"
    )


def test_neither_create_schema_is_unsatisfiable(tmp_path):
    """The user-visible symptom: some entity must be creatable first."""
    model = _booking_guest(second_association=False)
    PydanticGenerator(model=model, output_dir=str(tmp_path), backend=True).generate()
    code = (tmp_path / "pydantic_classes.py").read_text(encoding="utf-8")

    required = {
        name: _required_relationship_fields(_class_block(code, f"{name}Create"))
        for name in ("Booking", "Guest")
    }
    assert not (required["Booking"] and required["Guest"]), (
        f"both create schemas require the other: {required}"
    )


# ----------------------------------------------------------------------
# 2. The table cycle (needs two associations)
# ----------------------------------------------------------------------


def test_no_cycle_no_deferral():
    """The common case must be left completely alone."""
    assert get_deferred_fk_associations(_booking_guest(False)) == set()


def test_cycle_is_detected_and_one_edge_deferred():
    model = _booking_guest(second_association=True)
    fkeys = get_foreign_keys(model)
    owners = {fkeys[name][0] for name in fkeys}
    assert owners == {"Booking", "Guest"}, (
        f"fixture does not close a cycle; FK owners were {owners}"
    )

    deferred = get_deferred_fk_associations(model)
    assert len(deferred) == 1, f"expected exactly one broken edge, got {deferred}"


def test_deferral_choice_is_stable_across_runs():
    """A generator that picked a different edge each run would rewrite
    unrelated files on every regeneration."""
    picks = {
        frozenset(get_deferred_fk_associations(_booking_guest(True)))
        for _ in range(5)
    }
    assert len(picks) == 1


def test_deferred_fk_is_nullable_with_use_alter(tmp_path):
    model = _booking_guest(second_association=True)
    SQLAlchemyGenerator(model=model, output_dir=str(tmp_path)).generate()
    code = (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")

    deferred = get_deferred_fk_associations(model)
    assert deferred
    assert "use_alter=True" in code, code
    assert "post_update=True" in code, code
    assert code.count("use_alter=True") == len(deferred)


def test_the_api_creation_flow_actually_works(tmp_path):
    """Import the generated models and drive the exact sequence the generated
    REST API performs: one entity per request, each setting only the foreign
    key its own create schema carries.

    Before the fix the very FIRST request died --
    ``IntegrityError: NOT NULL constraint failed: guest.stayBooking_id`` --
    because both tables demanded a link to the other, so no row could be
    inserted at all.

    Note what this does NOT claim: assigning both sides in a single flush
    still raises SQLAlchemy's CircularDependencyError. That is SQLAlchemy's
    documented behaviour for mutually dependent rows, and the generated API
    never does it -- each endpoint commits one entity.
    """
    import importlib.util

    model = _booking_guest(second_association=True)
    SQLAlchemyGenerator(model=model, output_dir=str(tmp_path)).generate()

    spec = importlib.util.spec_from_file_location(
        "generated_sql_alchemy", os.path.join(str(tmp_path), "sql_alchemy.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)

    with Session(engine) as session:
        guest = module.Guest(fullName="Ada")
        session.add(guest)
        session.commit()                       # POST /guest/
        assert guest.id is not None

        booking = module.Booking(reference="R1", guest_id=guest.id)
        session.add(booking)
        session.commit()                       # POST /booking/
        assert booking.id is not None

        guest.stayBooking_id = booking.id      # PATCH /guest/{id}
        session.commit()
        assert guest.stayBooking_id == booking.id

    sys.modules.pop("generated_sql_alchemy", None)


def test_the_deferred_column_is_the_one_that_unblocks_it(tmp_path):
    """Pin WHICH column went nullable, so a future change to the tie-break
    cannot silently move the deadlock somewhere else."""
    model = _booking_guest(second_association=True)
    SQLAlchemyGenerator(model=model, output_dir=str(tmp_path)).generate()
    code = (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")

    fk_lines = [
        line.strip() for line in code.splitlines()
        if "mapped_column(ForeignKey_" in line and "primary_key" not in line
    ]
    nullable = [line for line in fk_lines if "nullable=True" in line]
    assert len(nullable) == 1, fk_lines
    assert "stayBooking_id" in nullable[0]
    # The other FK keeps its NOT NULL: the fix breaks one edge, not both.
    assert any("guest_id" in line and "nullable=True" not in line for line in fk_lines)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _class_block(code: str, class_name: str) -> str:
    """The body of one generated class, up to the next top-level class."""
    marker = f"class {class_name}("
    start = code.index(marker)
    rest = code[start + len(marker):]
    end = rest.find("\nclass ")
    return rest if end == -1 else rest[:end]


def _required_relationship_fields(block: str) -> list:
    return [
        line.strip()
        for line in block.splitlines()
        if "Relationship" in line and "Optional[" not in line and ":" in line
    ]
