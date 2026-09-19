"""A derived attribute must not appear in backend *Create* schemas.

A modeled ``is_derived=True`` attribute (e.g. ``Booking.totalPrice``, computed
server-side from the agreed prices of the booked rooms) is a value the client
must never supply. Before this fix, ``is_server_owned_attribute()`` checked
only the surrogate ``id`` and the audit timestamps -- it never looked at
``is_derived`` -- so the ``*Create`` schema declared the derived attribute as
a required field, forcing the client to invent a value for something the
system is supposed to calculate. This was measured across 15 live hotel-app
runs: ``BookingCreate`` required ``totalPrice``/``physicalStatus``/
``commercialStatus`` in all 15.

The full (read) schema keeps the field, and a non-derived attribute is
unaffected either way -- mirrors the audit-timestamp coverage in
``test_server_owned_timestamps.py``.
"""

import os
import tempfile

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, IntegerType, StringType, FloatType,
)
from besser.generators.pydantic_classes import PydanticGenerator


def _generate(model, backend):
    out = tempfile.mkdtemp()
    PydanticGenerator(model=model, output_dir=out, backend=backend).generate()
    with open(os.path.join(out, "pydantic_classes.py"), encoding="utf-8") as f:
        return f.read()


def _class_block(code, class_name):
    start = code.find(f"class {class_name}(")
    assert start != -1, f"{class_name} not found in generated code"
    end = code.find("\n\n", start)
    return code[start:end if end != -1 else None]


def _booking_model():
    booking = Class(name="Booking", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="reference", type=StringType),
        Property(name="totalPrice", type=FloatType, is_derived=True),
    })
    return DomainModel(name="M", types={booking})


def test_backend_create_excludes_derived_attribute():
    """FAILS before the fix: is_server_owned_attribute() ignored is_derived,
    so BookingCreate declared `totalPrice: float` as a required field."""
    code = _generate(_booking_model(), backend=True)
    block = _class_block(code, "BookingCreate")
    assert "reference" in block  # non-derived attribute stays
    assert "id" in block  # declared is_id PK stays client-supplied
    assert "totalPrice" not in block  # derived attribute excluded


def test_full_schema_keeps_derived_attribute():
    """The read/full model must still expose the derived field -- this
    generator's `backend=False` mode is what test_server_owned_timestamps.py
    calls the read model (no `Create` suffix, all attributes present)."""
    code = _generate(_booking_model(), backend=False)
    block = _class_block(code, "Booking")
    assert "totalPrice" in block
    assert "reference" in block
