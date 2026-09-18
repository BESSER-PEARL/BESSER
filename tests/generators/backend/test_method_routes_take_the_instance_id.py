"""A modeled method with no body is an instance operation; route it as one.

The backend template inferred "class-level method" from the absence of
``self`` in the method's code, so a method with NO code - the modeling agent's
default (implementationType "none"), and how a class diagram is normally drawn -
was registered at ``/booking/methods/cancel/`` with no id in the path. The React
generator addresses every method button as ``/booking/{booking_id}/methods/cancel/``
(serialization.py) and the GUI converter marks any run-method button wired to a
table as an instance method (component_parsers.py), so the UI could never reach
those routes. Live 2026-09-18, run 19h35 (hotel, 7 classes): all six modeled
methods had ``code: ''``, every method button 404ed, and 12 of 18 acceptance
scenarios were lost to that one mismatch. B-UML has no static methods - ``Method``
carries no such flag - so an unimplemented method is an unimplemented INSTANCE
method.
"""

import os

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Method, PrimitiveDataType, Property,
)
from besser.generators.backend import BackendGenerator


def _booking_methods_router(tmp_path, methods):
    booking = Class(name="Booking")
    booking.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    booking.methods = set(methods)
    model = DomainModel(name="Hotel", types={booking})
    BackendGenerator(model=model, output_dir=str(tmp_path)).generate()
    path = os.path.join(str(tmp_path), "routers", "booking_methods.py")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_a_body_less_method_is_routed_with_the_instance_id(tmp_path):
    content = _booking_methods_router(
        tmp_path, [Method(name="produceBill"), Method(name="cancel")],
    )
    for name in ("produceBill", "cancel"):
        # Exactly the shape the React generator emits for every method button.
        assert f'@router.post("/booking/{{booking_id}}/methods/{name}/"' in content, name
        assert f'"/booking/methods/{name}/"' not in content, name
    assert "booking_id: int" in content
    # Still honest stubs: an unimplemented method answers 501, never a fake success.
    assert content.count("status_code=501") == 2


def test_a_body_that_takes_self_is_unchanged(tmp_path):
    content = _booking_methods_router(
        tmp_path, [Method(name="cancel", code="def cancel(self):\n    return True\n")],
    )
    assert '@router.post("/booking/{booking_id}/methods/cancel/"' in content
    assert "status_code=501" not in content


def test_a_body_without_self_keeps_its_class_level_route(tmp_path):
    """Code that explicitly takes no instance stays class-level; only the
    inference from an EMPTY body changes."""
    content = _booking_methods_router(
        tmp_path, [Method(name="total", code="def total():\n    return 1\n")],
    )
    assert '@router.post("/booking/methods/total/"' in content
    assert "{booking_id}" not in content
