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

import ast
import asyncio
import os
from types import SimpleNamespace

import pytest
from fastapi import Body, Depends, HTTPException

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Method, Parameter, PrimitiveDataType, Property,
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
        tmp_path, [Method(name="produceBill"), Method(name="cancel", parameters={
            Parameter(name="reason", type=PrimitiveDataType("str")),
        })],
    )
    for name in ("produceBill", "cancel"):
        # Exactly the shape the React generator emits for every method button.
        assert f'@router.post("/booking/{{booking_id}}/methods/{name}/"' in content, name
        assert f'"/booking/methods/{name}/"' not in content, name
    assert "booking_id: int" in content
    # Still honest stubs: an unimplemented method answers 501, never a fake success.
    assert content.count("status_code=501") == 2
    tree = ast.parse(content)
    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))
    assert "sys.stdout" not in content
    assert "StringIO" not in content
    assert "params: dict = Body(default=None, embed=True)" in content

    # Execute only generated handlers, without importing an app or creating a
    # database. Missing instances still give 404; existing ones give honest 501
    # even when an unimplemented method declares parameters but receives none.
    namespace = {"Body": Body, "Depends": Depends, "HTTPException": HTTPException,
                 "Session": object, "get_db": lambda: None,
                 "Booking": SimpleNamespace(id=1)}
    for handler in (node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)):
        handler.decorator_list = []
        exec(compile(ast.Module(body=[handler], type_ignores=[]), "<stub>", "exec"), namespace)
        for instance, expected in ((None, 404), (SimpleNamespace(id=1), 501)):
            database = SimpleNamespace(query=lambda _: SimpleNamespace(
                filter=lambda _: SimpleNamespace(first=lambda: instance)))
            with pytest.raises(HTTPException) as error:
                asyncio.run(namespace[handler.name](1, None, database))
            assert error.value.status_code == expected


def test_a_body_that_takes_self_is_unchanged(tmp_path):
    content = _booking_methods_router(
        tmp_path, [Method(name="cancel", code="def cancel(self):\n    return True\n")],
    )
    assert '@router.post("/booking/{booking_id}/methods/cancel/"' in content
    assert "status_code=501" not in content
    assert "async def wrapper(" in content
    assert "sys.stdout" in content
    assert any(isinstance(node, ast.Try) for node in ast.walk(ast.parse(content)))


def test_a_body_without_self_keeps_its_class_level_route(tmp_path):
    """Code that explicitly takes no instance stays class-level; only the
    inference from an EMPTY body changes."""
    content = _booking_methods_router(
        tmp_path, [Method(name="total", code="def total():\n    return 1\n")],
    )
    assert '@router.post("/booking/methods/total/"' in content
    assert "{booking_id}" not in content
    assert "def _total_impl(" in content
    assert "sys.stdout" in content
    compile(content, "<class-method>", "exec")
