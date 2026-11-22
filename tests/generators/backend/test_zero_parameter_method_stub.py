"""The stub for a zero-parameter method says so, in the file being edited.

Every instance-method stub carries ``params: dict = Body(default=None,
embed=True)`` whether or not the model declares parameters, and the docstring
listed them only when there were some - so for ``Loan.renew()`` the agent read
a handler with a ``params`` argument and nothing anywhere saying the method
takes none. 20 of the 22 gpt-5.6-terra library runs whose model declares
``Loan.renew()`` then made a body key required, and the delivered Renew
button 422ed.
"""

import os

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Method, Parameter, PrimitiveDataType, Property,
)
from besser.generators.backend import BackendGenerator


def _loan_methods_router(tmp_path) -> str:
    loan = Class(name="Loan")
    loan.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    loan.methods = {
        Method(name="renew", type=PrimitiveDataType("bool")),
        Method(name="reschedule", type=PrimitiveDataType("bool"), parameters=[
            Parameter(name="dueDate", type=PrimitiveDataType("date")),
        ]),
    }
    BackendGenerator(model=DomainModel(name="Library", types={loan}),
                     output_dir=str(tmp_path)).generate()
    with open(os.path.join(str(tmp_path), "routers", "loan_methods.py"),
              encoding="utf-8") as fh:
        return fh.read()


def _docstring_of(content: str, function: str) -> str:
    body = content.split(f"async def {function}(", 1)[1]
    return body.split('"""')[1]


def test_the_zero_parameter_stub_states_that_it_takes_none(tmp_path):
    doc = _docstring_of(_loan_methods_router(tmp_path), "execute_loan_renew")
    assert "no parameters" in doc
    # The rule, not just the fact: choose a value rather than demand one.
    assert "empty request body" in doc
    assert "default" in doc


def test_a_method_with_parameters_still_documents_them(tmp_path):
    doc = _docstring_of(_loan_methods_router(tmp_path), "execute_loan_reschedule")
    assert "Parameters:" in doc
    assert "dueDate" in doc
    assert "no parameters" not in doc
