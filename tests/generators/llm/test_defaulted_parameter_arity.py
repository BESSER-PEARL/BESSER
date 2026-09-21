"""A parameter with a default is not something the caller must supply.

``action_arity`` is the callable contract for a modelled action, and its
own docstring says why it matters: "a zero-parameter action has to succeed
on an empty body, because that is all the frontend's method button ever
sends." The zero-argument check keys on ``arity == 0``.

It was counting every declared parameter, so ``renew(days: int = 7)`` read
as arity 1 and was skipped - even though the button calls it with ``{}``
and it is perfectly callable that way. The distinction only became
expressible when the serializer started emitting a parameter's ``default``.

The direction matters: this makes the check apply to MORE methods, so the
risk is false positives on methods that genuinely need an argument. The
last two tests pin that boundary.
"""

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Method, Parameter, StringType, IntegerType,
)
from besser.generators.llm.contract_checks import build_data_contract


def _model(*parameters) -> DomainModel:
    booking = Class(name="Booking", attributes=set())
    booking.methods = {Method(name="renew", parameters=set(parameters))}
    return DomainModel(name="m", types={booking})


def _arity(model) -> int:
    return build_data_contract(model).action_arity["Booking"]["renew"]


def test_a_method_with_no_parameters_is_zero_arity():
    assert _arity(_model()) == 0


def test_a_fully_defaulted_parameter_list_is_zero_arity():
    """The button sends {} and this method accepts it, so the rule applies."""
    model = _model(Parameter(name="days", type=IntegerType, default_value=7))

    assert _arity(model) == 0


def test_every_parameter_defaulted_is_still_zero_arity():
    model = _model(
        Parameter(name="days", type=IntegerType, default_value=7),
        Parameter(name="note", type=StringType, default_value="none"),
    )

    assert _arity(model) == 0


def test_a_required_parameter_is_still_counted():
    """The boundary: this genuinely cannot be called with an empty body."""
    model = _model(Parameter(name="dueDate", type=StringType))

    assert _arity(model) == 1


def test_only_the_required_parameters_are_counted_in_a_mixed_list():
    model = _model(
        Parameter(name="dueDate", type=StringType),
        Parameter(name="days", type=IntegerType, default_value=7),
    )

    assert _arity(model) == 1
