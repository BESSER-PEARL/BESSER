import pytest

from besser.BUML.metamodel.state_machine.state_machine import ConfigProperty, StateMachine


def test_new_state():
    sm1 = StateMachine('sm1')

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        state1 = sm1.new_state('state1', initial=False)
    assert "The first state of a StateMachine must be initial" in str(excinfo.value)

    state1 = sm1.new_state('state1', initial=True)
    state2 = sm1.new_state('state2')
    with pytest.raises(ValueError) as excinfo:
        # This should not work
        sm1.new_state('state2')
    assert "Duplicated state in StateMachine (state2)" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        sm1.new_state('state3', initial=True)
    assert "A StateMachine must have exactly 1 initial state" in str(excinfo.value)


def test_add_property():
    sm1 = StateMachine('sm1')
    parameter1 = sm1.add_property(ConfigProperty('section1', 'property1', 1))
    parameter2 = sm1.add_property(ConfigProperty('section1', 'property2', 2))
    with pytest.raises(ValueError) as excinfo:
        # This should not work
        parameter3 = sm1.add_property(ConfigProperty('section1', 'property2', 3))
    assert "Duplicated property in StateMachine (section1, property2)" in str(excinfo.value)


def test_new_property():
    sm1 = StateMachine('sm1')
    parameter1 = sm1.new_property('section1', 'property1', 1)
    parameter2 = sm1.new_property('section1', 'property2', 2)
    with pytest.raises(ValueError) as excinfo:
        # This should not work
        parameter3 = sm1.new_property('section1', 'property2', 3)
    assert "Duplicated property in StateMachine (section1, property2)" in str(excinfo.value)
