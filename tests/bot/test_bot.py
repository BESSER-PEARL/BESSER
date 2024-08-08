import operator
import pytest

from besser.BUML.metamodel.state_machine.bot import Bot, Intent, Entity


def test_add_intent():
    bot1 = Bot('bot1')
    intent1 = bot1.add_intent(Intent('intent1'))
    intent2 = bot1.add_intent(Intent('intent2'))
    with pytest.raises(ValueError) as excinfo:
        # This should not work
        intent3 = bot1.add_intent(Intent('intent2'))
    assert "A bot cannot have two intents with the same name (intent2)." in str(excinfo.value)


def test_new_intent():
    bot1 = Bot('bot1')
    intent1 = bot1.new_intent('intent1')
    intent2 = bot1.new_intent('intent2')
    with pytest.raises(ValueError) as excinfo:
        # This should not work
        intent3 = bot1.new_intent('intent2')
    assert "A bot cannot have two intents with the same name (intent2)." in str(excinfo.value)


def test_add_entity():
    bot1 = Bot('bot1')
    entity1 = bot1.add_entity(Entity('entity1'))
    entity2 = bot1.add_entity(Entity('entity2'))
    with pytest.raises(ValueError) as excinfo:
        # This should not work
        entity3 = bot1.add_entity(Entity('entity2'))
    assert "A bot cannot have two entities with the same name (entity2)." in str(excinfo.value)


def test_new_entity():
    bot1 = Bot('bot1')
    entity1 = bot1.new_entity('entity1')
    entity2 = bot1.new_entity('entity2')
    with pytest.raises(ValueError) as excinfo:
        # This should not work
        entity3 = bot1.new_entity('entity2')
    assert "A bot cannot have two entities with the same name (entity2)." in str(excinfo.value)


def test_intent_parameters():
    bot1 = Bot('bot1')
    entity1 = bot1.new_entity('entity1')
    intent1 = bot1.new_intent('intent1')
    intent1.parameter('parameter1', 'fragment', entity1)
    intent1.parameter('parameter2', 'fragment', entity1)
    with pytest.raises(ValueError) as excinfo:
        # This should not work
        intent1.parameter('parameter2', 'fragment', entity1)
    assert "An intent cannot have two parameters with the same name (parameter2)" in str(excinfo.value)


def test_new_state():
    bot1 = Bot('bot1')

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        state1 = bot1.new_state('state1', initial=False)
    assert "The first state of a bot must be initial" in str(excinfo.value)

    state1 = bot1.new_state('state1', initial=True)
    state2 = bot1.new_state('state2')
    with pytest.raises(ValueError) as excinfo:
        # This should not work
        bot1.new_state('state2')
    assert "Duplicated state in bot (state2)" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        bot1.new_state('state3', initial=True)
    assert "A bot must have exactly 1 initial state" in str(excinfo.value)


def test_when_intent_matched_go_to():
    bot1 = Bot('bot1')
    intent1 = bot1.new_intent('intent1')
    intent2 = bot1.new_intent('intent2')
    state1 = bot1.new_state('state1', initial=True)
    state2 = bot1.new_state('state2', initial=False)
    state1.when_intent_matched_go_to(intent1, state2)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        state1.when_intent_matched_go_to(intent1, state2)
    assert "Duplicated intent matching transition in a state (intent1)" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # This should not work
        state1.when_intent_matched_go_to(Intent('intent3'), state2)
    assert "Intent intent3 not found" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        bot2 = Bot('bot2')
        state3 = bot2.new_state('state3', initial=True)
        # This should not work
        state1.when_intent_matched_go_to(intent2, state3)
    assert "State state3 not found" in str(excinfo.value)


def test_when_no_intent_matched_go_to():
    bot1 = Bot('bot1')
    intent1 = bot1.new_intent('intent1')
    state1 = bot1.new_state('state1', initial=True)
    state2 = bot1.new_state('state2', initial=False)
    state1.when_no_intent_matched_go_to(state2)

    with pytest.raises(ValueError) as excinfo:
        bot2 = Bot('bot2')
        state3 = bot2.new_state('state3', initial=True)
        # This should not work
        state1.when_no_intent_matched_go_to(state3)
    assert "State state3 not found" in str(excinfo.value)


def test_when_variable_matches_operation_go_to():
    bot1 = Bot('bot1')
    intent1 = bot1.new_intent('intent1')
    state1 = bot1.new_state('state1', initial=True)
    state2 = bot1.new_state('state2', initial=False)
    state1.when_variable_matches_operation_go_to('var', operator.eq, 1, state2)

    with pytest.raises(ValueError) as excinfo:
        bot2 = Bot('bot2')
        state3 = bot2.new_state('state3', initial=True)
        # This should not work
        state1.when_variable_matches_operation_go_to('var', operator.eq, 1, state3)
    assert "State state3 not found" in str(excinfo.value)


def test_when_file_received_go_to():
    bot1 = Bot('bot1')
    intent1 = bot1.new_intent('intent1')
    state1 = bot1.new_state('state1', initial=True)
    state2 = bot1.new_state('state2', initial=False)
    state1.when_file_received_go_to(state2)

    with pytest.raises(ValueError) as excinfo:
        bot2 = Bot('bot2')
        state3 = bot2.new_state('state3', initial=True)
        # This should not work
        state1.when_file_received_go_to(state3)
    assert "State state3 not found" in str(excinfo.value)


def test_go_to():
    bot1 = Bot('bot1')
    intent1 = bot1.new_intent('intent1')
    state1 = bot1.new_state('state1', initial=True)
    state2 = bot1.new_state('state2', initial=False)
    state1.go_to(state2)

    with pytest.raises(ValueError) as excinfo:
        state3 = bot1.new_state('state3', initial=False)
        # This should not work
        state1.go_to(state3)
    assert "Auto transition conflicting" in str(excinfo.value)
