import datetime
from besser.BUML.metamodel.state_machine.state_machine import StateMachine, Session, Body, Event

sm = StateMachine(name='traffic_light')
# Examples to add parameters
parameter1 = sm.new_property(section='section1', name='property1', value=1)
parameter2 = sm.new_property(section='section1', name='property2', value='hello')
parameter2 = sm.new_property(section='section2', name='property1', value=2)
# Another way to define a property is with sm.add_property(...)

red_state = sm.new_state(name='red', initial=True)
green_state = sm.new_state(name='green')  # Implicitly, initial=False
amber_state = sm.new_state(name='amber')  # Implicitly, initial=False

def chronometer_finished_callable(session: Session, event_params: dict):
    # Python code here
    # Return Boolean value
    # Session can be read/written
    seconds = event_params['seconds']
    last_light_change_timestamp = session.get('last_light_change_timestamp')
    current_timestamp = datetime.datetime.now()
    if current_timestamp > (last_light_change_timestamp + seconds):
        return True
    return False

chronometer_finished = Event(name='chronometer_finished', callable=chronometer_finished_callable)

def light_body_callable(session: Session):
    # Python code here
    session.set('last_light_change_timestamp', datetime.datetime.now())
    # Example actions with the session
    # session.set('x', 555)
    # x = session.get('x')
    # session.delete('x')
    # Write anything else...
    # The return is ignored

light_body = Body(name='light_body', callable=light_body_callable)

red_state.set_body(body=light_body)
green_state.set_body(body=light_body)
amber_state.set_body(body=light_body)
# The same body can be assigned to more than 1 state
# In this example all states use the same body, although we could define different bodies for each state

# Define transitions between states

# Red --> Green (after 60 seconds)
red_state.when_event_go_to(event=chronometer_finished, dest=green_state, event_params={'seconds': 60})

# Green --> Amber (after 20 seconds)
green_state.when_event_go_to(event=chronometer_finished, dest=amber_state, event_params={'seconds': 20})

# Amber --> Red (after 3 seconds)
amber_state.when_event_go_to(event=chronometer_finished, dest=red_state, event_params={'seconds': 3})

# Fallback bodies are optional...
def fallback_body_callable(session: Session):
    # Python code here
    print('Something went wrong')
    # The return is ignored

fallback_body = Body(name='fallback_body', callable=fallback_body_callable)

# We can set the fallback body globally, only once
# This will assign the same fallback body to all the states
sm.set_global_fallback_body(fallback_body)

# Or we can set for each state
red_state.set_fallback_body(fallback_body)


print(sm)
