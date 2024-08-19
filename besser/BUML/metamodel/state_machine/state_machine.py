import inspect
from typing import Any, Callable

from besser.BUML.metamodel.structural import NamedElement, Model, Method, Parameter, Type


class ConfigProperty:
    """A configuration property of a state machine.

    Args:
        section (str): The section the configuration property belongs to
        name (str): The name of the configuration property
        value (Any): Te value of the configuration property

    Attributes:
        section (str): The section the configuration property belongs to
        name (str): The name of the configuration property
        value (Any): Te value of the configuration property
    """

    def __init__(self, section: str, name: str, value: Any):
        self.section: str = section
        self.name: str = name
        self.value: Any = value

    def __eq__(self, other):
        if type(other) is type(self):
            return self.section == other.section and self.name == other.name
        else:
            return False

    def __hash__(self):
        return hash((self.section, self.name))


class Body(Method):
    """The body of the state of a state machine.

    Each state has a Body, which defines the sequence of actions to be executed when an event causes the transition to a
    state (and a secondary body, i.e., a fallback body, that defines the actions to be executed in case of error in the
    machine).

    Args:
        name (str): The name of the body.
        callable (Callable): The function containing the body's code.

    Attributes:
        name (str): Inherited from Method, represents the name of the body.
        visibility (str): Inherited from Method, represents the visibility of the body.
        type (Type): Inherited from Method, represents the type of the body.
        is_abstract (bool): Inherited from Method, indicates if the body is abstract.
        parameters (set[Parameter]): Inherited from Method, the set of parameters for the body.
        owner (Type): Inherited from Method, the type that owns the property.
        code (str): Inherited from Method, code of the body.
    """

    def __init__(self, name: str, callable: Callable):
        super().__init__(
            name=name,
            parameters={Parameter(name='session', type=Type('Session'))},
            type=None,
            code=inspect.getsource(callable)
        )


class Event(Method):
    """The representation of an event (i.e., external or internal stimuli or input) that may cause the transition of state
    in a state machine.

    Args:
        name (str): The name of the event.
        callable (Callable): The function containing the event's code.

    Attributes:
        name (str): Inherited from Method, represents the name of the body.
        visibility (str): Inherited from Method, represents the visibility of the body.
        type (Type): Inherited from Method, represents the type of the body.
        is_abstract (bool): Inherited from Method, indicates if the body is abstract.
        parameters (set[Parameter]): Inherited from Method, the set of parameters for the body.
        owner (Type): Inherited from Method, the type that owns the property.
        code (str): Inherited from Method, code of the body.
    """

    def __init__(self, name: str, callable: Callable):
        if callable is not None:
            code = inspect.getsource(callable)
        else:
            code = None
        super().__init__(
            name=name,
            parameters={
                Parameter(name='session', type=Type('Session')),
                Parameter(name='event_params', type=Type('dict'))
            },
            type=Type('bool'),
            code=code
        )


class Transition(NamedElement):
    """A state machine transition from one state (source) to another (destination).

    A transition is triggered when an event occurs.

    Args:
        name (str): Inherited from NamedElement, the transition name
        source (State): the source state of the transition (from where it is triggered)
        dest (State): the destination state of the transition (where the machine moves to)
        event (Callable[[Session, dict], bool]): the event that triggers the transition
        event_params (dict): the parameters associated to the event

    Attributes:
        name (str): Inherited from NamedElement, the transition name
        visibility (str): Inherited from NamedElement, determines the kind of visibility of the named element (public as default).
        source (State): The source state of the transition (from where it is triggered)
        dest (State): The destination state of the transition (where the machine moves to)
        event (Event): The event that triggers the transition
        event_params (dict): The parameters associated to the event
    """
    def __init__(
            self,
            name: str,
            source: 'State',
            dest: 'State',
            event: Event,
            event_params: dict = {}
    ):
        super().__init__(name)
        self.source: 'State' = source
        self.dest: 'State' = dest
        self.event: Event = event
        self.event_params: dict = event_params


class State(NamedElement):
    """A state of a state machine.

    Args:
        sm (StateMachine): the state machine the state belongs to
        name (str): the state name
        initial (bool): whether the state is initial or not

    Attributes:
        name (str): Inherited from NamedElement, the state name
        visibility (str): Inherited from NamedElement, determines the kind of visibility of the state (public as default).
        sm (StateMachine): the state machine the state belongs to
        initial (bool): whether the state is initial or not
        transitions (list[Transition]): The state's transitions to other states
        body (Body): the body of the state
        fallback_body (Body): the fallback body of the state
        _transition_counter (int): Count the number of transitions of this state. Used to name the transitions.
    """

    def __init__(self, sm: 'StateMachine', name: str, initial: bool = False):

        super().__init__(name)
        self.sm: StateMachine = sm
        self.initial: bool = initial
        self.transitions: list[Transition] = []
        self.body: Body = None
        self.fallback_body: Body = None
        self._transition_counter: int = 0

    def __eq__(self, other):
        if type(other) is type(self):
            return self.name == other.name and self.sm.name == other.sm.name
        else:
            return False

    def __hash__(self):
        return hash((self.name, self.sm.name))

    def _t_name(self) -> str:
        """Name generator for transitions. Transition names are generic and enumerated. On each call, a new name is
        generated and the transition counter is incremented for the next name.

        Returns:
            str: a name for the next transition
        """
        self._transition_counter += 1
        return f"t_{self._transition_counter}"

    def set_body(self, body: Body) -> None:
        """Set the state body.

        Args:
            body (Body): the body
        """
        self.body = body

    def set_fallback_body(self, body: Body) -> None:
        """Set the state fallback body.

        Args:
            body (Body): the body
        """
        self.fallback_body = body

    def when_event_go_to(self, event: Event, dest: 'State', event_params: dict) -> None:
        self.transitions.append(Transition(name=self._t_name(), source=self, dest=dest, event=event, event_params=event_params))


class StateMachine(Model):
    """A state machine model.

    Args:
        name (str): the state machine name

    Attributes:
        name (str): Inherited from Model, represents the name of the state machine.
        visibility (str): Inherited from Model, determines the kind of visibility of the state machine (public as default).
        states (list[State]): the states of the state machine
        properties (list[ConfigProperty]): the configuration properties of the state machine.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.states: list[State] = []
        self.properties: list[ConfigProperty] = []

    def add_property(self, property: ConfigProperty) -> ConfigProperty:
        """Add a configuration property to the state machine.

        Args:
            property (ConfigProperty): the property to add

        Returns:
            ConfigProperty: the configuration property
        """
        if property in self.properties:
            raise ValueError(f"Duplicated property in StateMachine ({property.section}, {property.name})")
        self.properties.append(property)
        return property

    def new_property(self, section: str, name: str, value: Any) -> ConfigProperty:
        """Create a new configuration property on the state machine.

        Args:
            section (str): The section the configuration property belongs to
            name (str): The name of the configuration property
            value (Any): Te value of the configuration property

        Returns:
            ConfigProperty: the configuration property
        """
        new_property = ConfigProperty(section, name, value)
        if new_property in self.properties:
            raise ValueError(f"Duplicated property in StateMachine ({new_property.section}, {new_property.name})")
        self.properties.append(new_property)
        return new_property

    def new_state(self, name: str, initial: bool = False, ) -> State:
        """Create a new state in the state machine.

        Args:
            name (str): the state name. It must be unique in the state machine.
            initial (bool): whether the state is initial or not. A state machine must have 1 initial state.

        Returns:
            State: the state
        """
        new_state = State(self, name, initial)
        if new_state in self.states:
            raise ValueError(f"Duplicated state in StateMachine ({new_state.name})")
        if initial and self.initial_state():
            raise ValueError(f"A StateMachine must have exactly 1 initial state")
        if not initial and not self.states:
            raise ValueError(f"The first state of a StateMachine must be initial")
        self.states.append(new_state)
        return new_state

    def initial_state(self) -> State or None:
        """Get the state machine's initial state. It can be None if it has not been set.

        Returns:
            State or None: the initial state of the bot, if exists
        """
        for state in self.states:
            if state.initial:
                return state
        return None

    def set_global_fallback_body(self, body: Body) -> None:
        for state in self.states:
            state.fallback_body = body


class Session:
    """A user session in a state machine execution.

    When a user starts interacting with a state machine, a session is assigned to him/her to store user related
    information, such as the current state of the bot or any custom variable. A session can be accessed from the body of
    the states to read/write user information. If a state machine does not have the concept of 'users' (i.e., there are
    no concurrent executions of the state machine, but a single one) then it could simply have 1 unique session.

    Attributes:
        id (str): The session id, which must unique among all state machine sessions
        current_state (str): The current state in the state machine for this session
    """
    def __init__(self):
        self.id: str = None
        self.current_state: State = None

    def set(self, key: str, value: Any) -> None:
        """Set an entry to the session private data storage.

        Args:
            key (str): the entry key
            value (Any): the entry value
        """
        pass

    def get(self, key: str) -> Any:
        """Get an entry of the session private data storage.

        Args:
            key (str): the entry key

        Returns:
            Any: the entry value, or None if the key does not exist
        """
        pass

    def delete(self, key: str) -> None:
        """Delete an entry of the session private data storage.

        Args:
            key (str): the entry key
        """
        pass

    def move(self, transition: Transition) -> None:
        """Move to another state of the state machine.

        Args:
            transition (Transition): the transition that points to the state to move
        """
        pass
