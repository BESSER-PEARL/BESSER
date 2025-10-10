import inspect
import textwrap
from typing import Any, Callable, List, Optional

from besser.BUML.metamodel.structural import NamedElement, Model, Method, Parameter, Type
from abc import ABC, abstractmethod


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

    def __repr__(self):
        return f"ConfigProperty(section='{self.section}', name='{self.name}', value={repr(self.value)})"


class Action(ABC):
    """Base class for actions composing a Body.
    
    Actions represent discrete operations that can be performed in a state body,
    enabling programmatic manipulation and model transformation.
    """

    @abstractmethod
    def __repr__(self):
        pass


class CustomCodeAction(Action):
    """Arbitrary code action from a callable or raw source string.
    
    Args:
        source (str, optional): Raw Python source code
        callable (Callable, optional): A callable whose source will be extracted
        
    Attributes:
        source (str): The Python source code
    """
    
    def __init__(self, source: str = None, callable: Callable = None):
        if callable is not None:
            src = inspect.getsource(callable)
            self.code = textwrap.dedent(src)
        else:
            self.code = textwrap.dedent(source) if source else ""

    def to_code(self) -> str:
        return self.code

    def __repr__(self):
        return f"CustomCodeAction(source='{self.code[:50]}...')"


class Body(Method):
    """The body of the state of a state machine.

    Each state has a Body, which defines the sequence of actions to be executed when an event causes the transition to a
    state (and a secondary body, i.e., a fallback body, that defines the actions to be executed in case of error in the
    machine).

    Args:
        name (str): The name of the body.
        callable (Callable): The function containing the body's code.
        actions (Optional[List[Action]]): List of actions composing the body

    Attributes:
        name (str): Inherited from Method, represents the name of the body.
        visibility (str): Inherited from Method, represents the visibility of the body.
        type (Type): Inherited from Method, represents the type of the body.
        is_abstract (bool): Inherited from Method, indicates if the body is abstract.
        parameters (set[structural.Parameter]): Inherited from Method, the set of parameters for the body.
        owner (Type): Inherited from Method, the type that owns the property.
        code (str): Inherited from Method, code of the body (contains source of CustomCodeAction if present, otherwise "").
        actions (List[Action]): List of actions composing the body
    """

    def __init__(self, name: str, callable: Callable = None, actions: Optional[List[Action]] = None):
        # If the caller passed actions explicitly, use them; otherwise fallback to wrapping the callable.
        if actions is None:
            actions = []
            if callable is not None:
                actions.append(CustomCodeAction(callable=callable))

        self.actions: List[Action] = actions

        # Set code to the CustomCodeAction source if exists, otherwise ""
        code = self._extract_code()

        super().__init__(
            name=name,
            parameters={Parameter(name='session', type=Type('Session'))},
            type=None,
            code=code
        )

    def _extract_code(self) -> str:
        """Extract code from CustomCodeAction if present, otherwise return empty string."""
        for action in self.actions:
            if isinstance(action, CustomCodeAction):
                return action.to_code()
        return ""

    def add_action(self, action: Action) -> 'Body':
        """Add an action to the body.

        Args:
            action (Action): The action to add

        Returns:
            Body: Returns self for method chaining
        """
        self.actions.append(action)
        
        # Update code: if new action is CustomCodeAction, set code to its code, otherwise set to ""
        if isinstance(action, CustomCodeAction):
            self.code = action.to_code()
        else:
            self.code = ""
        
        return self

    def add_custom_code(self, source: str) -> 'Body':
        """Add custom Python code as an action.

        Args:
            source (str): The Python source code

        Returns:
            Body: Returns self for method chaining
        """
        self.add_action(CustomCodeAction(source=source))
        return self

    def __repr__(self):
        if self.code:
            return f"Body(name='{self.name}', code='{self.code[:50]}...')"
        names = [repr(a) for a in self.actions]
        return f"Body(name='{self.name}', actions={names})"


class Event(NamedElement):
    """The representation of an event (i.e., external or internal stimuli or input) that may cause the transition of state
    in a state machine.

    Args:
       name (str): The name of the event.

    Attributes:
       name (str): Inherited from NamedElement, represents the name of the event.
    """
    def __init__(self, name: str):
        super().__init__(name)


class Condition(Method):
    """The representation of a condition (i.e., a boolean function) that may cause the transition of state
    in a state machine.

    Args:
        name (str): The name of the condition.
        callable (Callable): The function containing the condition's code.

    Attributes:
        name (str): Inherited from Method, represents the name of the condition.
        visibility (str): Inherited from Method, represents the visibility of the condition.
        type (Type): Inherited from Method, represents the type of the condition.
        is_abstract (bool): Inherited from Method, indicates if the condition is abstract.
        parameters (set[structural.Parameter]): Inherited from Method, the set of parameters for the condition.
        owner (Type): Inherited from Method, the type that owns the property.
        code (str): Inherited from Method, code of the condition.
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
                Parameter(name='params', type=Type('dict'))
            },
            type=Type('bool'),
            code=code
        )

    def __repr__(self):
        return f"Condition(name='{self.name}')"


class TransitionBuilder:
    """A transition builder.

    This class is used to build transitions, allowing for a "fluent api" syntax where consecutive calls can be
    made on the same object.

    Args:
        source (State): the source state of the transition
        event (Event): the event linked to the transition (can be None)
        conditions (list[Condition]): the conditions associated to the transition (can be None)

    Attributes:
        source (State): The source state of the transition
        event (Event): The event linked to the transition (can be None)
        condition (list[Condition]): The conditions associated to the transition (can be None)
    """

    def __init__(self, source: 'State', event: Event = None, conditions: list[Condition] = None):
        self.source: 'State' = source
        self.event: Event = event
        if conditions is None:
            conditions = []
        self.conditions: list[Condition] = conditions

    def with_condition(
            self,
            condition: Condition
    ) -> 'TransitionBuilder':
        self.conditions.append(condition)
        return self

    def go_to(self, dest: 'State') -> None:
        """Set the destination state of the transition.

        Completes the transition builder and effectively adds the source state.

        Args:
            dest (State): the destination state
        """
        if dest not in self.source.sm.states:
            raise ValueError(f'State {dest.name} not found in state machine {self.source.sm.name}')

        for transition in self.source.transitions:
            if transition.is_auto():
                raise ValueError(f'State {self.source.name} cannot contain an auto transition with other transitions')

        self.source.transitions.append(Transition(
            name=self.source._t_name(),
            source=self.source,
            dest=dest,
            event=self.event,
            conditions=self.conditions
        ))


class Transition(NamedElement):
    """A state machine transition from one state (source) to another (destination).

    A transition is triggered when an event and/or condition/s occurs.

    Args:
        name (str): Inherited from NamedElement, the transition name
        source (State): the source state of the transition (from where it is triggered)
        dest (State): the destination state of the transition (where the machine moves to)
        event (Callable[[Session, dict], bool]): the event that triggers the transition
        conditions (list[Condition]): the conditions that trigger the transition

    Attributes:
        name (str): Inherited from NamedElement, the transition name
        visibility (str): Inherited from NamedElement, determines the kind of visibility of the named element (public as default).
        source (State): The source state of the transition (from where it is triggered)
        dest (State): The destination state of the transition (where the machine moves to)
        event (Event): The event that triggers the transition
        conditions (list[Condition]): The conditions that trigger the transition
    """
    def __init__(
            self,
            name: str,
            source: 'State',
            dest: 'State',
            event: Event,
            conditions: list[Condition]
    ):
        super().__init__(name)
        self.source: 'State' = source
        self.dest: 'State' = dest
        self.event: Event = event
        self.conditions: list[Condition] = conditions

    def is_auto(self) -> bool:
        """Check if the transition is `auto` (i.e. no event nor condition linked to it).

        Returns:
            bool: true if the transition is auto, false otherwise
        """
        return not self.event and not self.conditions

    def __repr__(self):
        return f"Transition(name='{self.name}', source='{self.source.name}', dest='{self.dest.name}', event='{self.event.name}', conditions='{[condition.name for condition in self.conditions]}')"


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

    def when_event(self, event: Event) -> TransitionBuilder:
        return TransitionBuilder(source=self, event=event)

    def when_condition(self, condition: Condition) -> TransitionBuilder:
        return TransitionBuilder(source=self, conditions=[condition])

    def __repr__(self):
        return f"State(name='{self.name}', initial={self.initial})"


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
            State or None: the initial state of the machine, if exists
        """
        for state in self.states:
            if state.initial:
                return state
        return None

    def set_global_fallback_body(self, body: Body) -> None:
        """Set the global fallback body for all states in the state machine.

        Args:
            body (Body): The fallback body to be set for all states.
        """
        for state in self.states:
            state.fallback_body = body

    def __repr__(self):
        states_str = ', '.join([str(state) for state in self.states])
        props_str = ', '.join([str(prop) for prop in self.properties])
        return f"StateMachine(name='{self.name}', states=[{states_str}], properties=[{props_str}])"


class Session:
    """A user session in a state machine execution.

    When a user starts interacting with a state machine, a session is assigned to him/her to store user related
    information, such as the current state or any custom variable. A session can be accessed from the body of
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

    def __repr__(self):
        return f"Session(id='{self.id}', current_state='{self.current_state.name if self.current_state else None}')"

