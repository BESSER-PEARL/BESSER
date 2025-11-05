from enum import Enum
from besser.BUML.metamodel.structural import Class, Property, Constraint, Element
from besser.BUML.metamodel.gui.graphical_ui import ViewComponent, Screen
from besser.BUML.metamodel.gui.style import Styling


# EventType Enum
class EventType(Enum):
    """Represents the type of event that can be triggered in the GUI.
    """
    OnClick = "onClick"
    OnSubmit = "onSubmit"
    OnHover = "onHover"
    OnScroll = "onScroll"
    OnKeyPress = "onKeyPress"



# Parameter
class Parameter(Element):
    """Represents a parameter for an action.
    
    Parameters are essential for passing data between screens and actions,
    such as entity IDs, filter values, or form data.

    Args:
        name (str): The name of the parameter.
        param_type (str): The type of the parameter (e.g., 'string', 'int', 'Property').
        value: The value of the parameter (optional).
        required (bool): Whether this parameter is required (default: False).

    Attributes:
        name (str): The name of the parameter.
        param_type (str): The type of the parameter.
        value: The value of the parameter.
        required (bool): Whether this parameter is required.
    """

    def __init__(self, name: str, param_type: str, value=None, required: bool = False):
        super().__init__()
        self.name: str = name
        self.param_type: str = param_type
        self.value = value
        self.required: bool = required

    @property
    def name(self) -> str:
        """str: Get the name of the parameter."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the name of the parameter."""
        self.__name = name

    @property
    def param_type(self) -> str:
        """str: Get the type of the parameter."""
        return self.__param_type

    @param_type.setter
    def param_type(self, param_type: str):
        """str: Set the type of the parameter."""
        self.__param_type = param_type

    @property
    def required(self) -> bool:
        """bool: Get whether the parameter is required."""
        return self.__required

    @required.setter
    def required(self, required: bool):
        """bool: Set whether the parameter is required."""
        self.__required = required

    def __repr__(self):
        return f'Parameter(name={self.name}, param_type={self.param_type}, value={self.value}, required={self.required})'


# Action
class Action(Element):
    """Represents an action that can be triggered in the GUI.
    
    Actions represent behavior/logic in the application. They are NOT visual components,
    but rather operations that can be performed (CRUD operations, navigation, etc.).

    Args:
        name (str): The name of the action.
        description (str): The description of the action.
        parameters (set[Parameter]): The set of parameters for the action (optional).
        triggered_by (ViewComponent): The component that triggers this action (optional).

    Attributes:
        name (str): The name of the action.
        description (str): The description of the action.
        parameters (set[Parameter]): The set of parameters for the action.
        triggered_by (ViewComponent): The component that triggers this action.
    """

    def __init__(self, name: str, description: str = "", parameters: set[Parameter] = None,
                 triggered_by: ViewComponent = None):
        super().__init__()
        self.name: str = name
        self.description: str = description
        self.parameters: set[Parameter] = parameters if parameters is not None else set()
        self.triggered_by: ViewComponent = triggered_by

    @property
    def name(self) -> str:
        """str: Get the name of the action."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the name of the action."""
        self.__name = name

    @property
    def description(self) -> str:
        """str: Get the description of the action."""
        return self.__description

    @description.setter
    def description(self, description: str):
        """str: Set the description of the action."""
        self.__description = description

    @property
    def parameters(self) -> set[Parameter]:
        """set[Parameter]: Get the set of parameters for the action."""
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters: set[Parameter]):
        """set[Parameter]: Set the set of parameters for the action."""
        self.__parameters = parameters

    @property
    def triggered_by(self) -> ViewComponent:
        """ViewComponent: Get the component that triggers this action."""
        return self.__triggered_by

    @triggered_by.setter
    def triggered_by(self, triggered_by: ViewComponent):
        """ViewComponent: Set the component that triggers this action."""
        self.__triggered_by = triggered_by

    def __repr__(self):
        return (f'Action(name={self.name}, description={self.description}, '
                f'parameters={len(self.parameters)} params, triggered_by={self.triggered_by})')


# Create
class Create(Action):
    """Represents a CREATE action (CRUD operation).
    
    Creates a new instance of a domain class, typically used with forms
    to add new entities to the system.

    Args:
        name (str): The name of the create action.
        description (str): The description of the create action.
        target_class (Class): The target class to create instances of.
        parameters (set[Parameter]): The set of parameters for the action (optional).
        triggered_by (ViewComponent): The component that triggers this action (optional).

    Attributes:
        name (str): The name of the create action.
        description (str): The description of the create action.
        target_class (Class): The target class to create instances of.
        parameters (set[Parameter]): The set of parameters for the action.
        triggered_by (ViewComponent): The component that triggers this action.
    """

    def __init__(self, name: str, target_class: Class = None, description: str = "",
                 parameters: set[Parameter] = None, triggered_by: ViewComponent = None):
        super().__init__(name, description, parameters, triggered_by)
        self.target_class: Class = target_class

    @property
    def target_class(self) -> Class:
        """Class: Get the target class to create."""
        return self.__target_class

    @target_class.setter
    def target_class(self, target_class: Class):
        """Class: Set the target class to create."""
        self.__target_class = target_class

    def __repr__(self):
        return (f'Create(name={self.name}, target_class={self.target_class}, '
                f'description={self.description})')


# Read
class Read(Action):
    """Represents a READ action (CRUD operation).
    
    Reads/queries data from a domain class, typically used to display
    lists, tables, or detailed views of entities.

    Args:
        name (str): The name of the read action.
        description (str): The description of the read action.
        target_class (Class): The target class to read from.
        filter_constraint (Constraint): Optional constraint to filter the data.
        parameters (set[Parameter]): The set of parameters for the action (optional).
        triggered_by (ViewComponent): The component that triggers this action (optional).

    Attributes:
        name (str): The name of the read action.
        description (str): The description of the read action.
        target_class (Class): The target class to read from.
        filter_constraint (Constraint): Optional constraint to filter the data.
        parameters (set[Parameter]): The set of parameters for the action.
        triggered_by (ViewComponent): The component that triggers this action.
    """

    def __init__(self, name: str, target_class: Class = None, description: str = "",
                 filter_constraint: Constraint = None, parameters: set[Parameter] = None,
                 triggered_by: ViewComponent = None):
        super().__init__(name, description, parameters, triggered_by)
        self.target_class: Class = target_class
        self.filter_constraint: Constraint = filter_constraint

    @property
    def target_class(self) -> Class:
        """Class: Get the target class to read from."""
        return self.__target_class

    @target_class.setter
    def target_class(self, target_class: Class):
        """Class: Set the target class to read from."""
        self.__target_class = target_class

    @property
    def filter_constraint(self) -> Constraint:
        """Constraint: Get the filter constraint."""
        return self.__filter_constraint

    @filter_constraint.setter
    def filter_constraint(self, filter_constraint: Constraint):
        """Constraint: Set the filter constraint."""
        self.__filter_constraint = filter_constraint

    def __repr__(self):
        return (f'Read(name={self.name}, target_class={self.target_class}, '
                f'description={self.description}, filter_constraint={self.filter_constraint})')


# Update
class Update(Action):
    """Represents an UPDATE action (CRUD operation).
    
    Updates an existing instance of a domain class, typically used with
    edit forms to modify existing entities.

    Args:
        name (str): The name of the update action.
        description (str): The description of the update action.
        target_class (Class): The target class to update instances of.
        parameters (set[Parameter]): The set of parameters for the action (optional).
        triggered_by (ViewComponent): The component that triggers this action (optional).

    Attributes:
        name (str): The name of the update action.
        description (str): The description of the update action.
        target_class (Class): The target class to update instances of.
        parameters (set[Parameter]): The set of parameters for the action.
        triggered_by (ViewComponent): The component that triggers this action.
    """

    def __init__(self, name: str, target_class: Class = None, description: str = "",
                 parameters: set[Parameter] = None, triggered_by: ViewComponent = None):
        super().__init__(name, description, parameters, triggered_by)
        self.target_class: Class = target_class

    @property
    def target_class(self) -> Class:
        """Class: Get the target class to update."""
        return self.__target_class

    @target_class.setter
    def target_class(self, target_class: Class):
        """Class: Set the target class to update."""
        self.__target_class = target_class

    def __repr__(self):
        return (f'Update(name={self.name}, target_class={self.target_class}, '
                f'description={self.description})')


# Delete
class Delete(Action):
    """Represents a DELETE action (CRUD operation).
    
    Deletes an existing instance of a domain class, typically triggered
    by delete buttons or confirmation dialogs.

    Args:
        name (str): The name of the delete action.
        description (str): The description of the delete action.
        target_class (Class): The target class to delete instances from.
        parameters (set[Parameter]): The set of parameters for the action (optional).
        triggered_by (ViewComponent): The component that triggers this action (optional).

    Attributes:
        name (str): The name of the delete action.
        description (str): The description of the delete action.
        target_class (Class): The target class to delete instances from.
        parameters (set[Parameter]): The set of parameters for the action.
        triggered_by (ViewComponent): The component that triggers this action.
    """

    def __init__(self, name: str, target_class: Class = None, description: str = "",
                 parameters: set[Parameter] = None, triggered_by: ViewComponent = None):
        super().__init__(name, description, parameters, triggered_by)
        self.target_class: Class = target_class

    @property
    def target_class(self) -> Class:
        """Class: Get the target class to delete from."""
        return self.__target_class

    @target_class.setter
    def target_class(self, target_class: Class):
        """Class: Set the target class to delete from."""
        self.__target_class = target_class

    def __repr__(self):
        return (f'Delete(name={self.name}, target_class={self.target_class}, '
                f'description={self.description})')


# Transition
class Transition(Action):
    """Represents a TRANSITION action for navigation between screens.
    
    Handles navigation logic, including passing parameters between screens
    for data flow (e.g., passing an entity ID to a detail screen).

    Args:
        name (str): The name of the transition.
        description (str): The description of the transition.
        target_screen (Screen): The target screen to navigate to.
        parameters (set[Parameter]): Parameters to pass to the target screen (optional).
        triggered_by (ViewComponent): The component that triggers this action (optional).

    Attributes:
        name (str): The name of the transition.
        description (str): The description of the transition.
        target_screen (Screen): The target screen to navigate to.
        parameters (set[Parameter]): Parameters to pass to the target screen.
        triggered_by (ViewComponent): The component that triggers this action.
    """

    def __init__(self, name: str, target_screen: "Screen" = None, description: str = "",
                 parameters: set[Parameter] = None, triggered_by: ViewComponent = None):
        super().__init__(name, description, parameters, triggered_by)
        self.target_screen: "Screen" = target_screen

    @property
    def target_screen(self) -> "Screen":
        """Screen: Get the target screen to navigate to."""
        return self.__target_screen

    @target_screen.setter
    def target_screen(self, target_screen: "Screen"):
        """Screen: Set the target screen to navigate to."""
        self.__target_screen = target_screen

    def __repr__(self):
        return (f'Transition(name={self.name}, target_screen={self.target_screen}, '
                f'description={self.description})')


# Event
class Event(ViewComponent):
    """Represents an event that can trigger one or more actions.
    
    Events are the bridge between user interactions and application behavior.
    When an event fires (e.g., onClick, onSubmit), it triggers its associated actions.

    Args:
        name (str): The name of the event.
        event_type (EventType): The type of event (onClick, onSubmit, etc.).
        description (str): The description of the event.
        actions (set[Action]): The set of actions triggered by this event.
        visibility (str): The visibility of the event.
        timestamp (int): The timestamp of the event.
        styling (Styling): The styling of the event.

    Attributes:
        name (str): The name of the event.
        event_type (EventType): The type of event.
        description (str): The description of the event.
        actions (set[Action]): The set of actions triggered by this event (1..* relationship).
        visibility (str): The visibility of the event.
        timestamp (int): The timestamp of the event.
        styling (Styling): The styling of the event.
    """

    def __init__(self, name: str, event_type: EventType = EventType.OnClick,
                 description: str = "", actions: set[Action] = None,
                 visibility: str = "public", timestamp: int = None, styling: Styling = None):
        super().__init__(name, description, visibility, timestamp, styling=styling)
        self.event_type: EventType = event_type
        self.actions: set[Action] = actions if actions is not None else set()

    @property
    def event_type(self) -> EventType:
        """EventType: Get the type of the event."""
        return self.__event_type

    @event_type.setter
    def event_type(self, event_type: EventType):
        """EventType: Set the type of the event."""
        self.__event_type = event_type

    @property
    def actions(self) -> set[Action]:
        """set[Action]: Get the set of actions triggered by this event."""
        return self.__actions

    @actions.setter
    def actions(self, actions: set[Action]):
        """set[Action]: Set the set of actions triggered by this event."""
        if actions is not None:
            if len(actions) == 0:
                raise ValueError("An event must have at least one action (1..* relationship).")
            names = [action.name for action in actions]
            if len(names) != len(set(names)):
                raise ValueError("An event cannot have two actions with the same name.")
        self.__actions = actions

    def add_action(self, action: Action):
        """Add an action to this event.
        
        Args:
            action (Action): The action to add.
        """
        if action.name in [a.name for a in self.actions]:
            raise ValueError(f"Action with name '{action.name}' already exists in this event.")
        self.actions.add(action)

    def remove_action(self, action_name: str):
        """Remove an action from this event by name.
        
        Args:
            action_name (str): The name of the action to remove.
        """
        action_to_remove = next((a for a in self.actions if a.name == action_name), None)
        if action_to_remove:
            self.actions.remove(action_to_remove)
        else:
            raise ValueError(f"Action with name '{action_name}' not found in this event.")

    def __repr__(self):
        return (f'Event(name={self.name}, event_type={self.event_type}, '
                f'description={self.description}, actions={len(self.actions)} actions)')

