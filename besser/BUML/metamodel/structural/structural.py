from abc import ABC
from datetime import datetime, timedelta
from typing import Any, Union, List
import time

# constant
UNLIMITED_MAX_MULTIPLICITY = 9999

class Element(ABC):
    """Element is the Superclass of all structural model elements.

    Args:
        timestamp (datetime): Object creation datetime (default is current time).
        is_derived (bool): Indicates whether the element is derived (False as default).
    
    Attributes:
        timestamp (datetime): Object creation datetime (default is current time).
        is_derived (bool): Indicates whether the element is derived (False as default).
    """

    def __init__(self, timestamp: datetime = None, is_derived: bool = False):
        self.timestamp: datetime = timestamp if timestamp is not None else datetime.now() + \
                         timedelta(microseconds=(time.perf_counter_ns() % 1_000_000) / 1000)
        self.is_derived: bool = is_derived

    @property
    def timestamp(self) -> datetime:
        """str: Get the timestamp of the element."""
        return self.__timestamp

    @timestamp.setter
    def timestamp(self, timestamp: datetime):
        """str: Set the timestamp of the element."""
        self.__timestamp = timestamp

    @property
    def is_derived(self) -> bool:
        """bool: Get whether the element is derived."""
        return self.__is_derived

    @is_derived.setter
    def is_derived(self, is_derived: bool):
        """bool: Set whether the element is derived."""
        self.__is_derived = is_derived

class NamedElement(Element):
    """NamedElement represent a structural element with a name.

    Args:
        name (str): The name of the named element
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the named element (None as default).
        visibility (str): Determines the kind of visibility of the named element (public as default).
        is_derived (bool): Indicates whether the element is derived (False as default).

    Attributes:
        name (str): The name of the named element
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the named element (None as default).
        visibility: Determines the kind of visibility of the named element (public as default).
        is_derived (bool): Indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, timestamp: datetime = None, synonyms: List[str] = None,
                 visibility: str = "public", is_derived: bool = False):
        super().__init__(timestamp, is_derived)
        self.name: str = name
        self.synonyms: List[str] = synonyms
        self.visibility: str = visibility


    @property
    def name(self) -> str:
        """str: Get the name of the named element."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """
        str: Set the name of the named element.
        
        Raises:
            ValueError: If the name is empty or contains any whitespace characters.
        """
        if ' ' in name:
            raise ValueError(f"'{name}' is invalid. Name cannot contain spaces.")
        self.__name = name

    @property
    def visibility(self) -> str:
        """str: Get the visibility of the named element."""
        return self.__visibility

    @visibility.setter
    def visibility(self, visibility: str):
        """
        str: Set the visibility of the named element.
        
        Raises:
            ValueError: If the visibility provided is none of these: public, 
            private, protected, or package.
        """
        if visibility not in ['public', 'private', 'protected', 'package']:
            raise ValueError("Invalid value of visibility")
        self.__visibility = visibility

    @property
    def synonyms(self) -> List[str]:
        """List[str]: Get the list of synonyms of the named element."""
        return self.__synonyms

    @synonyms.setter
    def synonyms(self, synonyms: List[str]):
        """List[str]: Set the list of synonyms of the named element."""
        self.__synonyms = synonyms

class Type(NamedElement):
    """Type is the Superclass of classes and data types in the model.

    Args:
        name (str): The name of the Type.
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the type (None as default).
        is_derived (bool): Indicates whether the element is derived (False as default).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the Type.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the type (None as default).
        is_derived (bool): Indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, timestamp: int = None, synonyms: List[str] = None,
                is_derived: bool = False):
        super().__init__(name, timestamp, synonyms, is_derived=is_derived)

    def __repr__(self):
        return f"Type({self.name}, {self.timestamp}, {self.synonyms})"

class DataType(Type):
    """Represents a data type.

    This class inherits from NamedElement and is used to model data types.

    Args:
        name (str): The name of the data type.
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the data type (None as default).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the data type.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the data type (None as default).
    """

    def __repr__(self):
        return f"DataType({self.name})"

class PrimitiveDataType(DataType):
    """Class representing a primitive data type.

    This class is a subclass of DataType and is used to represent primitive data types
    with a specified name.

    Args:
        name (str): The name of the primitive data type.
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the primitive data type (None as default).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the primitive data type.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the primitive data type (None as default).
    """

    @NamedElement.name.setter
    def name(self, name: str):
        """
        str: Set the name of the PrimitiveDataType. 
        
        Raises:
            ValueError: If an invalid primitive data type is provided.
                        Allowed values are int, float, str, bool, time, date, 
                        datetime, timedelta, and any.
        """
        if name not in ['int', 'float', 'str', 'bool', 'time', 'date', 'datetime', 'timedelta']:
            raise ValueError("Invalid primitive data type")
        super(PrimitiveDataType, PrimitiveDataType).name.fset(self, name)

    def __repr__(self):
        return f"PrimitiveDataType({self.name}, {self.timestamp}, {self.synonyms})"

# Define instances of PrimitiveDataType
StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")
FloatType = PrimitiveDataType("float")
BooleanType = PrimitiveDataType("bool")
TimeType = PrimitiveDataType("time")
DateType = PrimitiveDataType("date")
DateTimeType = PrimitiveDataType("datetime")
TimeDeltaType = PrimitiveDataType("timedelta")
AnyType = DataType("any")
data_types = {StringType, IntegerType, FloatType, BooleanType,
              TimeType, DateType, DateTimeType, TimeDeltaType, AnyType}

class EnumerationLiteral(NamedElement):
    """Class representing an enumeration literal.

    This class is a subclass of NamedElement and is used to represent individual
    literals within an enumeration.

    Args:
        name (str): The name of the enumeration literal.
        owner (DataType): The owner data type of the enumeration literal (None as default).
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the literal (None as default).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the enumeration literal.
        owner (DataType): Represents the owner data type of the enumeration literal (None as default).
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the literal (None as default).
    """

    def __init__(self, name: str, owner: DataType=None, timestamp: int = None, synonyms: List[str] = None):
        super().__init__(name, timestamp, synonyms)
        self.owner: DataType = owner

    @property
    def owner(self) -> DataType:
        """Datatype: Get the owner."""
        return self.__owner

    @owner.setter
    def owner(self, owner: DataType):
        """
        DataType: Set the owner. 
        
        Raises:
            ValueError: If the owner is not an enumeration.
        """
        if isinstance(owner, PrimitiveDataType):
            raise ValueError("Invalid owner")
        self.__owner = owner

    def __repr__(self):
        return f"EnumerationLiteral({self.name}, {self.owner}, {self.timestamp}, {self.synonyms})"

class Enumeration(DataType):
    """Class representing an enumeration.

    This class is a subclass of DataType and is used to represent enumerations
    with a specified name and a set of enumeration literals.

    Args:
        name (str): The name of the enumeration data type.
        literals (set[EnumerationLiteral]): Set of enumeration literals associated with the 
                enumeration (None as default).
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the enumeration (None as default).

    Attributes:
        name (str): Inherited from DataType, represents the name of the enumeration.
        literals (set[EnumerationLiteral]): Represents a set of enumeration literals associated 
                with the enumeration (None as default).
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is 
                current time).
        synonyms (List[str]): List of synonyms of the enumeration (None as default).
    """

    def __init__(self, name: str, literals: set[EnumerationLiteral] = None, timestamp: int = None,
                 synonyms: List[str] = None):
        super().__init__(name, timestamp, synonyms)
        self.literals: set[EnumerationLiteral] = literals if literals is not None else set()

    @property
    def literals(self) -> set[EnumerationLiteral]:
        """set[EnumerationLiteral]: Get the set of literals."""
        return self.__literals

    @literals.setter
    def literals(self, literals: set[EnumerationLiteral]):
        """
        DataType: Set the literals. 
        
        Raises:
            ValueError: if two literals have the same name.
        """
        if literals is not None:
            names = [literal.name for literal in literals]
            if len(names) != len(set(names)):
                raise ValueError("An enumeration cannot have two literals with the same name")
            for literal in literals:
                literal.owner = self
            self.__literals = literals
        else:
            self.__literals = set()

    def add_literal(self, literal: EnumerationLiteral):
        """
        Add an enumeration literal to the set.
        
        Raises:
            ValueError: if the enumeration literal name already exist.
        """
        if self.literals is not None:
            if literal.name in [literal.name for literal in self.literals]:
                raise ValueError(f"An enumeration cannot have two literals with the same name: '{literal.name}'")
        self.literals.add(literal)

    def __repr__(self):
        return f"Enumeration({self.name}, {self.literals}, {self.timestamp}, {self.synonyms})"

class TypedElement(NamedElement):
    """TypedElement is a subclass of NamedElement and is used to represent elements
    that have a specific type.

    Args:
        name (str): The name of the typed element.
        type (Type, str): The data type of the typed element.
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the typed element (None as default).
        visibility (str): Determines the kind of visibility of the typed element (public as default).
        is_derived (bool): Indicates whether the element is derived (False as default).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the typed element.
        type (Type): The data type of the typed element.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        type (Type, str): The data type of the typed element.
        synonyms (List[str]): List of synonyms of the typed element (None as default).
        visibility (str): Inherited from NamedElement, represents the visibility of the typed element (public as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    type_mapping = {
        "str": StringType,
        "string": StringType,
        "int": IntegerType,
        "float": FloatType,
        "bool": BooleanType,
        "time": TimeType,
        "date": DateType,
        "datetime": DateTimeType,
        "timedelta": TimeDeltaType
    }

    def __init__(self, name: str, type: Union[Type, str], timestamp: int = None, synonyms: List[str] = None,
                 visibility: str="public", is_derived: bool = False):
        super().__init__(name, timestamp, synonyms, visibility, is_derived)
        self.type = self.type_mapping.get(type, type)

    @property
    def type(self) -> Type:
        """Type: Get the type of the typed element."""
        return self.__type

    @type.setter
    def type(self, type: Type):
        """Type: Set the type of the typed element."""
        self.__type = type

class Multiplicity(Element):
    """Represents the multiplicity of a Property.

    It consists of a minimum and maximum value, indicating the allowed range.

    Args:
        min_multiplicity (int): The minimum multiplicity.
        max_multiplicity (int): The maximum multiplicity. Use "*" for unlimited.
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).

    Attributes:
        min (int): The minimum multiplicity.
        max (int): The maximum multiplicity. Use "*" for unlimited.
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, min_multiplicity: int, max_multiplicity: int, is_derived: bool = False):
        super().__init__(is_derived=is_derived)
        self.min: int = min_multiplicity
        self.max: int = max_multiplicity

    @property
    def min(self) -> int:
        """int: Get the minimum multiplicity."""
        return self.__min

    @min.setter
    def min(self, min_multiplicity: int):
        """
        int: Set the minimum multiplicity 
        
        Raises:
            ValueError: (Invalid min multiplicity) if the minimum multiplicity is less than 0.
        """
        if min_multiplicity < 0:
            raise ValueError("Invalid min multiplicity")
        self.__min = min_multiplicity

    @property
    def max(self) -> int:
        """int: Get the maximum multiplicity."""
        return self.__max

    @max.setter
    def max(self, max_multiplicity: int):
        """
        int: Set the maximum multiplicity.
        
        Raises:
            ValueError: (Invalid max multiplicity) if the maximum multiplicity is less than 0 or
            less than minimum multiplicity.
        """
        if max_multiplicity == "*":
            max_multiplicity = UNLIMITED_MAX_MULTIPLICITY
        if max_multiplicity <= 0:
            raise ValueError("Invalid max multiplicity")
        if max_multiplicity < self.min:
            raise ValueError("Invalid max multiplicity")
        self.__max = max_multiplicity

    def __repr__(self):
        return f'Multiplicity({self.min}, {self.max}, is_derived={self.is_derived})'


# Properties are owned by a class or an association and point to a type with a multiplicity
class Property(TypedElement):
    """A property can represents an attribute of a class or an end of an association.

    Properties are owned by a class or an association.

    Args:
        name (str): The name of the property.
        type (Type): The type of the property.
        owner (Type): The type that owns the property (None as default).
        multiplicity (Multiplicity): The multiplicity of the property (1..1 as default).
        visibility (str): The visibility of the property (public as default).
        is_composite (bool): Indicates whether the property is a composite (False as default).
        is_navigable (bool): Indicates whether the property is navigable in a relationship (True as default).
        is_id (bool): Indicates whether the property is an id (False as default).
        is_read_only (bool): Indicates whether the property is read only (False as default).
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the property (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).

    Attributes:
        name (str): Inherited from TypedElement, represents the name of the property.
        type (Type): Inherited from TypedElement, represents the type of the property.
        owner (Type): The type that owns the property (public as default).
        multiplicity (Multiplicity): The multiplicity of the property (1..1 as default).
        visibility (str): Inherited from TypedElement, represents the visibility of the property (public as default).
        is_composite (bool): Indicates whether the property is a composite (False as default).
        is_navigable (bool): Indicates whether the property is navigable in a relationship (True as default).
        is_id (bool): Indicates whether the property is an id (False as default).
        is_read_only (bool): Indicates whether the property is read only (False as default).
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the property (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, type: Type, owner: Type = None, multiplicity: Multiplicity = Multiplicity(1, 1),
                 visibility: str = 'public', is_composite: bool = False, is_navigable: bool = True,
                 is_id: bool = False, is_read_only: bool = False, timestamp: int = None,
                 synonyms: List[str] = None, is_derived: bool = False):
        super().__init__(name, type, timestamp, synonyms, visibility, is_derived)
        self.owner: Type = owner
        self.multiplicity: Multiplicity = multiplicity
        self.is_composite: bool = is_composite
        self.is_navigable: bool = is_navigable
        self.is_id: bool = is_id
        self.is_read_only: bool = is_read_only

    @property
    def owner(self) -> Type:
        """Type: Get the owner type of the property."""
        return self.__owner

    @owner.setter
    def owner(self, owner: Type):
        """
        Type: Set the owner type of the property.
        
        Raises:
            ValueError: (Invalid owner) if the owner is instance of DataType.
        """
        if isinstance(owner, DataType):
            raise ValueError("Invalid owner")
        self.__owner = owner

    @property
    def multiplicity(self) -> Multiplicity:
        """Multiplicity: Get the multiplicity of the property."""
        return self.__multiplicity

    @multiplicity.setter
    def multiplicity(self, multiplicity: Multiplicity):
        """Multiplicity: Set the multiplicity of the property."""
        self.__multiplicity = multiplicity

    @property
    def is_composite(self) -> bool:
        """bool: Get whether the property is composite."""
        return self.__is_composite

    @is_composite.setter
    def is_composite(self, is_composite: bool):
        """bool: Set whether the property is composite."""
        self.__is_composite = is_composite

    @property
    def is_navigable(self) -> bool:
        """bool: Get whether the property is navigable."""
        return self.__is_navigable

    @is_navigable.setter
    def is_navigable(self, is_navigable: bool):
        """bool: Set whether the property is navigable."""
        self.__is_navigable = is_navigable

    @property
    def is_id(self) -> bool:
        """bool: Get whether the property is an id."""
        return self.__is_id

    @is_id.setter
    def is_id(self, is_id: bool):
        """bool: Set whether the property is an id."""
        self.__is_id = is_id

    @property
    def is_read_only(self) -> bool:
        """bool: Get whether the property is read only."""
        return self.__is_read_only

    @is_read_only.setter
    def is_read_only(self, is_read_only: bool):
        """bool: Set whether the property is read only."""
        self.__is_read_only = is_read_only

    def __repr__(self):
        return (
            f'Property({self.name}, {self.visibility}, {self.type}, {self.multiplicity}, '
            f'is_composite={self.is_composite}, is_id={self.is_id}, '
            f'is_read_only={self.is_read_only}, {self.timestamp}, {self.synonyms}, '
            f'is_derived={self.is_derived})'
        )

class Parameter(TypedElement):
    """
    Parameter is used to represent a parameter of a method with a specific type.

    Args:
        name (str): The name of the parameter.
        type (Type): The data type of the parameter.
        default_value (Any): The default value of the parameter (None as default).
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the parameter (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the parameter.
        type (Type): Inherited from TypedElement, represents the type of the parameter.
        default_value (Any): The default value of the parameter (None as default).
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the parameter (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, type: Type, default_value: Any = None, timestamp: int = None,
                 synonyms: List[str] = None, is_derived: bool = False):
        super().__init__(name, type, timestamp, synonyms, is_derived=is_derived)
        self.default_value: Any = default_value

    @property
    def default_value(self) -> Any:
        """Type: Get the default value of the parameter."""
        return self.__default_value

    @default_value.setter
    def default_value(self, default_value: Any):
        """Type: Set the default value of the parameter."""
        self.__default_value = default_value

    def __repr__(self):
        return (
            f'Parameter({self.name}, {self.type}, {self.default_value}, {self.timestamp}, '
            f'{self.synonyms}, is_derived={self.is_derived})'
        )

class Method(TypedElement):
    """
    Method is used to represent a method of a class.

    Args:
        name (str): The name of the method.
        visibility (str): Determines the kind of visibility of the method (public as default).
        is_abstract (bool): Indicates if the method is abstract (False as default).
        parameters (set[Parameter]): The set of parameters for the method (set() as default).
        type (Type): The type of the method (None as default).
        owner (Type): The type that owns the method (None as default).
        code (str): code of the method ("" as default).
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the method (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).

    Attributes:
        name (str): Inherited from TypedElement, represents the name of the method.
        visibility (str): Inherited from TypedElement, represents the visibility of the method (public as default).
        is_abstract (bool): Indicates if the method is abstract. (False as default)
        parameters (set[Parameter]): The set of parameters for the method (set() as default).
        type (Type): Inherited from TypedElement, represents the type of the method (None as default).
        owner (Type): The type that owns the property (None as default).
        code (str): code of the method ("" as default).
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the method (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, visibility: str = "public", is_abstract: bool = False,
                 parameters: set[Parameter] = None, type: Type = None, owner: Type = None,
                 code: str = "", timestamp: int = None, synonyms: List[str] = None, is_derived: bool = False):
        super().__init__(name, type, timestamp, synonyms, visibility, is_derived)
        self.is_abstract: bool = is_abstract
        self.parameters: set[Parameter] = parameters if parameters is not None else set()
        self.owner: Type = owner
        self.code: str = code

    @property
    def is_abstract(self) -> bool:
        """bool: Get whether the method is abstract."""
        return self.__is_abstract

    @is_abstract.setter
    def is_abstract(self, is_abstract: bool):
        """bool: Set whether the method is abstract."""
        self.__is_abstract = is_abstract

    @property
    def parameters(self) -> set[Parameter]:
        """set[Parameter]: Get the set of parameters of the method."""
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters: set[Parameter]):
        """
        set[Parameter]: Set the parameters of the method.
        
        Raises:
            ValueError: if two parameters have the same name.
        """
        if parameters is not None:
            names_seen = set()
            duplicates = set()

            for parameter in parameters:
                if parameter.name in names_seen:
                    duplicates.add(parameter.name)
                names_seen.add(parameter.name)
                parameter.owner = self

            if duplicates:
                raise ValueError(f"A method cannot have parameters with duplicate names: {', '.join(duplicates)}")

            self.__parameters = parameters
        else:
            self.__parameters = set()

    def add_parameter(self, parameter: Parameter):
        """
        Parameter: Add a parameter to the set of class parameters.
        
        Raises:
            ValueError: if the parameter name already exist.
        """
        if self.parameters is not None:
            if parameter.name in [parameter.name for parameter in self.parameters]:
                raise ValueError(f"A method cannot have two parameters with the same name: '{parameter.name}'")
        self.parameters.add(parameter)

    @property
    def owner(self) -> Type:
        """Type: Get the owner type of the method."""
        return self.__owner

    @owner.setter
    def owner(self, owner: Type):
        """
        Type: Set the owner type of the method.
        
        Raises:
            ValueError: (Invalid owner) if the owner is instance of DataType.
        """
        if isinstance(owner, DataType):
            raise ValueError("Invalid owner")
        self.__owner = owner

    @property
    def code(self) -> str:
        """str: Get the code of the method."""
        return self.__code

    @code.setter
    def code(self, code: str):
        """str: Set the code of the method."""
        self.__code = code

    def __repr__(self):
        return (
            f'Method({self.name}, {self.visibility}, is_abstract={self.is_abstract}, {self.parameters}, '
            f'{self.type}, {self.owner}, {self.code}, {self.timestamp}, {self.synonyms}, '
            f'is_derived={self.is_derived})'
        )


class BehaviorImplementation(NamedElement):
    """A behaviorImplementation represents the body of a behavior associated with a class.

    Args:
        name (str): The name of the behavior implementation.
        
    Attributes:
        name (str): The name of the behavior implementation.  
    """
    
    def __init__(self, name: str):
        super().__init__(name)


    def __repr__(self):
        return f'BehaviorImplementation({self.name})'


class BehaviorDeclaration(NamedElement):
    """A BehaviorDeclaration represents the signature of a behavior associated with a class.

    Args:
        name (str): The name of the behavior.
        implementations (set[BehaviorImplementation]): The implementations associated with the behavior.
        
    Attributes:
        name (str): The name of the behavior.
        implementations (set[BehaviorImplementation]): The implementations associated with the behavior.
    """
    
    def __init__(self, name: str, implementations: set[BehaviorImplementation]):
        super().__init__(name)
        self.implementations: set[BehaviorImplementation] = implementations


    @property
    def implementations(self) -> set[BehaviorImplementation]:
        """set[BehaviorImplementation]: Get the implementations of the behavior."""
        return self.__implementations


    @implementations.setter
    def implementations(self, implementations: set[BehaviorImplementation]):
        """
        set[BehaviorImplementation]: Set the implementations of the behavior.

        Raises:
            ValueError: if two implementations have the same name.
        """
        if implementations is not None:
            names = [implementation.name for implementation in implementations]
            if len(names) != len(set(names)):
                raise ValueError("A behavior cannot have two implementations with the same name")
            self.__implementations = implementations
        else:
            self.__implementations = set()

    def __repr__(self):
        return f'BehaviorDeclaration({self.name}, {self.implementations})'


class Class(Type):
    """Represents a class in a modeling context.

    A Class is a type that defines a blueprint for objects. It can have attributes, associations,
    and generalizations with other classes.

    Args:
        name (str): The name of the class.
        attributes (set[Property]): The set of attributes associated with the class.
        behaviors (set[BehaviorDeclaration]): The set of behaviors associated with the class (None as default).
        is_abstract (bool): Indicates whether the class is abstract.
        is_read_only (bool): Indicates whether the class is read only.

    Attributes:
        name (str): Inherited from Type, represents the name of the class.
        attributes (set[Property]): The set of attributes associated with the class.
        behaviors (set[BehaviorDeclaration]): The set of behaviors associated with the class (None as default).
        is_abstract (bool): Indicates whether the class is abstract.
        is_read_only (bool): Indicates whether the class is read only.
        attributes (set[Property]): The set of attributes associated with the class (set() as default).
        methods (set[Method]): The set of methods of the class (set() as default).
        is_abstract (bool): Indicates whether the class is abstract (False as default).
        is_read_only (bool): Indicates whether the class is read only (False as default).
        behaviors (set[BehaviorDeclaration]): The set of behaviors associated with the class (None as default).
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the class (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).

    Attributes:
        name (str): Inherited from Type, represents the name of the class.
        attributes (set[Property]): The set of attributes associated with the class (set() as default).
        methods (set[Method]): The set of methods of the class (set() as default).
        is_abstract (bool): Indicates whether the class is abstract (False as default).
        is_read_only (bool): Indicates whether the class is read only (False as default).
        behaviors (set[BehaviorDeclaration]): The set of behaviors associated with the class (None as default).
        __associations (set[Association]): Set of associations involving the class.
        __generalizations (set[Generalization]): Set of generalizations involving the class.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the class (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, attributes: set[Property] = None, methods: set[Method] = None,
                 is_abstract: bool= False, is_read_only: bool= False, behaviors: set[BehaviorDeclaration] = None,
                 timestamp: int = None, synonyms: List[str] = None, is_derived: bool = False):
        super().__init__(name, timestamp, synonyms, is_derived=is_derived)
        self.is_abstract: bool = is_abstract
        self.is_read_only: bool = is_read_only
        self.behaviors: set[BehaviorDeclaration] = behaviors if behaviors is not None else set()
        self.attributes: set[Property] = attributes if attributes is not None else set()
        self.methods: set[Method] = methods if methods is not None else set()
        self.__associations: set[Association] = set()
        self.__generalizations: set[Generalization] = set()

    @property
    def attributes(self) -> set[Property]:
        """set[Property]: Get the attributes of the class."""
        return self.__attributes

    @attributes.setter
    def attributes(self, attributes: set[Property]):
        """
        set[Property]: Set the attributes of the class.
        
        Raises:
            ValueError: if two attributes have the same name.
            ValueError: if two attributes are id.
        """
        if attributes is not None:
            names_seen = set()
            duplicates = set()
            id_counter = 0

            for attribute in attributes:
                if attribute.name in names_seen:
                    duplicates.add(attribute.name)
                names_seen.add(attribute.name)

                if attribute.is_id:
                    id_counter += 1

            if duplicates:
                raise ValueError(f"A class cannot have attributes with duplicate names: {', '.join(duplicates)}")

            if id_counter > 1:
                raise ValueError("A class cannot have more than one attribute marked as 'id'")

            for attribute in attributes:
                if attribute.owner and attribute.owner != self:
                    attribute.owner.attributes.discard(attribute)
                attribute.owner = self
            self.__attributes = attributes
        else:
            self.__attributes = set()

    @property
    def methods(self) -> set[Method]:
        """set[Method]: Get the methods of the class."""
        return self.__methods

    @methods.setter
    def methods(self, methods: set[Method]):
        """
        set[Method]: Set the methods of the class.
        
        Raises:
            ValueError: if two methods have the same name.
        """
        if methods is not None:
            names_seen = set()
            duplicates = set()

            for method in methods:
                if method.name in names_seen:
                    duplicates.add(method.name)
                names_seen.add(method.name)

            if duplicates:
                raise ValueError(f"A class cannot have methods with duplicate names: {', '.join(duplicates)}")

            for method in methods:
                method.owner = self
            self.__methods = methods
        else:
            self.__methods = set()

    def add_method(self, method: Method):
        """
        Method: Add a method to the set of class methods.
        
        Raises:
            ValueError: if the method name already exist.
        """
        if self.methods is not None:
            if method.name in [method.name for method in self.methods]:
                raise ValueError(f"A class cannot have two methods with the same name: '{method.name}'")
        method.owner = self
        self.methods.add(method)

    def all_attributes(self) -> set[Property]:
        """set[Property]: Get all attributes, including inherited ones."""
        inherited_attributes: set[Property] = self.inherited_attributes()
        return self.__attributes | inherited_attributes

    def add_attribute(self, attribute: Property):
        """
        Property: Add an attribute to the set of class attributes.
        
        Raises:
            ValueError: if the attribute name already exist.
        """
        if self.attributes is not None:
            if attribute.name in [attribute.name for attribute in self.attributes]:
                raise ValueError(f"A class cannot have two attributes with the same name: '{attribute.name}'")
        if attribute.owner and attribute.owner != self:
            attribute.owner.attributes.discard(attribute)
        attribute.owner = self
        self.attributes.add(attribute)

    @property
    def behaviors(self) -> set[BehaviorDeclaration]:
        """set[BehaviorDeclaration]: Get the behaviors associated with the class."""
        return self.__behaviors

    @behaviors.setter
    def behaviors(self, behaviors: set[BehaviorDeclaration]):
        """
        set[BehaviorDeclaration]: Set the behaviors associated with the class.
        
        Raises:
            ValueError: if two behaviors have the same name.
        """
        if behaviors is not None:
            names = [behavior.name for behavior in behaviors]
            if len(names) != len(set(names)):
                raise ValueError("A class cannot have two behaviors with the same name")
            self.__behaviors = behaviors
        else:
            self.__behaviors = set()

    @property
    def is_abstract(self) -> bool:
        """bool: Get whether the class is abstract."""
        return self.__is_abstract

    @is_abstract.setter
    def is_abstract(self, is_abstract: bool):
        """bool: Set whether the class is abstract."""
        self.__is_abstract = is_abstract

    @property
    def is_read_only(self) -> bool:
        """bool: Get whether the class is read only."""
        return self.__is_read_only

    @is_read_only.setter
    def is_read_only(self, is_read_only: bool):
        """bool: Set whether the class is read only."""
        self.__is_read_only = is_read_only

    @property
    def associations(self) -> set:
        """set[Association]: Get the set of associations involving the class."""
        return self.__associations

    def _add_association(self, association):
        """Association: Add an association to the set of class associations."""
        self.__associations.add(association)

    def _delete_association(self, association):
        """Association: Remove an association to the set of class associations."""
        self.__associations.discard(association)

    @property
    def generalizations(self) -> set:
        """set[Generalization]: Get the set of generalizations involving the class."""
        return self.__generalizations

    def _add_generalization(self, generalization):
        """Generalization: Add a generalization to the set of class generalizations."""
        self.__generalizations.add(generalization)

    def _delete_generalization(self, generalization):
        """Generalization: Remove a generalization to the set of class generalizations."""
        self.__generalizations.discard(generalization)

    def inherited_attributes(self) -> set[Property]:
        """set[Property]: Get the set of inherited attributes."""
        inherited_attributes = set()
        for parent in self.all_parents():
            inherited_attributes.update(parent.attributes)
        return inherited_attributes

    def association_ends(self) -> set:
        """set[Property]: Get the set of association ends of the class."""
        ends = set()
        for association in self.__associations:
            aends = association.ends
            ends.update(aends)
            l_aends = list(aends)
            if not(len(l_aends) == 2 and l_aends[0].type == l_aends[1].type):
                for end in aends:
                    if end.type == self:
                        ends.discard(end)
        return ends

    def all_association_ends(self) -> set[Property]:
        """set[Property]: Get the set of direct and indirect association ends of the class."""
        all_ends = self.association_ends()
        for parent in self.all_parents():
            ends = parent.association_ends()
            all_ends.update(ends)
        return all_ends

    def parents(self) -> set:
        """set[Class]: Get the set of direct parents of the class."""
        parents = set()
        for generalization in self.__generalizations:
            if generalization.general != self:
                parents.add(generalization.general)
        return parents

    def all_parents(self) -> set:
        """set[Class]: Get the set of direct and indirect parents of the class."""
        all_parents = set()
        all_parents.update(self.parents())
        for parent in self.parents():
            all_parents.update(parent.all_parents())
        return all_parents

    def specializations(self) -> set:
        """set[Class]: Get the set of direct specializations (children) of the class."""
        specializations = set()
        for generalization in self.__generalizations:
            if generalization.specific != self:
                specializations.add(generalization.specific)
        return specializations

    def all_specializations(self) -> set:
        """set[Class]: Get the set of direct and indirect specializations (children) of the class."""
        all_spec = set()
        all_spec.update(self.specializations())
        for specialization in self.specializations():
            all_spec.update(specialization.all_specializations())
        return all_spec

    def id_attribute(self) -> Property:
        """Property: Get the id attribute of the class."""
        for attribute in self.attributes:
            if attribute.is_id:
                return attribute
        return None

    def __repr__(self):
        return (
                f'Class({self.name}, {self.attributes}, {self.methods}, {self.timestamp}, {self.synonyms}, '
                f'is_abstract={self.is_abstract}, is_derived={self.is_derived})'
        )


class Association(NamedElement):
    """Represents an association between classes.

    An Association defines a relationship between classes and is composed of two or more ends,
    each associated with a class. An association must have more than one end.

    Args:
        name (str): The name of the association.
        ends (set[Property]): The set of ends related to the association.
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the association (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
        
    Attributes:
        name (str): Inherited from NamedElement, represents the name of the association.
        ends (set[Property]): The set of ends related to the association.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the association (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, ends: set[Property], timestamp: int = None, synonyms: List[str] = None,
                is_derived: bool = False):
        super().__init__(name, timestamp, synonyms, is_derived=is_derived)
        self.ends: set[Property] = ends

    @property
    def ends(self) -> set[Property]:
        """set[Property]: Get the ends of the association."""
        return self.__ends

    @ends.setter
    def ends(self, ends: set[Property]):
        """
        set[Property]: Set the ends of the association. Two or more ends are required.
        
        Raises:
            ValueError: if an association has less than two ends.
        """
        if len(ends) <= 1:
            raise ValueError("An association must have more than one end")
        if hasattr(self, "ends"):
            for end in self.ends:
                end.type._delete_association(association=self)
        for end in ends:
            end.owner = self
            end.type._add_association(association=self)
        self.__ends = ends

    def __repr__(self):
        return f'Association({self.name}, {self.ends}, {self.timestamp}, {self.synonyms}, is_derived={self.is_derived})'

class BinaryAssociation(Association):
    """Represents a binary association between two classes.

    A BinaryAssociation is a specialized form of Association that specifically involves
    two ends, each associated with a class. It enforces constraints on the association,
    such as having exactly two ends. Exactly two ends are required 

    Args:
        name (str): The name of the binary association.
        ends (set[Property]): The set of ends related to the binary association.
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the binary association (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).

    Attributes:
        name (str): Inherited from Association, represents the name of the binary association.
        ends (set[Property]): Inherited from NamedElement, represents the set of ends related to the binary association.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the binary association (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    @Association.ends.setter
    def ends(self, ends: set[Property]):
        """set[Property]: Set the ends of the association.
        
        Raises:
            ValueError: if the associaiton ends are not exactly two, or if both ends are tagged as agregation, or 
            if both ends are tagged as composition.
        """
        if len(ends) != 2:
            raise ValueError("A binary association must have exactly two ends")
        if list(ends)[0].is_composite is True and list(ends)[1].is_composite is True:
            raise ValueError("The composition attribute cannot be tagged at both ends")
        super(BinaryAssociation, BinaryAssociation).ends.fset(self, ends)

    def __repr__(self):
        return f'BinaryAssociation({self.name}, {self.ends}, {self.timestamp}, {self.synonyms}, is_derived={self.is_derived})'

class AssociationClass(Class):
    # Class that has an association nature
    """An AssociationClass is a class that that has an association nature.
    It inherits from Class and is associated with an underlying Association.

    Args:
        name (str): The name of the association class.
        attributes (set[Property]): The set of attributes associated with the association class.
        association (Association): The underlying association linked to the association class.
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the association class (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).

    Attributes:
        name (str): Inherited from Class, represents the name of the association class.
        attributes (set[Property]): Inherited from Class, represents the set of attributes associated with the association class.
        association (Association): The underlying association linked to the association class.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the association class (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, attributes: set[Property], association: Association, timestamp: int = None,
                 synonyms: List[str] = None, is_derived: bool = False):
        super().__init__(name, attributes, timestamp, synonyms, is_derived=is_derived)
        self.association: Association = association

    @property
    def association(self) -> Association:
        """Association: Get the underlying association of the association class."""
        return self.__association

    @association.setter
    def association(self, association: Association):
        """Association: Set the underlying association of the association class."""
        self.__association = association

    def __repr__(self):
        return (
            f'AssociationClass({self.name}, {self.attributes}, {self.association}, {self.timestamp}, '
            f'{self.synonyms}, is_derived={self.is_derived})'
        )

class Generalization(Element):
    """Represents a generalization relationship between two classes.

    A Generalization is a relationship between two classes, where one class (specific)
    inherits attributes and behaviors from another class (general).

    Args:
        general (Class): The general (parent) class in the generalization relationship.
        specific (Class): The specific (child) class in the generalization relationship.
        timestamp (datetime): Object creation datetime (default is current time).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    
    Attributes:
        general (Class): The general (parent) class in the generalization relationship.
        specific (Class): The specific (child) class in the generalization relationship.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, general: Class, specific: Class, timestamp: int = None, is_derived: bool = False):
        super().__init__(timestamp, is_derived)
        self.general: Class = general
        self.specific: Class = specific

    @property
    def general(self) -> Class:
        """Class: Get the general (parent) class."""
        return self.__general

    @general.setter
    def general(self, general: Class):
        """Class: Set the general (parent) class."""
        if hasattr(self, "general"):
            self.general._delete_generalization(generalization=self)
        general._add_generalization(generalization=self)
        self.__general = general

    @property
    def specific(self) -> Class:
        """Class: Get the specific (child) class."""
        return self.__specific

    @specific.setter
    def specific(self, specific: Class):
        """
        Class: Set the specific (child) class.
        
        Raises:
            ValueError: if the general class is equal to the specific class
        """
        if specific == self.general:
            raise ValueError("A class cannot be a generalization of itself")
        if hasattr(self, "specific"):
            self.specific._delete_generalization(generalization=self)
        specific._add_generalization(generalization=self)
        self.__specific = specific

    def __repr__(self):
        return f'Generalization({self.general}, {self.specific}, {self.timestamp}, is_derived={self.is_derived})'

class GeneralizationSet(NamedElement):
    """Represents a set of generalization relationships.

    Args:
        name (str): The name of the generalization set.
        generalizations (set[Generalization]): The set of generalization relationships in the set.
        is_disjoint (bool): Indicates whether the set is disjoint (instances cannot belong to more than one class
            in the set).
        is_complete (bool): Indicates whether the set is complete (every instance of the superclass must belong to
            a subclass).
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the generalization set (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the generalization set.
        generalizations (set[Generalization]): The set of generalization relationships in the set.
        is_disjoint (bool): Indicates whether the set is disjoint (instances cannot belong to more than one class
            in the set).
        is_complete (bool): Indicates whether the set is complete (every instance of the superclass must belong to
            a subclass).
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the generalization set (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, generalizations: set[Generalization], is_disjoint: bool, is_complete: bool,
                timestamp: int = None, synonyms: List[str] = None, is_derived: bool = False):
        super().__init__(name, timestamp, synonyms, is_derived=is_derived)
        self.generalizations: set[Generalization] = generalizations
        self.is_disjoint: bool = is_disjoint
        self.is_complete: bool = is_complete

    @property
    def generalizations(self) -> set[Generalization]:
        """set[Generalization]: Get the generalization relationships."""
        return self.__generalizations

    @generalizations.setter
    def generalizations(self, generalizations: set[Generalization]):
        """set[Generalization]: Set the generalization relationships."""
        self.__generalizations = generalizations

    @property
    def is_disjoint(self) -> bool:
        """bool: Get whether the set is disjoint."""
        return self.__is_disjoint

    @is_disjoint.setter
    def is_disjoint(self, is_disjoint: bool):
        """bool: Set whether the set is disjoint."""
        self.__is_disjoint = is_disjoint

    @property
    def is_complete(self) -> bool:
        """bool: Get whether the set is complete."""
        return self.__is_complete

    @is_complete.setter
    def is_complete(self, is_complete: bool):
        """bool: Set whether the set is complete."""
        self.__is_complete = is_complete

    def __repr__(self):
        return (
            f'GeneralizationSet({self.name}, {self.generalizations}, '
            f'is_disjoint={self.is_disjoint}, is_complete={self.is_complete}, {self.timestamp}, '
            f'{self.synonyms}, is_derived={self.is_derived})'
        )

class Package(NamedElement):
    """A Package is a grouping mechanism that allows organizing and managing a set of NamedElements.

    Attributes:
        name (str): The name of the package.
        elements (set[NamedElement]): The set of elements contained in the package.
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the package (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    
    Attributes:
        name (str): Inherited from NamedElement, represents the name of the package.
        elements (set[NamedElement]): The set of elements contained in the package.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the package (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, elements: set[NamedElement], timestamp: int = None, synonyms: List[str] = None,
                is_derived: bool = False):
        super().__init__(name, timestamp, synonyms, is_derived=is_derived)
        self.elements: set[NamedElement] = elements

    @property
    def elements(self) -> set[NamedElement]:
        """set[NamedElement]: Get the named elements contained in the package."""
        return self.__elements

    @elements.setter
    def elements(self, elements: set[NamedElement]):
        """set[NamedElement]: Set the named elements contained in the package."""
        self.__elements = elements

    def get_classes(self) -> set[Class]:
        """set[Class]: Get all classes within the package."""
        return {element for element in self.elements if isinstance(element, Class)}

    def get_associations(self) -> set[Association]:
        """set[Association]: Get all associations within the package."""
        return {element for element in self.elements if isinstance(element, Association)}

    def get_generalizations(self) -> set[Generalization]:
        """set[Generalization]: Get all generalizations within the package."""
        return {element for element in self.elements if isinstance(element, Generalization)}

    def get_enumerations(self) -> set[Enumeration]:
        """set[Enumeration]: Get all enumerations within the package."""
        return {element for element in self.elements if isinstance(element, Enumeration)}

    def __repr__(self):
        return f'Package({self.name}, {self.elements}), {self.timestamp}, {self.synonyms}, is_derived={self.is_derived}'

class Constraint(NamedElement):
    """A Constraint is a statement that restricts or defines conditions on the behavior,
    structure, or other aspects of the modeled system.

    Args:
        name (str): The name of the constraint.
        context (Class): The class to which the constraint is associated.
        expression (str): The expression or condition defined by the constraint.
        language (str): The language in which the constraint expression is written.
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the constraint (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the constraint.
        context (Class): The class to which the constraint is associated.
        expression (str): The expression or condition defined by the constraint.
        language (str): The language in which the constraint expression is written.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the constraint (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
    """

    def __init__(self, name: str, context: Class, expression: Any, language: str, timestamp: int = None,
                 synonyms: List[str] = None, is_derived: bool = False):
        super().__init__(name, timestamp, synonyms, is_derived=is_derived)
        self.context: Class = context
        self.expression: str = expression
        self.language: str = language

    @property
    def context(self) -> Class:
        """Class: Get the class to which the constraint is associated."""
        return self.__context

    @context.setter
    def context(self, context: Class):
        """Class: Set the class to which the constraint is associated."""
        self.__context = context

    @property
    def expression(self) -> str:
        """str: Get the expression or condition defined by the constraint."""
        return self.__expression

    @expression.setter
    def expression(self, expression: Any):
        """str: Set the expression or condition defined by the constraint."""
        self.__expression = expression

    @property
    def language(self) -> str:
        """str: Get the language in which the constraint expression is written."""
        return self.__language

    @language.setter
    def language(self, language: str):
        """str: Set the language in which the constraint expression is written."""
        self.__language = language

    def __repr__(self):
        return (
            f'Constraint({self.name}, {self.context.name}, {self.language}, {self.expression}, '
            f'{self.timestamp}, is_derived={self.is_derived})'
        )

class Model(NamedElement):
    """A model is the root element. A model is the root element. There are different types of models
    that inherit from this class. For example, DomainModel, ObjectModel, or GUIModel.

    Args:
        name (str): The name of the model.
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the model (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
        elements (set[Element]): Set of model Elements in the Model.
        
    Attributes:
        name (str): Inherited from NamedElement, represents the name of the model.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the model (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
        elements (set[Element]): Set of model Elements in the Model.
    """
    def __init__(self, name: str, timestamp: int = None, synonyms: List[str] = None, is_derived: bool = False,
                elements: set[Element] = None):
        super().__init__(name, timestamp, synonyms, is_derived=is_derived)
        self.elements: set[Element] = elements if elements is not None else set()

    @property
    def elements(self) -> set[Element]:
        """set[Element]: Get the set of model elements in the model."""
        return self.__elements

    @elements.setter
    def elements(self, elements: set[Element]):
        """set[Element]: Set the set of model elements in the model."""
        if elements is not None:
            self.__elements = elements
        else:
            self.__elements = set()

class DomainModel(Model):
    """A domain model comprises a number of types, associations, 
    generalizations, packages, constraints, and others.

    Args:
        name (str): The name of the domain model.
        types (set[Type]): The set of types (classes and datatypes) in the domain model (set() as default).
        associations (set[Association]): The set of associations in the domain model (set() as default).
        generalizations (set[Generalization]): The set of generalizations in the domain model (set() as default).
        packages (set[Package]): The set of packages in the domain model (set() as default).
        constraints (set[Constraint]): The set of constraints in the domain model (set() as default).
        timestamp (datetime): Object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the domain model (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
        elements (set[Element]): Set of model Elements in the Model. This property is derived (auto-calculated).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the domain model.
        types (set[Type]): The set of types (classes and datatypes) in the domain model (set() as default).
        associations (set[Association]): The set of associations in the domain model (set() as default).
        generalizations (set[Generalization]): The set of generalizations in the domain model (set() as default).
        packages (set[Package]): The set of packages in the domain model (set() as default).
        constraints (set[Constraint]): The set of constraints in the domain model (set() as default).
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        synonyms (List[str]): List of synonyms of the domain model (None as default).
        is_derived (bool): Inherited from NamedElement, indicates whether the element is derived (False as default).
        elements (set[Element]): Set of model Elements in the Model. This property is derived (auto-calculated).
    """

    def __init__(self, name: str, types: set[Type] = None, associations: set[Association] = None,
                generalizations: set[Generalization] = None, packages: set[Package] = None,
                constraints: set[Constraint] = None, timestamp: int = None, synonyms: List[str] = None,
                is_derived: bool = False, elements: set[Element] = None):
        super().__init__(name, timestamp, synonyms, is_derived=is_derived, elements=elements)
        # A flag to prevent premature `_update_elements` calls during initialization
        self.__initializing = True
        self.types: set[Type] = types if types is not None else set()
        self.packages: set[Package] = packages if packages is not None else set()
        self.constraints: set[Constraint] = constraints if constraints is not None else set()
        self.associations: set[Association] = associations if associations is not None else set()
        self.generalizations: set[Generalization] = generalizations if generalizations is not None else set()
        # initialization is done, we update elements
        self.__initializing = False
        self._update_elements()

    def _update_elements(self):
        """Recalculates the elements property by combining all relevant sets."""
        if not self.__initializing:
            self.elements = set(self.types) | set(self.packages) | set(self.constraints) \
                            | set(self.associations) | set(self.generalizations)

    @property
    def types(self) -> set[Type]:
        """set[Type]: Get the set of types in the domain model."""
        return self.__types

    @types.setter
    def types(self, types: set[Type]):
        """
        set[Type]: Set the set of types in the domain model, including primitive data types.
        
        Raises:
            ValueError: if there are two types with the same name.
        """
        types = types | data_types
        names_seen = set()
        duplicates = set()

        for type_ in types:
            if type_.name in names_seen:
                duplicates.add(type_.name)
            names_seen.add(type_.name)

        if duplicates:
            raise ValueError(f"The model cannot have types with duplicate names: {', '.join(duplicates)}")
        self.__types = types
        self._update_elements()

    def get_type_by_name(self, type_name: str) -> Type:
        """Type: Gets an Type by name."""
        return next(
            (type_element for type_element in self.types if type_element.name == type_name), None
            )

    def add_type(self, type_: Type):
        """Type: Add a type (Class or DataType) to the set of types of the model."""
        self.types = self.types | {type_}

    @property
    def associations(self) -> set[Association]:
        """set[Association]: Get the set of associations in the domain model."""
        return self.__associations

    @associations.setter
    def associations(self, associations: set[Association]):
        """
        set[Association]: Set the set of associations in the domain model.
        
        Raises:
            ValueError: if there are two associations with the same name.
        """
        if associations is not None:
            names_seen = set()
            duplicates = set()

            for association in associations:
                if association.name in names_seen:
                    duplicates.add(association.name)
                names_seen.add(association.name)

            if duplicates:
                raise ValueError(f"The model cannot have associations with duplicate names: {', '.join(duplicates)}")

            self.__associations = associations
        else:
            self.__associations = set()

        self._update_elements()

    def add_association(self, association: Association):
        """Association: Add an association to the set of associations of the model."""
        self.associations = self.associations | {association}

    @property
    def generalizations(self) -> set[Generalization]:
        """set[Generalization]: Get the set of generalizations in the domain model."""
        return self.__generalizations

    @generalizations.setter
    def generalizations(self, generalizations: set[Generalization]):
        """set[Generalization]: Set the set of generalizations in the domain model."""
        if generalizations is not None:
            self.__generalizations = generalizations
        else:
            self.__generalizations = set()

        self._update_elements()

    def add_generalization(self, generalization: Generalization):
        """Generalization: Add a generalization to the set of generalizations of the model."""
        self.generalizations = self.generalizations | {generalization}

    def get_enumerations(self) -> set[Enumeration]:
        """set[Enumeration]: Get the set of enumerations in the domain model."""
        return {element for element in self.types if isinstance(element, Enumeration)}

    @property
    def packages(self) -> set[Package]:
        """set[Package]: Get the set of packages in the domain model."""
        return self.__packages

    @packages.setter
    def packages(self, packages: set[Package]):
        """
        set[Package]: Get the set of packages in the domain model.
        
        Raises:
            ValueError: if there are two packages with the same name.
        """
        if packages is not None:
            names_seen = set()
            duplicates = set()

            for package in packages:
                if package.name in names_seen:
                    duplicates.add(package.name)
                names_seen.add(package.name)

            if duplicates:
                raise ValueError(f"The model cannot have packages with duplicate names: {', '.join(duplicates)}")

            self.__packages = packages
        else:
            self.__packages = set()

        self._update_elements()

    @property
    def constraints(self) -> set[Constraint]:
        """set[Constraint]: Get the set of constraints in the domain model."""
        return self.__constraints

    @constraints.setter
    def constraints(self, constraints: set[Constraint]):
        """
        set[Constraint]: Get the set of constraints in the domain model.
        
        Raises:
            ValueError: if there are two constraints with the same name.
        """
        if constraints is not None:
            names = [constraint.name for constraint in constraints]
            if len(names) != len(set(names)):
                raise ValueError("The model cannot have two constraints with the same name")
            self.__constraints = constraints
        else:
            self.__constraints = set()

        self._update_elements()

    def get_classes(self) -> set[Class]:
        """set[Class]: Get all classes within the domain model."""
        return {element for element in self.types if isinstance(element, Class)}

    def get_class_by_name(self, class_name: str) -> Class:
        """Class: Gets a class by name."""
        return next(
            (element for element in self.types if isinstance(element, Class) and
             element.name == class_name), None
            )

    def classes_sorted_by_inheritance(self) -> list[Class]:
        """list[Class]: Get the list of classes ordered by inheritance."""
        from besser.utilities import sort_by_timestamp
        classes = sort_by_timestamp(self.get_classes())
        # Set up a dependency graph
        child_map = {cl: set() for cl in classes}
        # Populating the child_map based on generalizations (edges in top-sort graph)
        for cl in classes:
            for generalization in cl.generalizations:
                child_map[generalization.general].add(cl)
        # Helper function for DFS
        def dfs(cl, visited, sorted_list):
            visited.add(cl)
            for child in child_map[cl]:
                if child not in visited:
                    dfs(child, visited, sorted_list)
            sorted_list.append(cl)
        # Perform DFS from each node that hasn't been visited yet
        visited = set()
        sorted_list = []
        for cl in classes:
            if cl not in visited:
                dfs(cl, visited, sorted_list)
        sorted_list.reverse()
        return sorted_list

    def __repr__(self):
        return (
            f'DomainModel({self.name}, {self.types}, {self.associations}, {self.generalizations}, '
            f'{self.packages}, {self.constraints}, {self.timestamp}, {self.synonyms})'
            f'is_derived={self.is_derived}'
        )
