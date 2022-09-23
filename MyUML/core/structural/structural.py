#constant
UNLIMITED_MAX_MULTIPLICITY = 9999


# Superclass of all structural elements with a name
class NamedElement:

    def __init__(self, name: str):
        self.name: str = name

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name


# Superclass of classes and data types in the model
class Type(NamedElement):

    def __init__(self, name: str):
        super().__init__(name)


class DataType(NamedElement):
    def __init__(self, name: str):
        super().__init__(name)


class PrimitiveDataType(DataType):
    def __init__(self, name: str):
        super().__init__(name)

    @NamedElement.name.setter
    def name(self, name: str):
        if name not in ['int', 'float', 'str', 'bool']:
            raise ValueError("Invalid primitive data type")
        self.name = name


# Min and max multiplicities of a Property
class Multiplicity:
    def __init__(self, min_multiplicity: int, max_multiplicity: int):
        self.min: int = min_multiplicity
        self.max: int = max_multiplicity

    @property
    def min(self) -> int:
        return self.__min

    @min.setter
    def min(self, min_multiplicity: int):
        if min_multiplicity < 0:
            raise ValueError("Invalid min multiplicity")
        self.__min = min_multiplicity

    @property
    def max(self) -> int:
        return self.__max

    @max.setter
    def max(self, max_multiplicity: int):
        if max_multiplicity < 0:
            raise ValueError("Invalid max multiplicity")
        self.__max = max_multiplicity

    def __repr__(self):
        return f'Multiplicity({self.min},{self.max})'


# Properties are owned by a class or an association and point to a type with a multiplicity
class Property(NamedElement):

    def __init__(self, name: str, property_type: Type, multiplicity: Multiplicity):
        super().__init__(name)
        self.type = property_type
        self.multiplicity: Multiplicity = multiplicity

    @property
    def type(self) -> Type:
        return self.__type

    @type.setter
    def type(self, property_type: Type):
        self.__type = property_type

    @property
    def multiplicity(self) -> Multiplicity:
        return self.__multiplicity

    @multiplicity.setter
    def multiplicity(self, multiplicity: Multiplicity):
        self.__multiplicity = multiplicity

    def __repr__(self):
        return f'Property({self.name},{self.type},{self.multiplicity})'


class Class(Type):
    pass


# A model is the root element that comprises a number of classes and associations
class DomainModel(NamedElement):

    def __init__(self, name: str, elements: set[Type]):
        super().__init__(name)
        self.elements: set[Type] = elements

    @property
    def elements(self) -> set[Type]:
        return self.__elements

    @elements.setter
    def elements(self, elements: set[Type]):
        self.__elements = elements
        #Check no duplicate names
        #Get a list of names from the elements
        names = [element.name for element in elements]
        if len(names) != len(set(names)):
            raise ValueError("The model cannot have two types with the same name")
        self.__elements = elements