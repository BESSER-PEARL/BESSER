from abc import ABC, abstractmethod

# constant
UNLIMITED_MAX_MULTIPLICITY = 9999


class Element(ABC):
    pass


# Superclass of all structural elements with a name
class NamedElement(Element):

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

    def __repr__(self):
        return f"Name({self.name})"


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
        # calling the setter of the superclass if there are no errors
        super(PrimitiveDataType, PrimitiveDataType).name.fset(self, name)


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
        if max_multiplicity < self.min:
            raise ValueError("Invalid max multiplicity")
        self.__max = max_multiplicity

    def __repr__(self):
        return f'Multiplicity({self.min},{self.max})'


# Properties are owned by a class or an association and point to a type with a multiplicity
class Property(NamedElement):

    def __init__(self, name: str, owner: Type, property_type: Type, multiplicity: Multiplicity):
        super().__init__(name)
        self.owner: Type = owner
        self.type: Type = property_type
        self.multiplicity: Multiplicity = multiplicity

    @property
    def owner(self) -> Type:
        return self.__owner

    @owner.setter
    def owner(self, owner: Type):
        # owner cannot be a datatype
        if isinstance(owner, DataType):
            raise ValueError("Invalid owner")
        self.__owner = owner

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
        return f'Property({self.name},{self.owner},{self.type},{self.multiplicity})'


class Class(Type):

    def __init__(self, name: str, attributes: set[Property]):
        super().__init__(name)
        self.attributes: set[Property] = attributes


    @property
    def attributes(self) -> set[Property]:
        return self.__attributes

    @attributes.setter
    def attributes(self, attributes: set[Property]):
        if(attributes is not None):
            names = [attribute.name for attribute in attributes]
            if len(names) != len(set(names)):
                raise ValueError("A class cannot have two attributes with the same name")
            for attribute in attributes:
                attribute.owner = self
            self.__attributes = attributes

    def __repr__(self):
        return f'Class({self.name},{self.attributes})'


class Association(NamedElement):
    def __init__(self, name: str, ends: set[Property]):
        super().__init__(name)
        self.ends: set[Property] = ends

    @property
    def ends(self) -> set[Property]:
        return self.__ends

    @ends.setter
    def ends(self, ends: set[Property]):
        if len(ends) <= 1:
            raise ValueError("An association must have more than one end")
        for end in ends:
            end.owner = self
        self.__ends = ends


class BinaryAssociation(Association):
    def __init__(self, name: str, ends: set[Property]):
        super().__init__(name, ends)

    @Association.ends.setter
    def ends(self, ends: set[Property]):
        if len(ends) != 2:
            raise ValueError("A binary must have exactly two ends")
        super(BinaryAssociation, BinaryAssociation).ends.fset(self, ends)


class AssociationClass(Class):
    # Class that has an association nature
    def __init__(self, name: str, attributes: set[Property], references: set[Property], association: Association):
        super().__init__(name, attributes)
        self.association: Association = association

    @property
    def association(self) -> Association:
        return self.__association

    @association.setter
    def association(self, association: Association):
        self.__association = association


class Generalization(Element):
    # Generalization between two classes
    def __init__(self, general: Class, specific: Class):
        self.general: Class = general
        self.specific: Class = specific

    @property
    def general(self) -> Class:
        return self.__general

    @general.setter
    def general(self, general: Class):
        self.__general = general

    @property
    def specific(self) -> Class:
        return self.__specific

    @specific.setter
    def specific(self, specific: Class):
        # specific cannot be the same class as general
        if specific == self.general:
            raise ValueError("you cannot have your own parent")
        self.__specific = specific

    def __repr__(self):
        return f'Generalization({self.general},{self.specific})'


class GeneralizationSet(NamedElement):
    # set of generalization relationships
    def __init__(self, name: str, generalizations: set[Generalization], is_disjoint: bool, is_complete: bool):
        super().__init__(name)
        self.generalizations: set[Generalization] = generalizations
        self.is_disjoint: bool = is_disjoint
        self.is_complete: bool = is_complete

    @property
    def generalizations(self) -> set[Generalization]:
        return self.__generalizations

    @generalizations.setter
    def generalizations(self, generalizations: set[Generalization]):
        self.__generalizations = generalizations

    @property
    def is_disjoint(self) -> bool:
        return self.__is_disjoint

    @is_disjoint.setter
    def is_disjoint(self, is_disjoint: bool):
        self.__is_disjoint = is_disjoint

    @property
    def is_complete(self) -> bool:
        return self.__is_complete

    @is_complete.setter
    def is_complete(self, is_complete: bool):
        self.__is_complete = is_complete


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
        # Check no duplicate names
        # Get a list of names from the elements
        names = [element.name for element in elements]
        if len(names) != len(set(names)):
            raise ValueError("The model cannot have two types with the same name")
        self.__elements = elements

    def get_classes(self) -> set[Class]:
        return {element for element in self.elements if isinstance(element, Class)}

    def get_associations(self) -> set[Association]:
        return {element for element in self.elements if isinstance(element, Association)}




