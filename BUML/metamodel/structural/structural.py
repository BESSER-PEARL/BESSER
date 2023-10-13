from abc import ABC, abstractmethod
from typing import Any
 
# constant
UNLIMITED_MAX_MULTIPLICITY = 9999

class Element(ABC):
    pass

# Superclass of all structural elements with a name
class NamedElement(Element):

    def __init__(self, name: str, visibility: str = "public"):
        self.name: str = name
        self.visibility = visibility

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def visibility(self) -> str:
        return self.__visibility

    @visibility.setter
    def visibility(self, visibility: str):
        if visibility not in ['public', 'private', 'protected', 'package']:
            raise ValueError("Invalid visibility")
        self.__visibility = visibility

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
        if name not in ['int', 'float', 'str', 'bool', 'time', 'date', 'datetime', 'timedelta']:
            raise ValueError("Invalid primitive data type")
        # calling the setter of the superclass if there are no errors
        super(PrimitiveDataType, PrimitiveDataType).name.fset(self, name)


class TypedElement(NamedElement):
    def __init__(self, name: str, type: Type):
        super().__init__(name)
        self.type: Type = type

    @property
    def type(self) -> Type:
        return self.__type

    @type.setter
    def type(self, type: Type):
        self.__type = type


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
        if max_multiplicity == "*":
            max_multiplicity = UNLIMITED_MAX_MULTIPLICITY
        if max_multiplicity < 0:
            raise ValueError("Invalid max multiplicity")
        if max_multiplicity < self.min:
            raise ValueError("Invalid max multiplicity")
        self.__max = max_multiplicity

    def __repr__(self):
        return f'Multiplicity({self.min},{self.max})'


# Properties are owned by a class or an association and point to a type with a multiplicity
class Property(TypedElement):

    def __init__(self, name: str, owner: Type, property_type: Type, multiplicity: Multiplicity = Multiplicity(1, 1), visibility: str = 'public', is_composite: bool = False, is_navigable: bool = True, is_aggregation: bool = False):
        super().__init__(name, visibility)
        self.owner: Type = owner
        self.type: Type = property_type
        self.multiplicity: Multiplicity = multiplicity
        self.is_composite: bool = is_composite
        self.is_navigable: bool = is_navigable
        self.is_aggregation: bool = is_aggregation

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
    def multiplicity(self) -> Multiplicity:
        return self.__multiplicity

    @multiplicity.setter
    def multiplicity(self, multiplicity: Multiplicity):
        self.__multiplicity = multiplicity

    @property
    def is_composite(self) -> bool:
        return self.__is_composite

    @is_composite.setter
    def is_composite(self, is_composite: bool):
        self.__is_composite = is_composite

    @property
    def is_navigable(self) -> bool:
        return self.__is_navigable

    @is_navigable.setter
    def is_navigable(self, is_navigable: bool):
        self.__is_navigable = is_navigable

    @property
    def is_aggregation(self) -> bool:
        return self.__is_aggregation

    @is_aggregation.setter
    def is_aggregation(self, is_aggregation: bool):
        self.__is_aggregation = is_aggregation

    def __repr__(self):
        return f'Property({self.name},{self.visibility},{self.owner},{self.type},{self.multiplicity},{self.is_composite})'


class Class(Type):

    def __init__(self, name: str, attributes: set[Property], is_abstract: bool= False):
        super().__init__(name)
        self.is_abstract: bool = is_abstract
        self.attributes: set[Property] = attributes

    @property
    def attributes(self) -> set[Property]:
        return self.__attributes

    @attributes.setter
    def attributes(self, attributes: set[Property]):
        if attributes is not None:
            names = [attribute.name for attribute in attributes]
            if len(names) != len(set(names)):
                raise ValueError("A class cannot have two attributes with the same name")
            for attribute in attributes:
                attribute.owner = self
            self.__attributes = attributes
        else:
            self.__attributes = set()

    #add attribute method
    def add_attribute(self, attribute: Property):
        if self.attributes is not None:
            if attribute.name in [attribute.name for attribute in self.attributes]:
                raise ValueError("A class cannot have two attributes with the same name")
        if attribute.owner is None:
            attribute.owner = self   
        self.attributes.add(attribute)
    
    @property
    def is_abstract(self) -> bool:
        return self.__is_abstract

    @is_abstract.setter
    def is_abstract(self, is_abstract: bool):
        self.__is_abstract = is_abstract

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
        if list(ends)[0].is_aggregation == True and list(ends)[1].is_aggregation == True:
            raise ValueError("The aggregation attribute cannot be tagged at both ends")
        if list(ends)[0].is_composite == True and list(ends)[1].is_composite == True:
            raise ValueError("The composition attribute cannot be tagged at both ends")
        super(BinaryAssociation, BinaryAssociation).ends.fset(self, ends)


class AssociationClass(Class):
    # Class that has an association nature
    # Note that Python does support multiple inheritance but we do not use it here as this is a diamon-shape structure
    # and we prefer the simpler solution to stick to single inheritance
    def __init__(self, name: str, attributes: set[Property], association: Association):
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


# A set of related classes that should be processed together
class Package(NamedElement):
    def __init__(self, name: str, classes: set[Class]):
        super().__init__(name)
        self.classes: set[Class] = classes

    @property
    def classes(self) -> set[Class]:
        return self.__classes

    @classes.setter
    def classes(self, classes: set[Class]):
        self.__classes = classes


# A constraint class to represent a constraint over a class
class Constraint(NamedElement):
    def __init__(self, name: str, context: Class, expression: Any, language: str):
        super().__init__(name)
        self.context: Class = context
        self.expression: str = expression
        self.language: str = language

    @property
    def context(self) -> Class:
        return self.__context

    @context.setter
    def context(self, context: Class):
        self.__context = context

    @property
    def expression(self) -> str:
        return self.__expression

    @expression.setter
    def expression(self, expression: Any):
        self.__expression = expression

    @property
    def language(self) -> str:
        return self.__language

    @language.setter
    def language(self, language: str):
        self.__language = language

    def __repr__(self):
        return f'Constraint({self.name},{self.context.name},{self.language},{self.expression})'


# A model is the root element that comprises a number of classes and associations
class DomainModel(NamedElement):

    def __init__(self, name: str, types: set[Type] = None, associations: set[Association] = None, generalizations: set[Generalization] = None, packages: set[Package] = None, constraints: set[Constraint] = None):
        super().__init__(name)
        self.types: set[Type] = types
        self.packages: set[Package] = packages
        self.constraints: set[Constraint] = constraints
        self.associations: set[Association] = associations
        self.generalizations: set[Generalization] = generalizations

    @property
    def types(self) -> set[Type]:
        return self.__types

    @types.setter
    def types(self, types: set[Type]):
        # Check no duplicate names
        if types is not None:
            # Get a list of names from the elements
            names = [type.name for type in types]
            if len(names) != len(set(names)):
                raise ValueError("The model cannot have two types with the same name")
            self.__types = types
        else:
            self.__types = set()

    @property
    def associations(self) -> set[Association]:
        return self.__associations

    @associations.setter
    def associations(self, associations: set[Association]):
        # Check no duplicate names
        if associations is not None:
            # Get a list of names from the elements
            names = [association.name for association in associations]
            if len(names) != len(set(names)):
                raise ValueError("The model cannot have two associations with the same name")
            self.__associations = associations
        else:
            self.__associations = set()

    @property
    def generalizations(self) -> set[Generalization]:
        return self.__generalizations

    @generalizations.setter
    def generalizations(self, generalizations: set[Generalization]):
        if generalizations is not None:
            self.__generalizations = generalizations
        else:
            self.__generalizations = set()

    @property
    def packages(self) -> set[Package]:
        return self.__packages

    @packages.setter
    def packages(self, packages: set[Package]):
        # Check no duplicate names
        if packages is not None:
            # Get a list of names from the elements
            names = [package.name for package in packages]
            if len(names) != len(set(names)):
                raise ValueError("The model cannot have two packages with the same name")
            self.__packages = packages
        else:
            self.__packages = set()

    @property
    def constraints(self) -> set[Constraint]:
        return self.__constraints

    @constraints.setter
    def constraints(self, constraints: set[Constraint]):
        # Check no duplicate names
        if constraints is not None:
            # Get a list of names from the elements
            names = [constraint.name for constraint in constraints]
            if len(names) != len(set(names)):
                raise ValueError("The model cannot have two constraints with the same name")
            self.__constraints = constraints
        else:
            self.__constraints = set()

    def get_classes(self) -> set[Class]:
        return {element for element in self.types if isinstance(element, Class)}
    
    def get_class_by_name(self, class_name: str) -> Class:
        return next((element for element in self.types if isinstance(element, Class) and element.name == class_name), None)

    def get_associations(self) -> set[Association]:
        return {element for element in self.elements if isinstance(element, Association)}

    def get_packages(self) -> set[Package]:
        return {element for element in self.elements if isinstance(element, Package)}
