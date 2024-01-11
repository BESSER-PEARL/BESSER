from abc import ABC, abstractmethod 
from typing import Any
 
# constant
UNLIMITED_MAX_MULTIPLICITY = 9999

class Element(ABC):
    pass

class NamedElement(Element):
    """The NamedElement is the Superclass of all structural elements with a name.

    Args:
        name (str): the name of the named element
        visibility: Determines the kind of visibility of the named element (public as default).
    
    Attributes:
        name (str): the name of the named element
        visibility: Determines the kind of visibility of the named element (public as default).
    """

    def __init__(self, name: str, visibility: str = "public"):
        self.name: str = name
        self.visibility = visibility

    @property
    def name(self) -> str:
        """str: Get the name of the named element."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the name of the named element."""
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

class Type(NamedElement):
    """Type is the Superclass of classes and data types in the model.

    Args:
        name (str): the name of the Type.

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the Type.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self):
        return f"Type({self.name})"

class DataType(Type):
    """Represents a data type.

    This class inherits from NamedElement and is used to model data types.

    Args:
        name (str): The name of the data type.

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the data type.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self):
        return f"DataType({self.name})"
    
class PrimitiveDataType(DataType):
    """Class representing an enumeration literal.

    This class is a subclass of NamedElement and is used to represent individual
    literals within an enumeration.

    Args:
        name (str): the name of the enumeration literal.
        owner (DataType): the owner data type of the enumeration literal.

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the enumeration literal.
        owner (DataType): Represents the owner data type of the enumeration literal.
    """    

    def __init__(self, name: str):
        super().__init__(name)

    @NamedElement.name.setter
    def name(self, name: str):
        """
        str: Set the name of the PrimitiveDataType. 
        
        Raises:
            ValueError: If an invalid primitive data type is provided.
                        Allowed values are int, float, str, bool, time, date, datetime, and timedelta.
        """
        if name not in ['int', 'float', 'str', 'bool', 'time', 'date', 'datetime', 'timedelta']:
            raise ValueError("Invalid primitive data type")
        super(PrimitiveDataType, PrimitiveDataType).name.fset(self, name)
    
    def __repr__(self):
        return f"PrimitiveDataType({self.name})"
    
class EnumerationLiteral(NamedElement):
    """Class representing a primitive data type.

    This class is a subclass of DataType and is used to represent primitive data types
    with a specified name.

    Args:
        name (str): the name of the primitive data type.
        owner (DataType): the owner data type (Enumeration) of the enumeration literal.

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the enumeration literal.
        owner (DataType): Represents the owner data type (Enumeration) of the enumeration literal.
    """
    
    def __init__(self, name: str, owner: DataType):
        super().__init__(name)
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
        return f"EnumerationLiteral({self.name})"
    
class Enumeration(DataType):
    """Class representing an enumeration.

    This class is a subclass of DataType and is used to represent enumerations
    with a specified name and a set of enumeration literals.

    Args:
        name (str): the name of the enumeration data type.
        literals (set[EnumerationLiteral]): set of enumeration literals associated with the enumeration.

    Attributes:
        name (str): Inherited from DataType, represents the name of the enumeration.
        literals (set[EnumerationLiteral]): Represents a set of enumeration literals associated with the enumeration.
    """
    
    def __init__(self, name: str, literals: set[EnumerationLiteral]):
        super().__init__(name)
        self.literals: set[EnumerationLiteral] = literals

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

    def __repr__(self):
        return f"Enumeration({self.name}, {self.literals})"
    
class TypedElement(NamedElement):
    """TypedElement is a subclass of NamedElement and is used to represent elements
    that have a specific type.

    Args:
        name (str): The name of the typed element.
        type (Type): The data type of the typed element.
        visibility: Determines the kind of visibility of the typed element (public as default).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the typed element.
        visibility: Inherited from NamedElement, represents the visibility of the typed element.
        type (Type): The data type of the typed element.
    """

    def __init__(self, name: str, type: Type, visibility: str="public"):
        super().__init__(name, visibility)
        self.type: Type = type

    @property
    def type(self) -> Type:
        return self.__type

    @type.setter
    def type(self, type: Type):
        self.__type = type

class Multiplicity:
    """Represents the multiplicity of a Property.

    It consists of a minimum and maximum value, indicating the allowed range.

    Args:
        min_multiplicity (int): The minimum multiplicity.
        max_multiplicity (int): The maximum multiplicity. Use "*" for unlimited.

    Attributes:
        min (int): The minimum multiplicity.
        max (int): The maximum multiplicity. Use "*" for unlimited.
    """

    def __init__(self, min_multiplicity: int, max_multiplicity: int):
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
            ValueError: (Invalid min multiplicity) if the minimun multiplicity is less than 0.
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
            less than minimun multiplicity.
        """
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
    """A property can represents an attribute of a class or an end of an association.

    Properties are owned by a class or an association.

    Args:
        name (str): The name of the property.
        property_type (Type): The type of the property.
        owner (Type): The type that owns the property.
        multiplicity (Multiplicity): The multiplicity of the property.
        visibility (str): The visibility of the property ('public', 'private', etc.).
        is_composite (bool): Indicates whether the property is a composite.
        is_navigable (bool): Indicates whether the property is navigable in a relationship.
        is_aggregation (bool): Indicates whether the property represents an aggregation.
        is_id (bool): Indicates whether the property is an id.

    Attributes:
        name (str): Inherited from TypedElement, represents the name of the property.
        property_type (Type): Inherited from TypedElement, represents the type of the property.
        owner (Type): The type that owns the property.
        multiplicity (Multiplicity): The multiplicity of the property.
        visibility (str): Inherited from TypedElement, represents the visibility of the property.
        is_composite (bool): Indicates whether the property is a composite.
        is_navigable (bool): Indicates whether the property is navigable in a relationship.
        is_aggregation (bool): Indicates whether the property represents an aggregation.
        is_id (bool): Indicates whether the property is an id.
    """
    
    def __init__(self, name: str, property_type: Type, owner: Type = None, multiplicity: Multiplicity = Multiplicity(1, 1), 
                 visibility: str = 'public', is_composite: bool = False, is_navigable: bool = True, is_aggregation: bool = False,
                 is_id: bool = False):
        super().__init__(name, property_type, visibility)
        self.owner: Type = owner
        self.multiplicity: Multiplicity = multiplicity
        self.is_composite: bool = is_composite
        self.is_navigable: bool = is_navigable
        self.is_aggregation: bool = is_aggregation
        self.is_id: bool = is_id

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
        """bool: Get wheter the property is composite."""
        return self.__is_composite

    @is_composite.setter
    def is_composite(self, is_composite: bool):
        """bool: Set wheter the property is composite."""
        self.__is_composite = is_composite

    @property
    def is_navigable(self) -> bool:
        """bool: Get wheter the property is navigable."""
        return self.__is_navigable

    @is_navigable.setter
    def is_navigable(self, is_navigable: bool):
        """bool: Set wheter the property is navigable."""
        self.__is_navigable = is_navigable

    @property
    def is_aggregation(self) -> bool:
        """bool: Get wheter the property represents an aggregation."""
        return self.__is_aggregation

    @is_aggregation.setter
    def is_aggregation(self, is_aggregation: bool):
        """bool: Set wheter the property represents an aggregation."""
        self.__is_aggregation = is_aggregation
    
    @property
    def is_id(self) -> bool:
        """bool: Get wheter the property is an id."""
        return self.__is_id

    @is_id.setter
    def is_id(self, is_id: bool):
        """bool: Set wheter the property is an id."""
        self.__is_id = is_id

    def __repr__(self):
        return f'Property({self.name}, {self.visibility}, {self.type}, {self.multiplicity}, is_composite={self.is_composite}, is_id={self.is_id})'

class Class(Type):
    """Represents a class in a modeling context.

    A Class is a type that defines a blueprint for objects. It can have attributes, associations,
    and generalizations with other classes.

    Args:
        name (str): The name of the class.
        attributes (set[Property]): The set of attributes associated with the class.
        is_abstract (bool): Indicates whether the class is abstract.

    Attributes:
        name (str): Inherited from Type, represents the name of the class.
        attributes (set[Property]): The set of attributes associated with the class.
        is_abstract (bool): Indicates whether the class is abstract.
        __associations (set[Association]): Set of associations involving the class.
        __generalizations (set[Generalization]): Set of generalizations involving the class.
    """

    def __init__(self, name: str, attributes: set[Property], is_abstract: bool= False):
        super().__init__(name)
        self.is_abstract: bool = is_abstract
        self.attributes: set[Property] = attributes
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
        """
        if attributes is not None:
            names = [attribute.name for attribute in attributes]
            if len(names) != len(set(names)):
                raise ValueError("A class cannot have two attributes with the same name")
            for attribute in attributes:
                attribute.owner = self
            self.__attributes = attributes
        else:
            self.__attributes = set()

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
                raise ValueError("A class cannot have two attributes with the same name")
        attribute.owner = self
        self.attributes.add(attribute)
    
    @property
    def is_abstract(self) -> bool:
        """bool: Get wheter the class is abstract."""
        return self.__is_abstract

    @is_abstract.setter
    def is_abstract(self, is_abstract: bool):
        """bool: Set wheter the class is abstract."""
        self.__is_abstract = is_abstract

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
    
    def __repr__(self):
        return f'Class({self.name}, {self.attributes})'

class Association(NamedElement):
    """Represents an association between classes.

    An Association defines a relationship between classes and is composed of two or more ends,
    each associated with a class. An association must have more than one end.

    Args:
        name (str): The name of the association.
        ends (set[Property]): The set of ends related to the association.
        
    Attributes:
        name (str): Inherited from NamedElement, represents the name of the association.
        ends (set[Property]): The set of ends related to the association.
    """

    def __init__(self, name: str, ends: set[Property]):
        super().__init__(name)
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
        return f'Association({self.name}, {self.ends})'
    
class BinaryAssociation(Association):
    """Represents a binary association between two classes.

    A BinaryAssociation is a specialized form of Association that specifically involves
    two ends, each associated with a class. It enforces constraints on the association,
    such as having exactly two ends. Exactly two ends are required 

    Args:
        name (str): The name of the binary association.
        ends (set[Property]): The set of ends related to the binary association. 

    Attributes:
        name (str): Inherited from Association, represents the name of the binary association.
        ends (set[Property]): Inherited from NamedElement, represents the set of ends related to the binary association.
    """

    def __init__(self, name: str, ends: set[Property]):
        super().__init__(name, ends)

    @Association.ends.setter
    def ends(self, ends: set[Property]):
        """set[Property]: Set the ends of the association.
        
        Raises:
            ValueError: if the associaiton ends are not exactly two, or if both ends are tagged as agregation, or 
            if both ends are tagged as composition.
        """
        if len(ends) != 2:
            raise ValueError("A binary must have exactly two ends")
        if list(ends)[0].is_aggregation == True and list(ends)[1].is_aggregation == True:
            raise ValueError("The aggregation attribute cannot be tagged at both ends")
        if list(ends)[0].is_composite == True and list(ends)[1].is_composite == True:
            raise ValueError("The composition attribute cannot be tagged at both ends")
        super(BinaryAssociation, BinaryAssociation).ends.fset(self, ends)

    def __repr__(self):
        return f'BinaryAssociation({self.name}, {self.ends})'
    
class AssociationClass(Class):
    # Class that has an association nature
    """An AssociationClass is a class that that has an association nature.
    It inherits from Class and is associated with an underlying Association.

    Args:
        name (str): The name of the association class.
        attributes (set[Property]): The set of attributes associated with the association class.
        association (Association): The underlying association linked to the association class.

    Attributes:
        name (str): Inherited from Class, represents the name of the association class.
        attributes (set[Property]): Inherited from Class, represents the set of attributes associated with the association class.
        association (Association): The underlying association linked to the association class.
    """

    def __init__(self, name: str, attributes: set[Property], association: Association):
        super().__init__(name, attributes)
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
        return f'AssociationClass({self.name}, {self.attributes}, {self.association})'

class Generalization(Element):
    """Represents a generalization relationship between two classes.

    A Generalization is a relationship between two classes, where one class (specific)
    inherits attributes and behaviors from another class (general).

    Args:
        general (Class): The general (parent) class in the generalization relationship.
        specific (Class): The specific (child) class in the generalization relationship.
    
    Attributes:
        general (Class): The general (parent) class in the generalization relationship.
        specific (Class): The specific (child) class in the generalization relationship.
    """

    def __init__(self, general: Class, specific: Class):
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
            raise ValueError("you cannot have your own parent")
        if hasattr(self, "specific"):
            self.specific._delete_generalization(generalization=self)
        specific._add_generalization(generalization=self)
        self.__specific = specific

    def __repr__(self):
        return f'Generalization({self.general}, {self.specific})'

class GeneralizationSet(NamedElement):
    """Represents a set of generalization relationships.

    Args:
        name (str): The name of the generalization set.
        generalizations (set[Generalization]): The set of generalization relationships in the set.
        is_disjoint (bool): Indicates whether the set is disjoint (instances cannot belong to more than one class in the set).
        is_complete (bool): Indicates whether the set is complete (every instance of the superclass must belong to a subclass).

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the generalization set.
        generalizations (set[Generalization]): The set of generalization relationships in the set.
        is_disjoint (bool): Indicates whether the set is disjoint (instances cannot belong to more than one class in the set).
        is_complete (bool): Indicates whether the set is complete (every instance of the superclass must belong to a subclass).
    """

    def __init__(self, name: str, generalizations: set[Generalization], is_disjoint: bool, is_complete: bool):
        super().__init__(name)
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
        return f'GeneralizationSet({self.name}, {self.generalizations}, is_disjoint={self.is_disjoint}, is_complete={self.is_complete})'
    
class Package(NamedElement):
    """A Package is a grouping mechanism that allows organizing and managing a set of classes.

    Attributes:
        name (str): The name of the package.
        classes (set[Class]): The set of classes contained in the package.
    
    Attributes:
        name (str): Inherited from NamedElement, represents the name of the package.
        classes (set[Class]): The set of classes contained in the package.
    """

    def __init__(self, name: str, classes: set[Class]):
        super().__init__(name)
        self.classes: set[Class] = classes

    @property
    def classes(self) -> set[Class]:
        """set[Class]: Get the classes contained in the package."""
        return self.__classes

    @classes.setter
    def classes(self, classes: set[Class]):
        """set[Class]: Set the classes contained in the package."""
        self.__classes = classes

    def __repr__(self):
        return f'Package({self.name}, {self.classes})'
    
class Constraint(NamedElement):
    """A Constraint is a statement that restricts or defines conditions on the behavior,
    structure, or other aspects of the modeled system.

    Args:
        name (str): The name of the constraint.
        context (Class): The class to which the constraint is associated.
        expression (str): The expression or condition defined by the constraint.
        language (str): The language in which the constraint expression is written.

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the constraint.
        context (Class): The class to which the constraint is associated.
        expression (str): The expression or condition defined by the constraint.
        language (str): The language in which the constraint expression is written.
    """
        
    def __init__(self, name: str, context: Class, expression: Any, language: str):
        super().__init__(name)
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
        return f'Constraint({self.name},{self.context.name},{self.language},{self.expression})'

class DomainModel(NamedElement):
    """A domain model is the root element that comprises a number of types, associations, 
    generalizations, packages, constraints, and others.

    Args:
        name (str): The name of the domain model.
        types (set[Type]): The set of types (classes and datatypes) in the domain model.
        associations (set[Association]): The set of associations in the domain model.
        generalizations (set[Generalization]): The set of generalizations in the domain model.
        enumerations (set[Enumeration]): The set of enumerations in the domain model.
        packages (set[Package]): The set of packages in the domain model.
        constraints (set[Constraint]): The set of constraints in the domain model.

    Attributes:
        name (str): Inherited from NamedElement, represents the name of the domain model.
        types (set[Type]): The set of types (classes and datatypes) in the domain model.
        associations (set[Association]): The set of associations in the domain model.
        generalizations (set[Generalization]): The set of generalizations in the domain model.
        enumerations (set[Enumeration]): The set of enumerations in the domain model.
        packages (set[Package]): The set of packages in the domain model.
        constraints (set[Constraint]): The set of constraints in the domain model.
    """

    def __init__(self, name: str, types: set[Type] = None, associations: set[Association] = None, generalizations: set[Generalization] = None, 
                 enumerations: set[Enumeration] = None, packages: set[Package] = None, constraints: set[Constraint] = None):
        super().__init__(name)
        self.types: set[Type] = types
        self.packages: set[Package] = packages
        self.constraints: set[Constraint] = constraints
        self.associations: set[Association] = associations
        self.enumerations: set[Enumeration] = enumerations
        self.generalizations: set[Generalization] = generalizations

    @property
    def types(self) -> set[Type]:
        """set[Type]: Get the set of types in the domain model."""
        return self.__types

    @types.setter
    def types(self, types: set[Type]):
        """
        set[Type]: Set the set of types in the domain model.
        
        Raises:
            ValueError: if there are two types with the same name.
        """
        if types is not None:
            names = [type.name for type in types]
            if len(names) != len(set(names)):
                raise ValueError("The model cannot have two types with the same name")
            self.__types = types
        else:
            self.__types = set()

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
            names = [association.name for association in associations]
            if len(names) != len(set(names)):
                raise ValueError("The model cannot have two associations with the same name")
            self.__associations = associations
        else:
            self.__associations = set()

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

    @property
    def enumerations(self) -> set[Enumeration]:
        """set[Enumeration]: Get the set of enumerations in the domain model."""
        return self.__enumerations

    @enumerations.setter
    def enumerations(self, enumerations: set[Enumeration]):
        """
        set[Enumeration]: Set the set of enumerations in the domain model.
        
        Raises:
            ValueError: if there are two enumerations with the same name.
        """
        if enumerations is not None:
            names = [enumeration.name for enumeration in enumerations]
            if len(names) != len(set(names)):
                raise ValueError("The model cannot have two enumerations with the same name")
            self.__enumerations = enumerations
        else:
            self.__enumerations = set()

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
            names = [package.name for package in packages]
            if len(names) != len(set(names)):
                raise ValueError("The model cannot have two packages with the same name")
            self.__packages = packages
        else:
            self.__packages = set()

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

    def get_classes(self) -> set[Class]:
        """set[Class]: Get all classes within the domain model."""
        return {element for element in self.types if isinstance(element, Class)}
    
    def get_class_by_name(self, class_name: str) -> Class:
        """Class: Gets a class by name."""
        return next((element for element in self.types if isinstance(element, Class) and element.name == class_name), None)
    
    def classes_sorted_by_inheritance(self) -> list[Class]:
        """list[Class]: Get the list of classes ordered by inheritance."""
        classes: set[Class] = self.get_classes()
        ordered_classes: list = []
        while len(classes) != 0:
            for cl in classes:
                if len(cl.parents()) == 0 or all(parent in ordered_classes for parent in cl.parents()):
                    ordered_classes.append(cl)
            classes.difference_update(ordered_classes)
        return ordered_classes
    
    def __repr__(self):
        return f'Package({self.name}, {self.types}, {self.associations}, {self.generalizations}, {self.enumerations}, {self.packages}, {self.constraints})'