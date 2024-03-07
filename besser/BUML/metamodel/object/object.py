import string
from besser.BUML.metamodel.structural import NamedElement, Property, Type, Association

class AttributeLink():
    """An attribute link is a named slot in an instance, which holds the value of an attribute
    
    Args:
        name (str): the name of the attribute link
        value (DataValue): the value of the attribute.
        attribute (Property): the attribute or property from the structural metamodel.
    
    Attributes:
        name (str): inherited from NamedElement, represents the name of the attribute link.
        value (DataValue): the value of the attribute.
        attribute (Property): the attribute or property from the structural metamodel.
    """

    def __init__(self, value: "DataValue", attribute: Property):
        self.value: DataValue = value
        self.attribute: Property = attribute

    @property
    def value(self) -> "DataValue":
        """DataValue: Get value of the attribute."""
        return self.__value

    @value.setter
    def value(self, value: "DataValue"):
        """DataValue: Set the value of the attribute."""
        self.__value = value

    @property
    def attribute(self) -> Property:
        """Property: Get the attribute."""
        return self.__attribute

    @value.setter
    def attribute(self, attribute: Property):
        """Property: Set the attribute."""
        self.__attribute = attribute

    def __repr__(self) -> str:
        return f'Attribute Link({self.name}, {self.value}, {self.attribute})'

class Instance(NamedElement):
    """The instance defines an entity to which a set of operations can be applied and which has a state that stores the effects of the operations.
    
    Args:
        name (str): the name of the instance
        classifier (Type): the classifier of the instance. It could be for example a Class or a PrimitiveDataType of the structural metamodel.

    Attributes:
        name (str): inherited from NamedElement, represents the name of the instance.
        classifier (Type): the classifier of the instance. It could be for example a Class or a PrimitiveDataType of the structural metamodel.
    """

    def __init__(self, name: str, classifier: Type):
        super().__init__(name)
        self.classifier: Type = classifier

    @property
    def classifier(self) -> Type:
        """Type: Get the classifier."""
        return self.__classifier

    @classifier.setter
    def classifier(self, classifier: Type):
        """Type: Set the classifier."""
        self.__classifier = classifier

class Object(Instance):
    """ An object is an instance that originates from a class.
    
    Args:
        name (str): the name of the object instance
        classifier (Type): the classifier of the object instance. It could be for example a Class or a PrimitiveDataType of the structural metamodel.
        slots (list[AttributeLink]): list of properties of the instance
        link_ends (list[LinkEnd]): The set of LinkEnds of the connected Links that are attached to the Instance.
    
    Attributes:
        name (str): inherited from NamedElement, represents the name of the object instance.
        classifier (Type): Inherited from Instance, represents the classifier of the object.
        slots (list[AttributeLink]): list of properties of the instance
        link_ends (list[LinkEnd]): The set of LinkEnds of the connected Links that are attached to the Instance.
    """
    def __init__(self, name: str, classifier: Type, slots: list[AttributeLink] = []):
        super().__init__(name, classifier)
        self.slots: list[AttributeLink] = slots

    @property
    def slots(self) -> list[AttributeLink]:
        """list[AttributeLink]: Get the slots."""
        return self.__slots

    @slots.setter
    def slots(self, slots: list[AttributeLink]):
        """list[AttributeLink]: Set the slots."""
        self.__slots = slots

    def add_slot(self, slot: AttributeLink):
        """ Method to add attribute link to slots"""
        self.slots.append(slot)

    def __repr__(self):
         return f'Object({self.name}, {self.classifier}, {self.slots}, {self.link_ends})'


class DataValue(Instance):
    """ An DataValue represent the value of a property or attribute of an Object.
    
    Args:
        classifier (Type): the classifier of the DataValue. It could be for example a Class or a PrimitiveDataType of the structural metamodel.
        value: value of the property Instance.
    
    Attributes:
        classifier (Type): Inherited from Instance, represents the classifier of the DataValue instance.
        value: value of the property Instance.
    """

    def __init__(self, classifier: Type, value, name=""):
        super().__init__(name, classifier)
        self.__value = value

    @property
    def value(self):
       """Method to retrieve Value"""
       return self.__value

    @value.setter
    def value(self, val):
        """Method to set Value"""
        self.__value = val

class LinkEnd(NamedElement):
    """ A link end is an end point of a link.

    Args:
        name (str): the name of the LinkEnd
        association_end (Property): the end represeted by the LinkEnd
        object (Object): the object pointed to by the LinkEnd
    
    Attributes:
        name (str): inherited from NamedElement, represents the name of the LinkEnd
        association_end (Property): the end of the link
        object (Object): the object pointed to by the LinkEnd
    """

    def __init__(self, name:str, association_end: Property, object: Object):
        super().__init__(name)
        self.association_end: Property = association_end
        self.object: Object = object

    @property
    def association_end(self):
       """Property: Method to retrieve the association end"""
       return self.__association_end

    @association_end.setter
    def association_end(self, association_end: Property):
        """Property: Method to set the association end"""
        self.__association = association_end

    @property
    def object(self):
       """Object: Method to retrieve the object"""
       return self.__association

    @object.setter
    def object(self, object: Object):
        """Object: Method to set the object"""
        self.__object = object

class Link(NamedElement):
    """ A link represent a relationship between objects.

    Args:
        name (str): the name of the Link
        association (Association): the Association that represents the Link
        connections: list of link ends.
    
    Attributes:
        name (str): inherited from NamedElement, represents the name of the Link
        association (Association): the Association that represents the Link
        connections: list of link ends.
    """

    def __init__(self, name: str, association: Association, connections: list[LinkEnd]):
        super().__init__(name)
        self.association: Association = association
        self.connection: list[LinkEnd] = connections
    
    @property
    def association(self):
       """Association: Method to retrieve the association"""
       return self.__association

    @association.setter
    def association(self, association: Association):
        """Association: Method to set the association"""
        self.__association = association

    @property
    def connections(self):
       """list[LinkEnd]: Method to retrieve the connections"""
       return self.__connections

    @connections.setter
    def connections(self, connections: list[LinkEnd]):
        """list[LinkEnd]: Method to set the connections"""
        self.__connections = connections

    def add_to_connection(self,linkEnd):
        """Method to add linkend"""
        self.connection.append(linkEnd)
