import string
from besser.BUML.metamodel.structural import NamedElement, Property, Type, Association

class AttributeLink(NamedElement):
    """An attribute link is a named slot in an instance, which holds the value of an attribute
    
    Args:
        name (str): the name of the attribute link
        value (DataValue): the value of the attribute.
        attribute (Property): the attribute or property from the structural metamodel.
    
    Attributes:
        name (str): Inherited from NamedElement, represents the name of the association.
        value (DataValue): the value of the attribute.
        attribute (Property): the attribute or property from the structural metamodel.
    """

    def __init__(self, name, value: "DataValue", attribute: Property):
        super().__init__(name)
        self.value: DataValue = value
        self.attribute: Property = attribute

    @property
    def value(self) -> "DataValue":
        """DataValue: Get value of the attribute."""
        return self.__value

    @value.setter
    def value(self, value: "DataValue"):
        """bool: Set the value of the attribute."""
        self.__value = value

    @property
    def attribute(self) -> Property:
        """DataValue: Get the attribute."""
        return self.__attribute

    @value.setter
    def attribute(self, attribute: Property):
        """bool: Set the attribute."""
        self.__attribute = attribute

    def __repr__(self) -> str:
        return f'Attribute Link({self.name}, {self.value}, {self.attribute})'

class Instance(NamedElement):
    """The instance defines an entity to which a set of operations can be applied and which has a state that stores the effects of the operations.
    
    Attributes:
        classifier: classname of the instance
        slots: list of properties
        ownedLink: the set of Links that are owned by the Instance.
        linkEnd: The set of LinkEnds of the connected Links that are attached to the Instance.
           """

    def __init__(self):
        self.classifier:Type = None
        self.slots = None
        self.ownedLink = None
        self.linkEnd = None



class Object(Instance):
    """ An object is an instance that originates from a class.
     Attributes:
          slots: list of properties
          linkEnd: The set of LinkEnds of the connected Links that are attached to the
Instance.
    """
    def __init__(self):
        super().__init__()
        self.slots: list[AttributeLink] = []
        self.linkEnd: list[LinkEnd] = []
    def add_slot(self, slot):
        """ Method to add attribute link to slots"""
        self.slots.append(slot)

    def add_to_link_end(self, link):
        """ Method to add link to linkend"""

        self.linkEnd.append(link)

    def get_slots(self):
        """ slots: Method to retrieve slots"""

        return self.slots

    def __repr__(self):
         """str: printing the value of object."""
         toRet =""
         for s in self.slots:
            toRet = toRet + str(s) +'\n'
         return toRet


class DataValue(Instance):
    """ An DataValue is an instance with no identity.
         Attributes:
              value: value of the property
    Instance.
        """
    def __init__(self,value):
        super().__init__()
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
    args:
    name : name of the link
    ins: Instance of the link
    Attributes:
        associationEnd: property pointing to link
        instance : instance of link
            """
    def __init__(self,name,ins):
        super().__init__(name)
        self.associationEnd: Property = None
        self.instance= ins

class Link(NamedElement):
    """ A link contains list of linkends.
    attributes:
    connection: list of linkends
    """
    def __init__(self):
        self.connection : list[LinkEnd] = []
    def add_to_connection(self,linkEnd):
        """Method to add linkend"""
        self.connection.append(linkEnd)
